#!/usr/bin/env python3
"""
HoloMe Tri-View 3D Bust Reconstructor & Anatomical Auto-Rigger.

Takes a 3-panel character sheet (Front, Left, Right) on a light background and:
1. Automatically crops and extracts clean foreground silhouettes for Front, Left, Right.
2. Runs Depth Anything V2 on Apple Silicon MPS / CPU across all 3 views.
3. Runs MediaPipe FaceLandmarker (478 landmarks) on the Front view.
4. Fuses Tri-View Silhouettes + Calibrated Depth Maps into a high-resolution,
   structured 3D sculpted bust (head + ears + beard + hoodie collar & drawstrings).
5. Computes anatomical facial animation weights (jaw pivot, lips/visemes, eyelids,
   eyebrows, head-vs-shoulders neck blend) and tri-view photo detail/edge maps.
6. Exports compact binary + JSON assets for the Three.js Hologram Renderer.
"""

import argparse
import json
import math
import struct
import sys
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import torch
from PIL import Image
from scipy.ndimage import binary_fill_holes, gaussian_filter, gaussian_filter1d
from transformers import AutoImageProcessor, AutoModelForDepthEstimation


# MediaPipe FaceMesh landmark indices
LIP_OUTER = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146]
LIP_INNER_UPPER = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308]
LIP_INNER_LOWER = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308]
LEFT_EYE_UPPER = [33, 246, 161, 160, 159, 158, 157, 173, 133]
LEFT_EYE_LOWER = [33, 7, 163, 144, 145, 153, 154, 155, 133]
RIGHT_EYE_UPPER = [362, 398, 384, 385, 386, 387, 388, 466, 263]
RIGHT_EYE_LOWER = [362, 382, 381, 380, 374, 373, 390, 249, 263]
LEFT_BROW = [70, 63, 105, 66, 107, 55, 65, 52, 53, 46]
RIGHT_BROW = [300, 293, 334, 296, 336, 285, 295, 282, 283, 276]
FACE_OVAL_LOWER = [172, 136, 150, 149, 176, 148, 152, 377, 400, 378, 379, 365, 397]


def detect_three_panels(img_bgr: np.ndarray) -> list[np.ndarray]:
    """Detect and crop the 3 rectangular panels (Front, Left, Right) from the sheet.

    Kept for backward compatibility: bordered detection with an equal-thirds fallback.
    """
    panels = detect_bordered_panels(img_bgr)
    if panels is not None:
        return panels
    return split_thirds(img_bgr, *detect_frame_rows(img_bgr))


def split_thirds(img_bgr: np.ndarray, y0: int, y1: int) -> list[np.ndarray]:
    """Split a sheet into 3 equal-width panels with a small margin."""
    H, W = img_bgr.shape[:2]
    pw = W // 3
    m = max(4, int(W * 0.012))
    return [
        img_bgr[y0:y1, m : pw - m].copy(),
        img_bgr[y0:y1, pw + m : 2 * pw - m].copy(),
        img_bgr[y0:y1, 2 * pw + m : W - m].copy(),
    ]


def estimate_background(img_bgr: np.ndarray, patch: int = 12) -> np.ndarray:
    """Median color of the four image corners."""
    p = patch
    samples = np.concatenate([
        img_bgr[:p, :p].reshape(-1, 3),
        img_bgr[:p, -p:].reshape(-1, 3),
        img_bgr[-p:, :p].reshape(-1, 3),
        img_bgr[-p:, -p:].reshape(-1, 3),
    ], axis=0)
    return np.median(samples, axis=0).astype(np.float32)


def split_by_subjects(img_bgr: np.ndarray) -> list[np.ndarray] | None:
    """Split a borderless sheet by finding the three largest foreground subjects.

    All three crops share one vertical range (top of the highest head to the lowest
    common bottom), so the skull-top-to-bottom span stays consistent across views.
    Returns None when three subjects cannot be found.
    """
    H, W = img_bgr.shape[:2]
    bg = estimate_background(img_bgr)
    diff = np.linalg.norm(img_bgr.astype(np.float32) - bg[None, None, :], axis=2)
    fg = (diff > 28.0).astype(np.uint8)
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    num, labels, stats, _ = cv2.connectedComponentsWithStats(fg, connectivity=8)
    if num <= 3:
        return None
    areas = stats[1:, cv2.CC_STAT_AREA]
    order = np.argsort(areas)[::-1][:3]
    if len(order) < 3 or areas[order[2]] < 0.01 * H * W:
        return None
    boxes = []
    for idx in order:
        x, y, w, h = stats[idx + 1, :4]
        boxes.append((int(x), int(y), int(x + w), int(y + h)))
    boxes.sort(key=lambda b: b[0])
    # Subjects must not overlap horizontally
    for a, b in zip(boxes, boxes[1:]):
        if a[2] > b[0]:
            return None

    top = max(0, min(b[1] for b in boxes) - int(0.03 * H))
    bottom = min(b[3] for b in boxes)
    if bottom - top < 0.3 * H:
        return None

    margin = max(16, int(0.02 * W))
    panels = []
    for i, (x0, _, x1, _) in enumerate(boxes):
        left_limit = 0 if i == 0 else (boxes[i - 1][2] + x0) // 2
        right_limit = W if i == 2 else (x1 + boxes[i + 1][0]) // 2
        cx0 = max(left_limit, x0 - margin)
        cx1 = min(right_limit, x1 + margin)
        panels.append(img_bgr[top:bottom, cx0:cx1].copy())
    return panels


def load_sheet_panels(img_bgr: np.ndarray, layout: str = "auto") -> tuple[list[np.ndarray], str]:
    """Return (panels, method) using bordered detection, subject split, or equal thirds."""
    if layout in ("auto", "bordered"):
        panels = detect_bordered_panels(img_bgr)
        if panels is not None:
            return panels, "bordered"
        if layout == "bordered":
            raise RuntimeError("Bordered layout requested but 6 panel border lines were not found.")
    if layout in ("auto", "subjects"):
        panels = split_by_subjects(img_bgr)
        if panels is not None:
            return panels, "subjects"
        if layout == "subjects":
            raise RuntimeError("Subject layout requested but three separate subjects were not found.")
    return split_thirds(img_bgr, *detect_frame_rows(img_bgr)), "thirds"


def compose_separate_views(paths: list[Path], target_h: int = 960) -> list[np.ndarray]:
    """Turn three separate photos (front, left, right) into panels with matching framing.

    Each photo is cropped to its subject horizontally, from just above the top of the
    head down to the photo's bottom edge, then resized to a shared height. Photos must
    be framed the same way (same zoom, same bottom cut) and have a plain background.
    """
    panels = []
    for p in paths:
        img = cv2.imread(str(p))
        if img is None:
            raise FileNotFoundError(f"Could not read view image: {p}")
        H, W = img.shape[:2]
        bg = estimate_background(img)
        diff = np.linalg.norm(img.astype(np.float32) - bg[None, None, :], axis=2)
        fg = (diff > 28.0).astype(np.uint8)
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        ys, xs = np.where(fg > 0)
        if len(ys) < 0.02 * H * W:
            raise RuntimeError(f"Could not separate the subject from the background in {p}")
        margin = max(16, int(0.04 * W))
        x0 = max(0, int(xs.min()) - margin)
        x1 = min(W, int(xs.max()) + margin)
        y0 = max(0, int(ys.min()) - int(0.03 * H))
        crop = img[y0:H, x0:x1]
        scale = target_h / float(crop.shape[0])
        crop = cv2.resize(crop, (max(16, int(round(crop.shape[1] * scale))), target_h), interpolation=cv2.INTER_AREA)
        panels.append(crop)
    return panels


def detect_frame_rows(img_bgr: np.ndarray) -> tuple[int, int]:
    """Find the top and bottom panel frame lines (rows that are >70% dark across the sheet)."""
    H = img_bgr.shape[0]
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    row_dark = (gray < 235).mean(axis=1)
    rows = np.arange(H)
    top_candidates = np.where((rows > H * 0.05) & (rows < H * 0.20) & (row_dark > 0.70))[0]
    bot_candidates = np.where((rows > H * 0.80) & (rows < H * 0.96) & (row_dark > 0.70))[0]
    y0 = int(top_candidates[0]) + 3 if len(top_candidates) else int(H * 0.10)
    y1 = int(bot_candidates[-1]) - 3 if len(bot_candidates) else int(H * 0.92)
    return y0, y1


def detect_bordered_panels(img_bgr: np.ndarray) -> list[np.ndarray] | None:
    """Detect and crop 3 bordered panels. Returns None when 6 border lines are not found."""
    H, W = img_bgr.shape[:2]
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    y0, y1 = detect_frame_rows(img_bgr)

    # Vertical panel borders: look at thin columns in [y0+10:y1-10] that are nearly uniformly grey/border
    # Specifically, look for vertical borders adjacent to white gutters
    col_white = (gray[y0 + 10 : y1 - 10, :] > 245).mean(axis=0)
    # Gutters are columns that are >97% pure white
    is_gutter = col_white > 0.97

    # Find contiguous non-gutter regions? Better: find vertical lines with dark border (gray < 235 > 0.92 of height)
    col_border = (gray[y0 + 5 : y1 - 5, :] < 235).mean(axis=0)
    # Exclude interior subject columns by checking std (a straight border line has low intensity std!)
    col_std = gray[y0 + 5 : y1 - 5, :].std(axis=0)
    border_cols = np.where((col_border > 0.90) & (col_std < 35.0))[0]

    # Group contiguous border columns
    groups = []
    if len(border_cols):
        cur = [int(border_cols[0])]
        for c in border_cols[1:]:
            if c <= cur[-1] + 3:
                cur.append(int(c))
            else:
                groups.append((cur[0], cur[-1]))
                cur = [int(c)]
        groups.append((cur[0], cur[-1]))

    # We expect 6 border lines (left/right of each of the 3 panels)
    if len(groups) >= 6:
        # Pair them: (0,1), (2,3), (4,5)
        panels = []
        for idx in range(3):
            x0 = groups[2 * idx][1] + 2
            x1 = groups[2 * idx + 1][0] - 2
            panels.append(img_bgr[y0:y1, x0:x1].copy())
        # Reject degenerate detections (slivers instead of portrait panels)
        if all(p.shape[1] > 0.12 * W and p.shape[0] > 0.3 * H for p in panels):
            return panels
    return None


def extract_foreground_mask(panel_bgr: np.ndarray) -> np.ndarray:
    """Extract clean foreground mask from white-background studio portrait panel."""
    H, W = panel_bgr.shape[:2]
    # Estimate background color from top corners
    bg_samples = np.concatenate([
        panel_bgr[:15, :15].reshape(-1, 3),
        panel_bgr[:15, -15:].reshape(-1, 3),
    ], axis=0)
    bg_color = np.median(bg_samples, axis=0).astype(np.float32)

    diff = np.linalg.norm(panel_bgr.astype(np.float32) - bg_color[None, None, :], axis=2)
    # Seed floodfill from top border and upper left/right margins
    bg_candidate = (diff < 22.0).astype(np.uint8)

    # Connected components of background touching top/side borders
    num_labels, labels = cv2.connectedComponents(bg_candidate, connectivity=4)
    border_labels = set()
    border_labels.update(labels[0, :].tolist())
    border_labels.update(labels[: int(H * 0.75), 0].tolist())
    border_labels.update(labels[: int(H * 0.75), -1].tolist())
    border_labels.discard(0)

    is_bg = np.isin(labels, list(border_labels))
    fg = (~is_bg).astype(np.uint8)

    # Keep largest connected component (the person)
    num_fg, fg_labels, stats, _ = cv2.connectedComponentsWithStats(fg, connectivity=8)
    if num_fg > 1:
        largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        fg = (fg_labels == largest).astype(np.uint8)

    # Fill internal holes (e.g., white highlight on forehead or drawstring)
    fg = binary_fill_holes(fg > 0).astype(np.uint8)

    # Soft morphological cleanup
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, kernel)
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, kernel)
    return fg


def run_depth_anything(panels_bgr: list[np.ndarray], device: str) -> list[np.ndarray]:
    """Run Depth Anything V2 Base (or Small fallback) on all 3 portrait panels."""
    model_id = "depth-anything/Depth-Anything-V2-Small-hf"
    print(f"[build_bust] Loading {model_id} on {device}...", file=sys.stderr)
    processor = AutoImageProcessor.from_pretrained(model_id)
    model = AutoModelForDepthEstimation.from_pretrained(model_id).to(device)
    model.eval()

    depths = []
    for idx, bgr in enumerate(panels_bgr):
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        inputs = processor(images=pil_img, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            pred = outputs.predicted_depth
        # Resize to panel resolution
        H, W = bgr.shape[:2]
        pred_resized = torch.nn.functional.interpolate(
            pred.unsqueeze(1), size=(H, W), mode="bicubic", align_corners=False
        ).squeeze().cpu().numpy().astype(np.float32)
        depths.append(pred_resized)
    return depths


def detect_face_landmarks(front_bgr: np.ndarray, model_path: Path) -> np.ndarray:
    """Detect 478 MediaPipe 3D face landmarks in normalized front-panel coordinates."""
    rgb = cv2.cvtColor(front_bgr, cv2.COLOR_BGR2RGB)
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    BaseOptions = mp.tasks.BaseOptions
    FaceLandmarker = mp.tasks.vision.FaceLandmarker
    FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(model_path)),
        running_mode=VisionRunningMode.IMAGE,
        num_faces=1,
    )
    with FaceLandmarker.create_from_options(options) as landmarker:
        result = landmarker.detect(mp_img)

    if not result.face_landmarks:
        raise RuntimeError("No face detected in FRONT panel of character sheet!")

    lms = result.face_landmarks[0]
    arr = np.array([[lm.x, lm.y, lm.z] for lm in lms], dtype=np.float32)
    return arr


def row_extents(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return per-row (min_col, max_col, valid) for a binary mask."""
    H, W = mask.shape
    mins = np.full(H, -1.0, dtype=np.float32)
    maxs = np.full(H, -1.0, dtype=np.float32)
    valid = np.zeros(H, dtype=bool)
    for r in range(H):
        cols = np.where(mask[r] > 0)[0]
        if len(cols) >= 3:
            mins[r] = float(cols[0])
            maxs[r] = float(cols[-1])
            valid[r] = True
    return mins, maxs, valid


def compute_detail_maps(panel_bgr: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute enhanced perceptual luminance + fine edge/texture detail map for hologram rendering."""
    gray = cv2.cvtColor(panel_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0

    # Local contrast enhancement (CLAHE) + subtle unsharp detail so beard hairs, eyes, brows, ear folds & drawstrings pop!
    clahe = cv2.createCLAHE(clipLimit=2.8, tileGridSize=(8, 8))
    enhanced = clahe.apply((gray * 255).astype(np.uint8)).astype(np.float32) / 255.0
    blur_usm = cv2.GaussianBlur(gray, (0, 0), 2.0)
    unsharp = np.clip(gray + (gray - blur_usm) * 1.45, 0.0, 1.0)

    # Blend original luminance, CLAHE local contrast, and unsharp fine detail
    lum = 0.38 * gray + 0.34 * enhanced + 0.28 * unsharp
    # Gently stretch foreground dynamic range so skin highlights shine & dark eyes/creases stay crisp
    fg_vals = lum[mask > 0]
    if len(fg_vals) > 100:
        p5, p97 = float(np.percentile(fg_vals, 5)), float(np.percentile(fg_vals, 97))
        lum = np.clip((lum - p5) / (p97 - p5 + 1e-5), 0.0, 1.0)

    # Multi-scale Sobel edge magnitude (captures eyebrows, eyelashes, nose, beard hairs, ears, drawstrings)
    gx1 = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy1 = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    edge1 = np.sqrt(gx1 * gx1 + gy1 * gy1)

    blur = cv2.GaussianBlur(gray, (5, 5), 1.2)
    gx2 = cv2.Sobel(blur, cv2.CV_32F, 1, 0, ksize=3)
    gy2 = cv2.Sobel(blur, cv2.CV_32F, 0, 1, ksize=3)
    edge2 = np.sqrt(gx2 * gx2 + gy2 * gy2)

    edge = np.clip(edge1 * 1.85 + edge2 * 1.15, 0.0, 1.0)
    eroded = cv2.erode(mask, np.ones((5, 5), np.uint8), iterations=1)
    edge *= eroded.astype(np.float32)

    return np.clip(lum, 0.0, 1.0), np.clip(edge, 0.0, 1.0)


def sample_bilinear(img: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Fast bilinear sampling on 2D numpy array."""
    H, W = img.shape[:2]
    xf = np.clip(x, 0.0, W - 1.001)
    yf = np.clip(y, 0.0, H - 1.001)
    x0 = xf.astype(np.int32)
    y0 = yf.astype(np.int32)
    x1 = np.minimum(x0 + 1, W - 1)
    y1 = np.minimum(y0 + 1, H - 1)
    wx = xf - x0
    wy = yf - y0
    if img.ndim == 2:
        return (
            img[y0, x0] * (1 - wx) * (1 - wy)
            + img[y0, x1] * wx * (1 - wy)
            + img[y1, x0] * (1 - wx) * wy
            + img[y1, x1] * wx * wy
        )
    else:
        wx = wx[..., None]
        wy = wy[..., None]
        return (
            img[y0, x0] * (1 - wx) * (1 - wy)
            + img[y0, x1] * wx * (1 - wy)
            + img[y1, x0] * (1 - wx) * wy
            + img[y1, x1] * wx * wy
        )


CANONICAL_PANEL_H = 468
FACE_EDGE_TARGET_P50 = 0.53
FACE_EDGE_MAX_GAIN = 2.4


def to_canonical_height(panel: np.ndarray, target_h: int = CANONICAL_PANEL_H) -> np.ndarray:
    H, W = panel.shape[:2]
    if H == target_h:
        return panel
    interp = cv2.INTER_AREA if H > target_h else cv2.INTER_CUBIC
    return cv2.resize(panel, (max(16, int(round(W * target_h / float(H)))), target_h), interpolation=interp)


def mask_stats(mask: np.ndarray) -> dict:
    H, W = mask.shape
    coverage = float(mask.mean())
    touches_top = bool(mask[:2, :].any())
    touches_sides_upper = bool(mask[: int(H * 0.4), :2].any() or mask[: int(H * 0.4), -2:].any())
    return {"coverage": round(coverage, 3), "touchesTop": touches_top, "touchesSidesUpper": touches_sides_upper}


def write_qa_preview(out_path: Path, panels: list[np.ndarray], masks: list[np.ndarray], lms: np.ndarray):
    """Side-by-side panels with silhouette outlines, front landmarks, and facing hints."""
    target_h = 640
    tiles = []
    captions = ["FRONT: faces camera", "LEFT: nose points LEFT", "RIGHT: nose points RIGHT"]
    for i, (p, m) in enumerate(zip(panels, masks)):
        tile = p.copy()
        contours, _ = cv2.findContours((m > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(tile, contours, -1, (255, 0, 255), max(2, tile.shape[1] // 250))
        if i == 0:
            H, W = tile.shape[:2]
            for idx in (33, 133, 362, 263, 1, 13, 14, 61, 291, 152, 468, 473):
                if idx < len(lms):
                    cv2.circle(tile, (int(lms[idx, 0] * W), int(lms[idx, 1] * H)), max(3, W // 160), (0, 200, 0), -1)
        scale = target_h / float(tile.shape[0])
        tile = cv2.resize(tile, (int(tile.shape[1] * scale), target_h), interpolation=cv2.INTER_AREA)
        bar = np.full((56, tile.shape[1], 3), 255, np.uint8)
        cv2.putText(bar, captions[i], (10, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (30, 30, 30), 2, cv2.LINE_AA)
        tiles.append(np.vstack([tile, bar]))
        tiles.append(np.full((target_h + 56, 12, 3), 255, np.uint8))
    cv2.imwrite(str(out_path), np.hstack(tiles[:-1]))


def build_bust(
    sheet_path: Path | None,
    out_dir: Path,
    model_path: Path,
    n_theta: int = 240,
    n_y: int = 280,
    view_paths: list[Path] | None = None,
    layout: str = "auto",
    swap_profiles: bool = False,
    mirror_profile: str | None = None,
    panels_only: bool = False,
):
    out_dir.mkdir(parents=True, exist_ok=True)

    if view_paths:
        print("[build_bust] Composing panels from 3 separate view images...", file=sys.stderr)
        p_front, p_left, p_right = compose_separate_views(view_paths)
        layout_used = "separate"
        source = [str(p.resolve()) for p in view_paths]
    else:
        img_bgr = cv2.imread(str(sheet_path))
        if img_bgr is None:
            raise FileNotFoundError(f"Could not read character sheet: {sheet_path}")
        print("[build_bust] Detecting 3 panels (Front, Left, Right)...", file=sys.stderr)
        (p_front, p_left, p_right), layout_used = load_sheet_panels(img_bgr, layout)
        source = str(sheet_path.resolve())
    print(f"[build_bust] Panel layout: {layout_used}", file=sys.stderr)

    if swap_profiles:
        p_left, p_right = p_right, p_left
    if mirror_profile == "left":
        p_right = cv2.flip(p_left, 1)
    elif mirror_profile == "right":
        p_left = cv2.flip(p_right, 1)

    # Keep full-resolution panels: the front one is the LTX opening frame (driving-frame), and
    # all three are the inputs for image-to-3D models such as Tripo P2 Multi View.
    hires = {}
    for name, panel in (("front", p_front), ("left", p_left), ("right", p_right)):
        hires[name] = str(out_dir / f"panel_{name}_hires.png")
        cv2.imwrite(hires[name], panel)

    # Edge and detail maps are computed per pixel, and the shaders were tuned on 468 px tall
    # panels. Resample every panel to that height so sheets of any resolution bake alike.
    p_front, p_left, p_right = [to_canonical_height(p) for p in (p_front, p_left, p_right)]

    # Save cropped panels for inspection
    cv2.imwrite(str(out_dir / "panel_front.png"), p_front)
    cv2.imwrite(str(out_dir / "panel_left.png"), p_left)
    cv2.imwrite(str(out_dir / "panel_right.png"), p_right)

    # Extract silhouettes
    m_front = extract_foreground_mask(p_front)
    m_left = extract_foreground_mask(p_left)
    m_right = extract_foreground_mask(p_right)

    # Detect MediaPipe landmarks on Front
    print("[build_bust] Extracting 478 MediaPipe facial landmarks...", file=sys.stderr)
    lms = detect_face_landmarks(p_front, model_path)
    Hf, Wf = p_front.shape[:2]

    # --- QA report for agents: silhouettes, framing, frontal pose ---
    warnings = []
    stats = {"front": mask_stats(m_front), "left": mask_stats(m_left), "right": mask_stats(m_right)}
    for name, s in stats.items():
        if s["coverage"] < 0.12:
            warnings.append(f"{name}: subject covers only {s['coverage']:.0%} of the panel (silhouette probably broken or subject too small)")
        if s["coverage"] > 0.85:
            warnings.append(f"{name}: subject covers {s['coverage']:.0%} of the panel (background probably not separated)")
        if s["touchesTop"]:
            warnings.append(f"{name}: silhouette touches the panel top (head cut off or background leak)")
        if s["touchesSidesUpper"]:
            warnings.append(f"{name}: silhouette touches the panel sides near the head (background leak or tight crop)")
    eye_l = lms[33, :2] * [Wf, Hf]
    eye_r = lms[263, :2] * [Wf, Hf]
    nose = lms[1, :2] * [Wf, Hf]
    eye_dist = float(np.linalg.norm(eye_r - eye_l)) + 1e-6
    frontal_offset = float(abs(nose[0] - 0.5 * (eye_l[0] + eye_r[0])) / eye_dist)
    if frontal_offset > 0.12:
        warnings.append(f"front: face is turned (nose offset {frontal_offset:.2f} of eye distance); regenerate a straight-on front view")
    face_xs = lms[:, 0] * Wf
    face_ys = lms[:, 1] * Hf
    face_box = [float(face_xs.min()), float(face_ys.min()), float(face_xs.max()), float(face_ys.max())]
    write_qa_preview(out_dir / "qa_panels.png", [p_front, p_left, p_right], [m_front, m_left, m_right], lms)
    if panels_only:
        print(json.dumps({
            "status": "ok",
            "layout": layout_used,
            "qaPreview": str(out_dir / "qa_panels.png"),
            "panels": hires,
            "silhouettes": stats,
            "frontalOffset": round(frontal_offset, 3),
            "warnings": warnings,
        }))
        return
    Hl, Wl = p_left.shape[:2]
    Hr, Wr = p_right.shape[:2]

    # Run Depth Anything V2 on all 3 views
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    d_front_raw, d_left_raw, d_right_raw = run_depth_anything([p_front, p_left, p_right], device)

    # Normalize each foreground depth map into [0..1] (1 = closest to camera)
    def norm_depth(d, m):
        vals = d[m > 0]
        lo, hi = np.percentile(vals, 2), np.percentile(vals, 99.5)
        dn = np.clip((d - lo) / (hi - lo + 1e-6), 0.0, 1.0)
        return dn * (m > 0).astype(np.float32)

    d_front = norm_depth(d_front_raw, m_front)
    d_left = norm_depth(d_left_raw, m_left)
    d_right = norm_depth(d_right_raw, m_right)

    # Save debug depth visualizations
    cv2.imwrite(str(out_dir / "debug_depth_front.png"), (d_front * 255).astype(np.uint8))
    cv2.imwrite(str(out_dir / "debug_depth_left.png"), (d_left * 255).astype(np.uint8))
    cv2.imwrite(str(out_dir / "debug_depth_right.png"), (d_right * 255).astype(np.uint8))

    # Compute enhanced luminance and edge detail maps for all 3 views
    lum_f, edge_f = compute_detail_maps(p_front, m_front)
    lum_l, edge_l = compute_detail_maps(p_left, m_left)
    lum_r, edge_r = compute_detail_maps(p_right, m_right)

    # --- Coordinate Alignment across Front, Left, Right ---
    # Find vertical top of skull and bottom in each view
    f_min, f_max, f_val = row_extents(m_front)
    l_min, l_max, l_val = row_extents(m_left)
    r_min, r_max, r_val = row_extents(m_right)

    yf_top = int(np.where(f_val)[0][0])
    yf_bot = int(np.where(f_val)[0][-1])
    yl_top = int(np.where(l_val)[0][0])
    yl_bot = int(np.where(l_val)[0][-1])
    yr_top = int(np.where(r_val)[0][0])
    yr_bot = int(np.where(r_val)[0][-1])

    # Vertical scale: match skull-top to panel-bottom across all 3 views
    hf_span = float(yf_bot - yf_top)
    hl_span = float(yl_bot - yl_top)
    hr_span = float(yr_bot - yr_top)

    # Front horizontal center from MediaPipe nose bridge (landmark 6) & mid-eyes
    x_center_f = float(0.5 * (lms[33, 0] + lms[263, 0]) * Wf * 0.5 + lms[6, 0] * Wf * 0.5)

    # World scale: total bust height = 2.0 units -> Y in [-1.0, +1.0] (top of skull = +1.0, bottom = -1.0)
    # 1 world unit in pixel coordinates:
    px_per_unit_f = hf_span / 2.0
    px_per_unit_l = hl_span / 2.0
    px_per_unit_r = hr_span / 2.0

    # Find Z-center of cranium in LEFT (facing left: nose = min col, back = max col)
    # and RIGHT (facing right: nose = max col, back = min col)
    # Use upper head rows (15% to 45% down from skull top) to define cranial Z-origin
    def cranial_z_center(mins, maxs, y_top, span, flip_sign):
        r0 = int(y_top + 0.15 * span)
        r1 = int(y_top + 0.45 * span)
        mids = 0.5 * (mins[r0:r1] + maxs[r0:r1])
        return float(np.median(mids))

    zl_center_px = cranial_z_center(l_min, l_max, yl_top, hl_span, -1)
    zr_center_px = cranial_z_center(r_min, r_max, yr_top, hr_span, +1)

    # Sample vertical grid v in [0..1] (0 = very top of skull +1.0, 1 = bottom of hoodie -1.0)
    # Use cosine spacing near the very top crown so the dome cap has dense, smooth rings!
    u_lin = np.linspace(0.0, 1.0, n_y, dtype=np.float32)
    # Slightly concentrate rings near crown and face
    v_grid = 0.5 * (1.0 - np.cos(math.pi * u_lin)) * 0.18 + u_lin * 0.82
    v_grid[0] = 0.002  # tiny offset from exact pole to avoid degenerate radius
    v_grid[-1] = 0.996

    y_world = 1.0 - 2.0 * v_grid  # +1.0 (crown) down to -1.0 (bottom of bust)

    # Per-row pixel coordinates in each panel
    yf_px = yf_top + v_grid * hf_span
    yl_px = yl_top + v_grid * hl_span
    yr_px = yr_top + v_grid * hr_span

    # Interpolate silhouette bounds in world units for each height row!
    def interp_extent(ext_arr, valid_arr, y_px):
        valid_idx = np.where(valid_arr)[0].astype(np.float32)
        return np.interp(y_px, valid_idx, ext_arr[valid_arr]).astype(np.float32)

    xf_min_w = (interp_extent(f_min, f_val, yf_px) - x_center_f) / px_per_unit_f
    xf_max_w = (interp_extent(f_max, f_val, yf_px) - x_center_f) / px_per_unit_f

    # In LEFT view (facing left): +Z is to the LEFT (smaller column index)
    zl_front_w = (zl_center_px - interp_extent(l_min, l_val, yl_px)) / px_per_unit_l
    zl_back_w = (zl_center_px - interp_extent(l_max, l_val, yl_px)) / px_per_unit_l  # negative

    # In RIGHT view (facing right): +Z is to the RIGHT (larger column index)
    zr_front_w = (interp_extent(r_max, r_val, yr_px) - zr_center_px) / px_per_unit_r
    zr_back_w = (interp_extent(r_min, r_val, yr_px) - zr_center_px) / px_per_unit_r  # negative

    # Combine Left & Right profile silhouettes (smooth average for rock-solid nose/brow/beard/skull profile!)
    z_front_w = 0.5 * (zl_front_w + zr_front_w)
    z_back_w = 0.5 * (zl_back_w + zr_back_w)

    # Smooth per-row silhouette bounds
    xf_min_w = gaussian_filter1d(xf_min_w, sigma=1.1)
    xf_max_w = gaussian_filter1d(xf_max_w, sigma=1.1)
    z_front_w = gaussian_filter1d(z_front_w, sigma=0.9)
    z_back_w = gaussian_filter1d(z_back_w, sigma=1.2)

    # Separate true skull/cheek width from ear protrusion!
    # In the Front view, ears stick out horizontally around y in [nose_y - 0.12 .. eye_y + 0.08].
    # We compute a smoothly interpolated skull-only width across the ear band, and add the ear
    # protrusion ONLY around the anatomical ear azimuth (theta ≈ ±95 deg, i.e., cos_t ≈ -0.10)!
    eye_y_est = 1.0 - 2.0 * ((0.5 * (lms[33, 1] + lms[263, 1]) * Hf - yf_top) / hf_span)
    nose_y_est = 1.0 - 2.0 * ((lms[1, 1] * Hf - yf_top) / hf_span)
    ear_top_y = eye_y_est + 0.09
    ear_bot_y = nose_y_est - 0.14

    def smooth_over_ears(arr_w):
        out = arr_w.copy()
        in_ear = (y_world <= ear_top_y) & (y_world >= ear_bot_y)
        if np.any(in_ear) and np.any(~in_ear):
            # Fit smooth curve connecting temple above ear to jawline below ear
            out_interp = np.interp(v_grid, v_grid[~in_ear], arr_w[~in_ear])
            # Blend gently so temple->cheek transition is sleek and anatomical
            out[in_ear] = 0.72 * out_interp[in_ear] + 0.28 * arr_w[in_ear]
        return gaussian_filter1d(out, sigma=2.2)

    xf_min_skull = smooth_over_ears(xf_min_w)
    xf_max_skull = smooth_over_ears(xf_max_w)
    ear_delta_pos = np.maximum(0.0, xf_max_w - xf_max_skull)
    ear_delta_neg = np.maximum(0.0, xf_min_skull - xf_min_w)

    # --- Reconstruct Sculpted 3D Surface on Cylindrical Grid (n_y, n_theta) ---
    # theta = 0 -> straight forward (+Z)
    # theta = +pi/2 -> viewer right / subject left (+X)
    # theta = -pi/2 -> viewer left / subject right (-X)
    # theta = ±pi -> back of skull (-Z)
    thetas = np.linspace(-math.pi, math.pi, n_theta, endpoint=False, dtype=np.float32)
    sin_t = np.sin(thetas)[None, :]  # (1, n_theta)
    cos_t = np.cos(thetas)[None, :]  # (1, n_theta)

    # Localized angular lobe for anatomical ear placement (centered at ±98 deg, narrow lobe)
    ear_lobe_pos = np.exp(-(((thetas - 1.70) / 0.34) ** 2))[None, :]
    ear_lobe_neg = np.exp(-(((thetas + 1.70) / 0.34) ** 2))[None, :]

    # Per-row center & quadrant semi-axes from exact 3-view silhouettes
    xc = 0.5 * (xf_min_skull + xf_max_skull)[:, None]
    rx_pos = np.maximum(0.015, xf_max_skull - 0.5 * (xf_min_skull + xf_max_skull))[:, None] + ear_delta_pos[:, None] * ear_lobe_pos
    rx_neg = np.maximum(0.015, 0.5 * (xf_min_skull + xf_max_skull) - xf_min_skull)[:, None] + ear_delta_neg[:, None] * ear_lobe_neg

    # Graceful slight shoulder flare at bottom 15% of hoodie for cinematic bust silhouette
    shoulder_flare = 1.0 + 0.12 * np.power(np.clip((v_grid - 0.80) / 0.20, 0.0, 1.0), 1.6)[:, None]
    rx_pos = rx_pos * shoulder_flare
    rx_neg = rx_neg * shoulder_flare

    # Note: keep cranial spine zc slightly behind midline of head so nose doesn't shift the whole skull forward!
    zc_base = np.clip(0.5 * (z_front_w + z_back_w), -0.18, 0.12)
    zc = gaussian_filter1d(zc_base, sigma=16.0)[:, None]

    # Soften bottom 7% Z-front spike where hanging drawstrings touch bottom panel border
    bot_damp = np.clip((v_grid - 0.91) / 0.09, 0.0, 1.0)
    z_front_clean = z_front_w * (1.0 - 0.22 * bot_damp) + np.minimum(z_front_w, 0.50) * (0.22 * bot_damp)

    rz_pos = np.maximum(0.015, z_front_clean[:, None] - zc)
    rz_neg = np.maximum(0.015, zc - z_back_w[:, None])

    rx = np.where(sin_t >= 0, rx_pos, rx_neg)
    rz = np.where(cos_t >= 0, rz_pos, rz_neg)

    # Step A: Base Visual-Hull Ellipsoidal Surface (already matches Front, Left, Right outlines!)
    X = xc + rx * sin_t
    Z = zc + rz * cos_t
    Y = np.repeat(y_world[:, None], n_theta, axis=1)

    # Step B: FRONT DEPTH SCULPTING (Carve real eye sockets, narrow protruding nose, cheeks, lips, beard, hoodie folds!)
    # For every front-facing point (cos_t > 0), sample Depth Anything V2 on FRONT at (X, Y)
    u_f = x_center_f + X * px_per_unit_f
    v_f = np.repeat(yf_px[:, None], n_theta, axis=1)

    df_sample = sample_bilinear(d_front, u_f, v_f)
    mf_sample = sample_bilinear(m_front.astype(np.float32), u_f, v_f)

    # Calibrate Front Depth Map row-by-row AND globally against the TRUE Left/Right profile silhouette!
    # For each row, the maximum depth on the front face (e.g., nose tip, brow, lips, chin)
    # corresponds to z_front_w(y), while the left/right edges (sin_t = ±1) sit at zc(y)!
    front_mask_cols = cos_t[0] > 0.0
    Z_sculpt_front = Z.copy()

    # Compute smooth global depth scale from nose-tip to cheek/ear difference
    # Find nose tip landmark (index 1) and side cheek/ear depth
    nose_x_px = lms[1, 0] * Wf
    nose_y_px = lms[1, 1] * Hf
    d_nose = float(sample_bilinear(d_front, np.array([nose_x_px]), np.array([nose_y_px]))[0])
    # Cheek/temple near outer face
    d_side_l = float(sample_bilinear(d_front, np.array([lms[234, 0] * Wf]), np.array([lms[234, 1] * Hf]))[0])
    d_side_r = float(sample_bilinear(d_front, np.array([lms[454, 0] * Wf]), np.array([lms[454, 1] * Hf]))[0])
    d_side = 0.5 * (d_side_l + d_side_r)

    # True physical Z span from nose tip to ear-plane (from our profile views!)
    nose_row = int(np.clip(np.argmin(np.abs(yf_px - nose_y_px)), 0, n_y - 1))
    true_nose_z = float(z_front_w[nose_row] - zc[nose_row, 0])
    global_depth_scale = true_nose_z / max(d_nose - d_side, 0.15)
    # Clamp scale to physically plausible range
    global_depth_scale = float(np.clip(global_depth_scale, 0.55, 1.35))

    for r in range(n_y):
        # Sample depth across front hemisphere
        df_row = df_sample[r, front_mask_cols]
        if df_row.max() < 0.05:
            continue
        # Edge depth at left/right silhouette of this row
        d_edge = 0.5 * (df_sample[r, np.argmin(np.abs(thetas + math.pi / 2))] +
                        df_sample[r, np.argmin(np.abs(thetas - math.pi / 2))])
        d_peak = float(np.max(df_row))

        # True protrusion at this row from Left/Right side profiles
        row_rz = float(rz_pos[r, 0])
        if d_peak - d_edge > 0.04:
            # Blend row-exact scaling with global scaling for consistency without row-to-row wobble
            row_scale = row_rz / (d_peak - d_edge)
            scale = 0.65 * row_scale + 0.35 * global_depth_scale
            z_from_depth = float(zc[r, 0]) + (df_sample[r] - d_edge) * scale
            # Anchor peak cleanly to z_front_w[r] so side profile is 100% faithful!
            peak_err = float(z_front_w[r]) - float(np.max(z_from_depth[front_mask_cols]))
            z_from_depth = z_from_depth + peak_err * np.clip(cos_t[0], 0.0, 1.0)
        else:
            z_from_depth = Z[r]

        # Smoothly blend depth-sculpted Z with base ellipsoid Z toward the +/-90deg side seam
        # Weight is strong (0.92) across the face (-70deg..+70deg), fading to 0 at ±90deg
        w_f = np.power(np.clip(cos_t[0], 0.0, 1.0), 0.75) * 0.90
        # Keep top crown dome smooth (top 6% of skull)
        crown_fade = float(np.clip((v_grid[r] - 0.03) / 0.07, 0.0, 1.0))
        w_f *= crown_fade
        Z_sculpt_front[r] = (1.0 - w_f) * Z[r] + w_f * z_from_depth

    Z = Z_sculpt_front

    # Step C: SIDE DEPTH SCULPTING (Carve real 3D ears, cheekbones, jawline, side hood folds from LEFT & RIGHT views!)
    # Project current (Z, Y) into LEFT (+X side) and RIGHT (-X side) panels
    ul_px = zl_center_px - Z * px_per_unit_l
    vl_px = np.repeat(yl_px[:, None], n_theta, axis=1)
    dl_sample = sample_bilinear(d_left, ul_px, vl_px)

    ur_px = zr_center_px + Z * px_per_unit_r
    vr_px = np.repeat(yr_px[:, None], n_theta, axis=1)
    dr_sample = sample_bilinear(d_right, ur_px, vr_px)

    # Extract local relief (high-pass + mid-pass depth detail) on sides so ears & jawbone pop in 3D!
    dl_smooth = gaussian_filter(dl_sample, sigma=(4.0, 12.0))
    dr_smooth = gaussian_filter(dr_sample, sigma=(4.0, 12.0))
    # Subtle relief displacement along +/-X around the ears & side of head
    # Only apply in the side-facing zones (|sin_t| > 0.35)
    w_left_side = np.power(np.clip(sin_t, 0.0, 1.0), 1.8)
    w_right_side = np.power(np.clip(-sin_t, 0.0, 1.0), 1.8)
    crown_fade_col = np.clip((v_grid - 0.05) / 0.08, 0.0, 1.0)[:, None]

    # Relief amplitude (~0.09 world units for ear pinna & side hood relief)
    X += w_left_side * crown_fade_col * np.clip(dl_sample - dl_smooth, -0.35, 0.45) * 0.14
    X -= w_right_side * crown_fade_col * np.clip(dr_sample - dr_smooth, -0.35, 0.45) * 0.14

    # Step D: Subtle anatomical smoothing (preserving crisp facial contours while eliminating any noise)
    # Wrap-aware smoothing in theta!
    def smooth_cylindrical(arr, sigma_y=0.9, sigma_t=1.2):
        # Pad in theta periodically
        pad = 8
        padded = np.concatenate([arr[:, -pad:], arr, arr[:, :pad]], axis=1)
        sm = gaussian_filter(padded, sigma=(sigma_y, sigma_t))
        return sm[:, pad:-pad]

    X = smooth_cylindrical(X, sigma_y=0.8, sigma_t=1.1)
    Z = smooth_cylindrical(Z, sigma_y=0.7, sigma_t=1.0)

    # Re-enforce smooth pole closure at very top of skull (row 0)
    crown_x = float(np.mean(X[0]))
    crown_z = float(np.mean(Z[0]))
    for r in range(4):
        alpha = r / 4.0
        X[r] = (1.0 - alpha) * crown_x + alpha * X[r]
        Z[r] = (1.0 - alpha) * crown_z + alpha * Z[r]

    # --- Compute Accurate 3D Surface Normals ---
    # Tangent along theta (periodic) and along v (downward)
    dX_dt = 0.5 * (np.roll(X, -1, axis=1) - np.roll(X, 1, axis=1))
    dY_dt = 0.5 * (np.roll(Y, -1, axis=1) - np.roll(Y, 1, axis=1))
    dZ_dt = 0.5 * (np.roll(Z, -1, axis=1) - np.roll(Z, 1, axis=1))

    dX_dv = np.gradient(X, axis=0)
    dY_dv = np.gradient(Y, axis=0)
    dZ_dv = np.gradient(Z, axis=0)

    # Cross product (t_v x t_theta) -> outward normal
    # Since v goes downward (-Y) and theta goes -pi..+pi (+X at front),
    # t_theta at front is (+1, 0, 0), t_v is (0, -1, 0), so t_theta x t_v = (0, 0, -1) -> flip sign!
    Nx = dY_dv * dZ_dt - dZ_dv * dY_dt
    Ny = dZ_dv * dX_dt - dX_dv * dZ_dt
    Nz = dX_dv * dY_dt - dY_dv * dX_dt
    n_len = np.sqrt(Nx * Nx + Ny * Ny + Nz * Nz) + 1e-8
    Nx /= n_len
    Ny /= n_len
    Nz /= n_len

    # Verify outward orientation at front nose (should have Nz > 0)
    front_col = int(np.argmin(np.abs(thetas)))
    if Nz[n_y // 3, front_col] < 0:
        Nx, Ny, Nz = -Nx, -Ny, -Nz

    # --- Tri-View Blended Photo Luminance & Edge Detail Baking ---
    u_f = x_center_f + X * px_per_unit_f
    v_f = np.repeat(yf_px[:, None], n_theta, axis=1)
    ul_px = zl_center_px - Z * px_per_unit_l
    vl_px = np.repeat(yl_px[:, None], n_theta, axis=1)
    ur_px = zr_center_px + Z * px_per_unit_r
    vr_px = np.repeat(yr_px[:, None], n_theta, axis=1)

    lf_s = sample_bilinear(lum_f, u_f, v_f)
    ef_s = sample_bilinear(edge_f, u_f, v_f)
    mf_s = sample_bilinear(m_front.astype(np.float32), u_f, v_f)

    ll_s = sample_bilinear(lum_l, ul_px, vl_px)
    el_s = sample_bilinear(edge_l, ul_px, vl_px)
    ml_s = sample_bilinear(m_left.astype(np.float32), ul_px, vl_px)

    lr_s = sample_bilinear(lum_r, ur_px, vr_px)
    er_s = sample_bilinear(edge_r, ur_px, vr_px)
    mr_s = sample_bilinear(m_right.astype(np.float32), ur_px, vr_px)

    # View-angle blending weights based on outward normal + silhouette validity
    wf = np.power(np.clip(Nz, 0.0, 1.0), 1.6) * np.clip(mf_s, 0.0, 1.0)
    wl = np.power(np.clip(Nx, 0.0, 1.0), 1.4) * np.clip(ml_s, 0.0, 1.0)
    wr = np.power(np.clip(-Nx, 0.0, 1.0), 1.4) * np.clip(mr_s, 0.0, 1.0)

    # Back-of-head smooth fallback (uses side profiles smoothly wrapped around back)
    wb = np.power(np.clip(-Nz, 0.0, 1.0), 1.5) * 0.45
    back_lum = 0.5 * (ll_s + lr_s) * 0.85 + 0.12
    back_edge = 0.5 * (el_s + er_s) * 0.5

    w_sum = wf + wl + wr + wb + 1e-6
    lum_baked = (wf * lf_s + wl * ll_s + wr * lr_s + wb * back_lum) / w_sum
    edge_baked = (wf * ef_s + wl * el_s + wr * er_s + wb * back_edge) / w_sum

    # --- Anatomical Facial Landmarks in World 3D Coordinates ---
    def lm_to_world_xy(idx: int) -> tuple[float, float]:
        x_w = (lms[idx, 0] * Wf - x_center_f) / px_per_unit_f
        y_w = 1.0 - 2.0 * ((lms[idx, 1] * Hf - yf_top) / hf_span)
        return float(x_w), float(y_w)

    def lm_poly_xy(indices: list[int]) -> np.ndarray:
        return np.array([lm_to_world_xy(i) for i in indices], dtype=np.float32)

    mouth_center_x, mouth_center_y = lm_to_world_xy(14)  # lower inner lip center
    upper_lip_x, upper_lip_y = lm_to_world_xy(13)
    mouth_left_x, mouth_left_y = lm_to_world_xy(61)
    mouth_right_x, mouth_right_y = lm_to_world_xy(291)
    chin_x, chin_y = lm_to_world_xy(152)
    nose_x, nose_y = lm_to_world_xy(1)
    left_eye_pts = lm_poly_xy(LEFT_EYE_UPPER)
    left_eye_low = lm_poly_xy(LEFT_EYE_LOWER)
    right_eye_pts = lm_poly_xy(RIGHT_EYE_UPPER)
    right_eye_low = lm_poly_xy(RIGHT_EYE_LOWER)
    left_brow_pts = lm_poly_xy(LEFT_BROW)
    right_brow_pts = lm_poly_xy(RIGHT_BROW)

    mouth_half_w = max(0.06, 0.5 * abs(mouth_right_x - mouth_left_x))
    mouth_slit_y = 0.5 * (upper_lip_y + mouth_center_y)

    # Sample exact 3D surface Z at any (x, y) on the front hemisphere
    front_only = cos_t > 0.3
    def sample_front_z(wx, wy):
        d2 = (X - wx) ** 2 + (Y - wy) ** 2 + np.where(front_only, 0.0, 100.0)
        rr, cc = np.unravel_index(int(np.argmin(d2)), X.shape)
        return float(Z[rr, cc])

    mouth_center_z = sample_front_z(mouth_center_x, mouth_slit_y)

    # Exact iris / pupil centers from MediaPipe refined landmarks (468 = viewer-left eye, 473 = viewer-right eye)
    if len(lms) > 473:
        li_x, li_y = lm_to_world_xy(468)
        ri_x, ri_y = lm_to_world_xy(473)
    else:
        li_x, li_y = float(np.mean(left_eye_pts[:, 0])), float(0.5 * (np.mean(left_eye_pts[:, 1]) + np.mean(left_eye_low[:, 1])))
        ri_x, ri_y = float(np.mean(right_eye_pts[:, 0])), float(0.5 * (np.mean(right_eye_pts[:, 1]) + np.mean(right_eye_low[:, 1])))
    li_z = sample_front_z(li_x, li_y)
    ri_z = sample_front_z(ri_x, ri_y)

    # Anatomical TMJ jaw pivot (slightly below eye level, near ear plane z ≈ -0.04)
    eye_y_avg = float(0.5 * (np.mean(left_eye_pts[:, 1]) + np.mean(right_eye_pts[:, 1])))
    jaw_pivot = [0.0, float(0.55 * nose_y + 0.45 * eye_y_avg), -0.04]

    # Neck / head rotation pivot (below chin, top of hoodie collar)
    neck_pivot_y = float(chin_y - 0.16)
    neck_pivot = [0.0, neck_pivot_y, -0.06]

    # --- Compute Per-Vertex Anatomical Animation Masks ---
    # 1. Head vs Hoodie-Shoulders Weight (headW: 1.0 = full head rotation, 0.0 = anchored hoodie shoulders)
    # Transition smoothly around the collar/neck zone so the head turns naturally inside the hoodie!
    collar_y_center = chin_y - 0.14
    head_w = np.clip((Y - (collar_y_center - 0.18)) / 0.30, 0.0, 1.0)
    # Smoothstep
    head_w = head_w * head_w * (3.0 - 2.0 * head_w)
    # Keep front chin & beard moving 100% with head even if beard hangs low
    beard_protect = np.exp(-((X / 0.32) ** 2 + np.maximum(0.0, collar_y_center - Y) ** 2 / 0.03)) * np.clip(cos_t, 0.0, 1.0)
    head_w = np.clip(np.maximum(head_w, beard_protect * 0.95), 0.0, 1.0)

    # 2. Inner-lip slit curve y_slit(x) across mouth width for clean upper/lower separation!
    inner_up_xy = lm_poly_xy(LIP_INNER_UPPER)
    inner_lo_xy = lm_poly_xy(LIP_INNER_LOWER)
    slit_x = inner_up_xy[:, 0]
    slit_y = 0.5 * (inner_up_xy[:, 1] + inner_lo_xy[:, 1])
    sort_idx = np.argsort(slit_x)
    slit_x_sorted = slit_x[sort_idx]
    slit_y_sorted = slit_y[sort_idx]
    y_slit_at_X = np.interp(X, slit_x_sorted, slit_y_sorted)

    # Jaw weight (jawW): vertices below the mouth slit on the lower face/chin/goatee/jawline
    # Hard anatomical split right across the inner mouth slit (zero floating halfway vertices!),
    # smoothly blended outside the mouth corners across the cheeks!
    dist_below_slit = y_slit_at_X - Y  # positive below mouth slit
    inside_mouth_x = np.exp(-((X - mouth_center_x) / (mouth_half_w * 1.18)) ** 6)
    hard_below = (dist_below_slit >= 0.0).astype(np.float32)
    soft_below = 1.0 / (1.0 + np.exp(-np.clip(dist_below_slit * 16.0, -20.0, 20.0)))
    below_mask = inside_mouth_x * hard_below + (1.0 - inside_mouth_x) * soft_below

    # Horizontal envelope for jaw (strong across mouth/chin, tapers smoothly toward ears & back)
    jaw_x_env = np.exp(-((X - mouth_center_x) / (mouth_half_w * 2.9)) ** 2)
    # Vertical bottom envelope (fade out cleanly onto neck/hoodie below chin/beard)
    jaw_bot_y = chin_y - 0.12
    jaw_bot_env = np.clip((Y - (jaw_bot_y - 0.16)) / 0.18, 0.0, 1.0)
    jaw_bot_env = jaw_bot_env * jaw_bot_env * (3.0 - 2.0 * jaw_bot_env)
    # Front-hemisphere guard (back of neck doesn't move with jaw)
    jaw_front_env = np.clip((cos_t + 0.15) / 0.65, 0.0, 1.0)

    jaw_w = np.clip(below_mask * jaw_x_env * jaw_bot_env * jaw_front_env, 0.0, 1.0)

    # 3. Lip region weight (lipW: positive on upper lip, negative on lower lip! Encodes both mask & sign!)
    # Used for mouthFunnel (O/U), mouthWide (E/I), mouthClose (M/B/P), mouthF (F/V), upperLipRaise!
    dx_m = (X - mouth_center_x) / (mouth_half_w * 1.35)
    dy_m = (Y - y_slit_at_X) / (mouth_half_w * 0.72)
    lip_env = np.exp(-(dx_m * dx_m + dy_m * dy_m)) * np.clip(cos_t, 0.0, 1.0)
    # Signed: +1 above slit (upper lip), -1 below slit (lower lip) — crisp hard separation
    lip_sign = np.where(Y >= y_slit_at_X, 1.0, -1.0).astype(np.float32)
    lip_signed_w = np.clip(lip_env * lip_sign, -1.0, 1.0)

    # 4. Upper Eyelid Blink Weight (blinkW)
    def eye_lid_mask(upper_pts, lower_pts):
        ex_c = float(np.mean(upper_pts[:, 0]))
        ey_top = float(np.max(upper_pts[:, 1]))
        ey_bot = float(np.min(lower_pts[:, 1]))
        ey_c = 0.5 * (ey_top + ey_bot)
        ew = max(0.04, 0.5 * float(np.max(upper_pts[:, 0]) - np.min(upper_pts[:, 0])))
        eh = max(0.018, 0.5 * (ey_top - ey_bot))
        # Upper lid region: centered slightly above eye center
        dx = (X - ex_c) / (ew * 1.25)
        dy = (Y - (ey_c + eh * 0.45)) / (eh * 1.65)
        m = np.exp(-(dx * dx + dy * dy))
        # Only above lower eyelid
        above_lower = 1.0 / (1.0 + np.exp(-np.clip((Y - (ey_bot + eh * 0.15)) * 120.0, -20.0, 20.0)))
        return m * above_lower * np.clip(cos_t, 0.0, 1.0), (ex_c, ey_c, ew, eh)

    l_blink, l_eye_info = eye_lid_mask(left_eye_pts, left_eye_low)
    r_blink, r_eye_info = eye_lid_mask(right_eye_pts, right_eye_low)
    blink_w = np.clip(l_blink + r_blink, 0.0, 1.0)

    # 5. Eyebrow Raise Weight (browW)
    def brow_mask(b_pts):
        bx_c = float(np.mean(b_pts[:, 0]))
        by_c = float(np.mean(b_pts[:, 1]))
        bw = max(0.06, 0.5 * float(np.max(b_pts[:, 0]) - np.min(b_pts[:, 0])))
        dx = (X - bx_c) / (bw * 1.35)
        dy = (Y - (by_c + 0.015)) / 0.065
        return np.exp(-(dx * dx + dy * dy)) * np.clip(cos_t, 0.0, 1.0)

    brow_w = np.clip(brow_mask(left_brow_pts) + brow_mask(right_brow_pts), 0.0, 1.0)

    # Face detail normalization: the shaders were tuned on a bearded, textured face whose edge
    # channel is near saturation. Smooth skin bakes a much weaker edge channel and renders as a
    # dim mask, so lift the front of the head until the face's median edge reaches the target.
    # Faces already at or above the target are left unchanged.
    face_sel = (np.abs(X - nose_x) < 0.25) & (Y > chin_y) & (Y < eye_y_avg + 0.12) & (Nz > 0.5)
    face_edge_p50 = float(np.median(edge_baked[face_sel])) if np.any(face_sel) else 1.0
    face_detail_gain = float(np.clip(FACE_EDGE_TARGET_P50 / max(face_edge_p50, 1e-3), 1.0, FACE_EDGE_MAX_GAIN))
    if face_detail_gain > 1.01:
        front_head = head_w * np.clip((Nz + 0.2) / 0.7, 0.0, 1.0)
        edge_baked = np.clip(edge_baked * (1.0 + (face_detail_gain - 1.0) * front_head), 0.0, 1.0)
        print(f"[build_bust] Smooth face detail (median edge {face_edge_p50:.2f}); edge gain x{face_detail_gain:.2f}", file=sys.stderr)

    # Boost holographic edge definition subtly on brows and lip contours so talking & expressions pop cleanly!
    edge_baked = np.clip(edge_baked + brow_w * 0.22 + np.abs(lip_signed_w) * 0.18, 0.0, 1.0)

    # 6. Bottom hoodie dissolve & region tagging
    hoodie_w = 1.0 - head_w

    # --- Pack Compact Binary Asset (bust.bin) ---
    num_verts = n_y * n_theta
    v_norm_grid = np.repeat(v_grid[:, None], n_theta, axis=1)

    verts = np.stack([
        X, Y, Z,
        Nx, Ny, Nz,
        lum_baked, edge_baked,
        head_w, jaw_w, lip_signed_w, blink_w, brow_w,
        v_norm_grid,
    ], axis=-1).astype(np.float32)

    bin_path = out_dir / "bust.bin"
    verts.tofile(str(bin_path))

    meta = {
        "version": 1,
        "sourceSheet": source,
        "layout": layout_used,
        "faceDetailGain": round(face_detail_gain, 3),
        "frontPanel": {
            "size": [Wf, Hf],
            "faceBox": [round(v, 1) for v in face_box],
        },
        "gridRows": n_y,
        "gridCols": n_theta,
        "numVertices": num_verts,
        "strideFloats": 14,
        "attributes": [
            "x", "y", "z", "nx", "ny", "nz",
            "lum", "edge", "headW", "jawW", "lipSignedW", "blinkW", "browW", "vNorm",
        ],
        "pivots": {
            "jaw": [round(v, 5) for v in jaw_pivot],
            "neck": [round(v, 5) for v in neck_pivot],
            "mouthCenter": [round(mouth_center_x, 5), round(mouth_slit_y, 5), round(mouth_center_z, 5)],
            "mouthHalfWidth": round(mouth_half_w, 5),
            "leftEye": [round(v, 5) for v in l_eye_info],
            "rightEye": [round(v, 5) for v in r_eye_info],
            "leftIris": [round(li_x, 5), round(li_y, 5), round(li_z, 5)],
            "rightIris": [round(ri_x, 5), round(ri_y, 5), round(ri_z, 5)],
            "noseTip": [round(nose_x, 5), round(nose_y, 5), round(float(z_front_w[nose_row]), 5)],
            "chin": [round(chin_x, 5), round(chin_y, 5)],
        },
        "bounds": {
            "min": [round(float(X.min()), 4), round(float(Y.min()), 4), round(float(Z.min()), 4)],
            "max": [round(float(X.max()), 4), round(float(Y.max()), 4), round(float(Z.max()), 4)],
        },
    }
    meta_path = out_dir / "bust_meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"[build_bust] Exported {num_verts:,} vertices -> {bin_path} ({bin_path.stat().st_size / 1e6:.2f} MB)", file=sys.stderr)
    print(json.dumps({
        "status": "ok",
        "numVertices": num_verts,
        "bin": str(bin_path),
        "meta": str(meta_path),
        "layout": layout_used,
        "qaPreview": str(out_dir / "qa_panels.png"),
        "panels": {
            "front": str(out_dir / "panel_front.png"),
            "left": str(out_dir / "panel_left.png"),
            "right": str(out_dir / "panel_right.png"),
        },
        "silhouettes": stats,
        "frontalOffset": round(frontal_offset, 3),
        "faceDetailGain": round(face_detail_gain, 3),
        "warnings": warnings,
    }))


def main():
    parser = argparse.ArgumentParser(description="Build sculpted & rigged 3D hologram bust from a 3-view character sheet")
    parser.add_argument("--sheet", help="Path to 3-panel character sheet image (Front, Left, Right)")
    parser.add_argument("--front", help="Separate front view image (use with --left and --right instead of --sheet)")
    parser.add_argument("--left", help="Separate left profile image (nose pointing to the image's left edge)")
    parser.add_argument("--right", help="Separate right profile image (nose pointing to the image's right edge)")
    parser.add_argument("--layout", default="auto", choices=["auto", "bordered", "subjects", "thirds"],
                        help="How to split --sheet into panels")
    parser.add_argument("--swap-profiles", action="store_true", help="Swap the left and right panels")
    parser.add_argument("--mirror-profile", choices=["left", "right"],
                        help="Use only this profile and mirror it to make the other one")
    parser.add_argument("--out", required=True, help="Output directory for bust assets")
    parser.add_argument("--panels-only", action="store_true",
                        help="Only split the sheet, write full-resolution panels and the QA preview, then stop")
    parser.add_argument("--landmarker", default="assets/models/face_landmarker.task", help="MediaPipe face_landmarker.task path")
    parser.add_argument("--rows", type=int, default=260, help="Vertical ring count")
    parser.add_argument("--cols", type=int, default=280, help="Circumferential column count")
    args = parser.parse_args()

    views = [args.front, args.left, args.right]
    if any(views) and not all(views):
        parser.error("--front, --left and --right must be given together")
    if not args.sheet and not all(views):
        parser.error("give --sheet, or --front, --left and --right")
    if args.sheet and all(views):
        parser.error("give either --sheet or the three view images, not both")

    build_bust(
        Path(args.sheet) if args.sheet else None,
        Path(args.out),
        Path(args.landmarker),
        n_theta=args.cols,
        n_y=args.rows,
        view_paths=[Path(v) for v in views] if all(views) else None,
        layout=args.layout,
        swap_profiles=args.swap_profiles,
        mirror_profile=args.mirror_profile,
        panels_only=args.panels_only,
    )


if __name__ == "__main__":
    main()
