#!/usr/bin/env python3
"""
HoloMe External 3D Bust Importer (.glb / .gltf / .obj -> Structured & Auto-Rigged HoloMe Bust).

Imports a sculpted 3D bust (e.g. generated on Scenario.com via Hunyuan-3D / Trellis / Rodin / Tripo)
and automatically:
1. Normalizes scale & centering into HoloMe world space (Y in [-1, +1]).
2. Auto-detects front-facing orientation (testing 0/90/180/270 deg yaw) via orthographic
   rendering + MediaPipe FaceLandmarker (478 landmarks).
3. Raycasts a structured (rows x cols) cylindrical holographic contour grid directly onto
   the high-res .glb surface, baking exact normals, UV texture luminance, and edge contrast.
4. Computes exact MediaPipe-anchored anatomical rig weights (jaw pivot, hard lip slit,
   eyelids, eyebrows, irises, head-vs-shoulders blend) so all HoloMe shaders, lip-sync,
   camera eye-contact, blinks, and constellation wireframe work seamlessly.
"""

import argparse
import json
import math
import sys
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from scipy.ndimage import gaussian_filter, median_filter
from scipy.spatial import cKDTree
import trimesh


LIP_INNER_UPPER = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308]
LIP_INNER_LOWER = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308]
LEFT_EYE_UPPER = [33, 246, 161, 160, 159, 158, 157, 173, 133]
LEFT_EYE_LOWER = [33, 7, 163, 144, 145, 153, 154, 155, 133]
RIGHT_EYE_UPPER = [362, 398, 384, 385, 386, 387, 388, 466, 263]
RIGHT_EYE_LOWER = [362, 382, 381, 380, 374, 373, 390, 249, 263]
LEFT_BROW = [70, 63, 105, 66, 107, 55, 65, 52, 53, 46]
RIGHT_BROW = [300, 293, 334, 296, 336, 285, 295, 282, 283, 276]


def load_combined_mesh(glb_path: Path) -> trimesh.Trimesh:
    loaded = trimesh.load(str(glb_path), force="scene")
    if isinstance(loaded, trimesh.Trimesh):
        return loaded
    meshes = []
    for geom in loaded.dump():
        if isinstance(geom, trimesh.Trimesh) and len(geom.vertices) > 0:
            # Bake PBR/UV textures to per-vertex colors if possible before concatenating
            try:
                if hasattr(geom.visual, "to_color"):
                    geom.visual = geom.visual.to_color()
            except Exception:
                pass
            meshes.append(geom)
    if not meshes:
        raise RuntimeError(f"No valid 3D mesh geometry found in {glb_path}")
    return trimesh.util.concatenate(meshes)


def render_ortho_front(mesh: trimesh.Trimesh, size: int = 512) -> tuple[np.ndarray, float, float]:
    """Fast orthographic front projection (+Z) of mesh using dense surface sampling for MediaPipe detection."""
    # Fixed seed: the same GLB always gives the same render, landmarks and rig pivots
    pts, f_idx, colors = trimesh.sample.sample_surface(mesh, 180000, sample_color=True, seed=0)
    norms = mesh.face_normals[f_idx]
    front = norms[:, 2] > -0.05
    pts = pts[front]
    norms = norms[front]
    if colors is not None and len(colors) == len(f_idx):
        rgb = colors[front, :3].astype(np.float32)
    else:
        shade = np.clip(0.35 + 0.65 * np.maximum(0.0, norms[:, 2]), 0.0, 1.0)
        rgb = np.stack([shade * 220, shade * 210, shade * 200], axis=-1)

    # Sort back-to-front by Z so closest points paint on top
    order = np.argsort(pts[:, 2])
    pts = pts[order]
    rgb = rgb[order]

    # Map world X, Y in [-1.15, +1.15] -> [0, size-1]
    span = 2.30
    px = np.clip(((pts[:, 0] + span * 0.5) / span * (size - 1)).astype(np.int32), 0, size - 1)
    py = np.clip(((span * 0.5 - pts[:, 1]) / span * (size - 1)).astype(np.int32), 0, size - 1)

    canvas = np.full((size, size, 3), 245, dtype=np.uint8)
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            xx = np.clip(px + dx, 0, size - 1)
            yy = np.clip(py + dy, 0, size - 1)
            canvas[yy, xx] = rgb.astype(np.uint8)

    # Light median fill to bridge tiny 1px pinholes
    canvas = cv2.medianBlur(canvas, 3)
    return canvas, span, float(size)


def detect_landmarks_on_rgb(img_rgb: np.ndarray, landmarker_path: Path):
    if not landmarker_path.exists():
        return None
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
    options = mp.tasks.vision.FaceLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(landmarker_path)),
        running_mode=mp.tasks.vision.RunningMode.IMAGE,
        num_faces=1,
    )
    with mp.tasks.vision.FaceLandmarker.create_from_options(options) as lm:
        res = lm.detect(mp_img)
    if not res.face_landmarks:
        return None
    return np.array([[p.x, p.y, p.z] for p in res.face_landmarks[0]], dtype=np.float32)


def import_glb(
    glb_path: Path,
    out_dir: Path,
    landmarker_path: Path,
    n_y: int = 260,
    n_theta: int = 280,
    yaw_deg: float | None = None,
):
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[import_glb] Loading {glb_path}...", file=sys.stderr)
    mesh = load_combined_mesh(glb_path)

    # Normalize bust into HoloMe world space: Y in [-1.0, +1.0], centered in X/Z
    bounds = mesh.bounds
    center = 0.5 * (bounds[0] + bounds[1])
    height = float(bounds[1][1] - bounds[0][1])
    if height < 1e-5:
        raise RuntimeError("Mesh has zero vertical extent!")
    mesh.apply_translation(-center)
    mesh.apply_scale(2.0 / height)

    # Auto-detect front orientation (+Z) using MediaPipe FaceLandmarker across 0/90/180/270 deg
    lms = None
    span = 2.30
    if yaw_deg is not None:
        rot = trimesh.transformations.rotation_matrix(np.radians(yaw_deg), [0, 1, 0])
        mesh.apply_transform(rot)
        front_rgb, span, _ = render_ortho_front(mesh)
        lms = detect_landmarks_on_rgb(front_rgb, landmarker_path)
        cv2.imwrite(str(out_dir / "glb_front_preview.png"), cv2.cvtColor(front_rgb, cv2.COLOR_RGB2BGR))
    else:
        for test_yaw in (0.0, 180.0, 90.0, -90.0):
            cand = mesh.copy()
            if abs(test_yaw) > 1e-3:
                cand.apply_transform(trimesh.transformations.rotation_matrix(np.radians(test_yaw), [0, 1, 0]))
            front_rgb, span, _ = render_ortho_front(cand)
            found = detect_landmarks_on_rgb(front_rgb, landmarker_path)
            if found is not None:
                print(f"[import_glb] Detected face facing +Z at yaw={test_yaw} deg!", file=sys.stderr)
                mesh = cand
                lms = found
                cv2.imwrite(str(out_dir / "glb_front_preview.png"), cv2.cvtColor(front_rgb, cv2.COLOR_RGB2BGR))
                break
        if lms is None:
            print("[import_glb] Warning: MediaPipe did not detect face on ortho render; using default proportions.", file=sys.stderr)
            front_rgb, span, _ = render_ortho_front(mesh)
            cv2.imwrite(str(out_dir / "glb_front_preview.png"), cv2.cvtColor(front_rgb, cv2.COLOR_RGB2BGR))

    # Re-center X on nose/eye center if detected
    if lms is not None:
        nose_x_w = (float(lms[6, 0]) - 0.5) * span
        mesh.apply_translation([-nose_x_w, 0.0, 0.0])

    # Build dense color lookup tree from high-res textured surface sampling
    print("[import_glb] Sampling high-res texture colors & building structured contour grid...", file=sys.stderr)
    c_pts, c_fidx, c_rgba = trimesh.sample.sample_surface(mesh, 450000, sample_color=True, seed=1)
    color_tree = cKDTree(c_pts)
    if c_rgba is not None and len(c_rgba) == len(c_pts):
        c_rgb = c_rgba[:, :3].astype(np.float32) / 255.0
        c_lum = 0.299 * c_rgb[:, 0] + 0.587 * c_rgb[:, 1] + 0.114 * c_rgb[:, 2]
        p3, p97 = float(np.percentile(c_lum, 3)), float(np.percentile(c_lum, 97))
        if p97 - p3 > 0.05:
            c_lum = np.clip((c_lum - p3) / (p97 - p3), 0.0, 1.0)
        else:
            c_lum = np.clip(0.45 + 0.45 * np.maximum(0.0, mesh.face_normals[c_fidx, 2]), 0.0, 1.0)
    else:
        c_lum = np.clip(0.45 + 0.45 * np.maximum(0.0, mesh.face_normals[c_fidx, 2]), 0.0, 1.0)

    # Structured cylindrical raycast: for each (row, col), cast inward from R=2.5 toward spine (0, y, 0)
    u_lin = np.linspace(0.0, 1.0, n_y, dtype=np.float32)
    v_grid = 0.5 * (1.0 - np.cos(math.pi * u_lin)) * 0.18 + u_lin * 0.82
    v_grid[0] = 0.008
    v_grid = np.clip(v_grid, 0.008, 0.965)
    y_world = 1.0 - 2.0 * v_grid

    thetas = np.linspace(-math.pi, math.pi, n_theta, endpoint=False, dtype=np.float32)
    sin_t = np.sin(thetas)
    cos_t = np.cos(thetas)

    yy = np.repeat(y_world[:, None], n_theta, axis=1)
    sx = np.repeat(sin_t[None, :], n_y, axis=0)
    cz = np.repeat(cos_t[None, :], n_y, axis=0)

    R_start = 2.6
    origins = np.stack([sx * R_start, yy, cz * R_start], axis=-1).reshape(-1, 3)
    dirs = np.stack([-sx, np.zeros_like(yy), -cz], axis=-1).reshape(-1, 3)

    # multiple_hits=False returns an arbitrary hit per ray (spatial-index order), which can be an
    # inner surface (layered hair, glasses) and varies between runs. Take every hit and keep the
    # one nearest the ray origin: the outermost surface seen from outside the head.
    all_locs, all_ray, all_tri = mesh.ray.intersects_location(origins, dirs, multiple_hits=True)
    if len(all_ray):
        dist = np.einsum('ij,ij->i', all_locs - origins[all_ray], dirs[all_ray])
        order = np.lexsort((dist, all_ray))
        first = np.ones(len(order), dtype=bool)
        first[1:] = all_ray[order][1:] != all_ray[order][:-1]
        keep = order[first]
        locs, idx_ray, idx_tri = all_locs[keep], all_ray[keep], all_tri[keep]
    else:
        locs, idx_ray, idx_tri = all_locs, all_ray, all_tri

    X = np.zeros((n_y, n_theta), dtype=np.float32)
    Y = yy.copy()
    Z = np.zeros((n_y, n_theta), dtype=np.float32)
    hit = np.zeros(n_y * n_theta, dtype=bool)

    flat_X = np.zeros(n_y * n_theta, dtype=np.float32)
    flat_Z = np.zeros(n_y * n_theta, dtype=np.float32)
    flat_X[idx_ray] = locs[:, 0]
    flat_Z[idx_ray] = locs[:, 2]
    hit[idx_ray] = True

    # Fill any grazing misses cleanly via nearest hit on cylindrical grid
    if not np.all(hit):
        hit_mask = hit.reshape(n_y, n_theta)
        valid_coords = np.argwhere(hit_mask)
        miss_coords = np.argwhere(~hit_mask)
        if len(valid_coords) > 0 and len(miss_coords) > 0:
            tree_2d = cKDTree(valid_coords)
            _, nn = tree_2d.query(miss_coords)
            src = valid_coords[nn]
            X_tmp = flat_X.reshape(n_y, n_theta)
            Z_tmp = flat_Z.reshape(n_y, n_theta)
            X_tmp[miss_coords[:, 0], miss_coords[:, 1]] = X_tmp[src[:, 0], src[:, 1]]
            Z_tmp[miss_coords[:, 0], miss_coords[:, 1]] = Z_tmp[src[:, 0], src[:, 1]]
            X, Z = X_tmp, Z_tmp
        else:
            X = flat_X.reshape(n_y, n_theta)
            Z = flat_Z.reshape(n_y, n_theta)
    else:
        X = flat_X.reshape(n_y, n_theta)
        Z = flat_Z.reshape(n_y, n_theta)

    # Remove isolated single-ray grazing jump outliers (e.g. hoodie collar edge) while preserving crisp sculpt detail
    def clean_cyl(arr):
        pad = 6
        p = np.concatenate([arr[:, -pad:], arr, arr[:, :pad]], axis=1)
        p = median_filter(p, size=(3, 3))
        p = gaussian_filter(p, sigma=(0.65, 0.75))
        return p[:, pad:-pad]

    X = clean_cyl(X)
    Z = clean_cyl(Z)

    # Smooth top crown pole
    cx0, cz0 = float(np.mean(X[0])), float(np.mean(Z[0]))
    for r in range(3):
        a = r / 3.0
        X[r] = (1 - a) * cx0 + a * X[r]
        Z[r] = (1 - a) * cz0 + a * Z[r]

    # Compute outward normals
    dX_dt = 0.5 * (np.roll(X, -1, axis=1) - np.roll(X, 1, axis=1))
    dY_dt = 0.5 * (np.roll(Y, -1, axis=1) - np.roll(Y, 1, axis=1))
    dZ_dt = 0.5 * (np.roll(Z, -1, axis=1) - np.roll(Z, 1, axis=1))
    dX_dv = np.gradient(X, axis=0)
    dY_dv = np.gradient(Y, axis=0)
    dZ_dv = np.gradient(Z, axis=0)

    Nx = dY_dv * dZ_dt - dZ_dv * dY_dt
    Ny = dZ_dv * dX_dt - dX_dv * dZ_dt
    Nz = dX_dv * dY_dt - dY_dv * dX_dt
    nlen = np.sqrt(Nx * Nx + Ny * Ny + Nz * Nz) + 1e-8
    Nx /= nlen; Ny /= nlen; Nz /= nlen
    front_col = int(np.argmin(np.abs(thetas)))
    if Nz[n_y // 3, front_col] < 0:
        Nx, Ny, Nz = -Nx, -Ny, -Nz

    # Sample baked texture luminance smoothly + enhance micro-contrast (CLAHE + unsharp) so pores, beard & hoodie pop!
    grid_pts = np.stack([X, Y, Z], axis=-1).reshape(-1, 3)
    _, nn_idx = color_tree.query(grid_pts, k=3)
    lum_raw = np.mean(c_lum[nn_idx], axis=-1).reshape(n_y, n_theta).astype(np.float32)

    lum_u8 = np.clip(lum_raw * 255.0, 0, 255).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=3.2, tileGridSize=(8, 8))
    lum_clahe = clahe.apply(lum_u8).astype(np.float32) / 255.0
    lum_blur = cv2.GaussianBlur(lum_raw, (0, 0), 1.6)
    lum_usm = np.clip(lum_raw + (lum_raw - lum_blur) * 1.8, 0.0, 1.0)
    lum_baked = np.clip(0.35 * lum_raw + 0.40 * lum_clahe + 0.25 * lum_usm, 0.0, 1.0).astype(np.float32)

    # Sobel edge magnitude on enhanced luminance + gentle geometric normal curvature so sculpted details shine cleanly
    gx = cv2.Sobel(lum_baked, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(lum_baked, cv2.CV_32F, 0, 1, ksize=3)
    tex_edge = np.clip(np.sqrt(gx * gx + gy * gy) * 1.85, 0.0, 0.95)
    n_curv = np.sqrt(
        np.gradient(Nx, axis=0) ** 2 + np.gradient(Ny, axis=0) ** 2 + np.gradient(Nz, axis=0) ** 2 +
        np.gradient(Nx, axis=1) ** 2 + np.gradient(Ny, axis=1) ** 2 + np.gradient(Nz, axis=1) ** 2
    ) * 1.55
    n_curv = gaussian_filter(np.clip(n_curv, 0.0, 0.45), sigma=0.7)
    edge_baked = np.clip(tex_edge + n_curv, 0.0, 1.0).astype(np.float32)

    # --- Anatomical Facial Landmarks & Rig Weights ---
    cos_t_grid = cos_t[None, :]
    front_only = cos_t_grid > 0.3

    def sample_front_z(wx, wy):
        d2 = (X - wx) ** 2 + (Y - wy) ** 2 + np.where(front_only, 0.0, 100.0)
        rr, cc = np.unravel_index(int(np.argmin(d2)), X.shape)
        return float(Z[rr, cc])

    if lms is not None:
        # Adjust landmark X for nose recentering we applied above
        nose_x_shift = (float(lms[6, 0]) - 0.5) * span
        def lm_xy(idx):
            return (float(lms[idx, 0]) - 0.5) * span - nose_x_shift, (0.5 - float(lms[idx, 1])) * span

        def lm_poly(indices):
            return np.array([lm_xy(i) for i in indices], dtype=np.float32)

        mx, my_lo = lm_xy(14)
        _, my_up = lm_xy(13)
        mouth_center_x = mx
        mouth_slit_y = 0.5 * (my_up + my_lo)
        ml_x, _ = lm_xy(61)
        mr_x, _ = lm_xy(291)
        mouth_half_w = max(0.06, 0.5 * abs(mr_x - ml_x))
        chin_x, chin_y = lm_xy(152)
        nose_x, nose_y = lm_xy(1)

        left_eye_pts = lm_poly(LEFT_EYE_UPPER)
        left_eye_low = lm_poly(LEFT_EYE_LOWER)
        right_eye_pts = lm_poly(RIGHT_EYE_UPPER)
        right_eye_low = lm_poly(RIGHT_EYE_LOWER)
        left_brow_pts = lm_poly(LEFT_BROW)
        right_brow_pts = lm_poly(RIGHT_BROW)

        # Clean smooth symmetric lip-parting curve (prevents single-row islands/flaps at mouth corners)
        corner_y = 0.5 * (lm_xy(61)[1] + lm_xy(291)[1])
        corner_drop = float(np.clip(mouth_slit_y - corner_y, -0.012, 0.022))
        u_mouth = np.clip((X - mouth_center_x) / max(mouth_half_w, 0.05), -1.15, 1.15)
        y_slit_at_X = mouth_slit_y - corner_drop * (u_mouth ** 2)

        li_x, li_y = lm_xy(468) if len(lms) > 473 else (float(np.mean(left_eye_pts[:, 0])), float(np.mean(left_eye_pts[:, 1])))
        ri_x, ri_y = lm_xy(473) if len(lms) > 473 else (float(np.mean(right_eye_pts[:, 0])), float(np.mean(right_eye_pts[:, 1])))

        def eye_info_and_mask(up, lo):
            ex_c = float(np.mean(up[:, 0]))
            ey_top, ey_bot = float(np.max(up[:, 1])), float(np.min(lo[:, 1]))
            ey_c = 0.5 * (ey_top + ey_bot)
            ew = max(0.04, 0.5 * float(np.max(up[:, 0]) - np.min(up[:, 0])))
            eh = max(0.018, 0.5 * (ey_top - ey_bot))
            dx = (X - ex_c) / (ew * 1.25)
            dy = (Y - (ey_c + eh * 0.45)) / (eh * 1.65)
            m = np.exp(-(dx * dx + dy * dy))
            above = 1.0 / (1.0 + np.exp(-np.clip((Y - (ey_bot + eh * 0.15)) * 120.0, -20.0, 20.0)))
            return m * above * np.clip(cos_t_grid, 0.0, 1.0), (ex_c, ey_c, ew, eh)

        l_blink, l_eye_info = eye_info_and_mask(left_eye_pts, left_eye_low)
        r_blink, r_eye_info = eye_info_and_mask(right_eye_pts, right_eye_low)
        blink_w = np.clip(l_blink + r_blink, 0.0, 1.0)

        def b_mask(pts_b):
            bx, by = float(np.mean(pts_b[:, 0])), float(np.mean(pts_b[:, 1]))
            bw = max(0.06, 0.5 * float(np.max(pts_b[:, 0]) - np.min(pts_b[:, 0])))
            return np.exp(-(((X - bx) / (bw * 1.35)) ** 2 + ((Y - (by + 0.015)) / 0.065) ** 2)) * np.clip(cos_t_grid, 0.0, 1.0)

        brow_w = np.clip(b_mask(left_brow_pts) + b_mask(right_brow_pts), 0.0, 1.0)
        eye_y_avg = 0.5 * (l_eye_info[1] + r_eye_info[1])
    else:
        mouth_center_x, mouth_slit_y, mouth_half_w = 0.0, -0.13, 0.15
        chin_x, chin_y, nose_x, nose_y, eye_y_avg = 0.0, -0.45, 0.0, 0.10, 0.34
        y_slit_at_X = np.full_like(X, mouth_slit_y)
        li_x, li_y, ri_x, ri_y = -0.20, 0.34, 0.20, 0.34
        l_eye_info, r_eye_info = (-0.20, 0.34, 0.085, 0.025), (0.20, 0.34, 0.085, 0.025)
        blink_w = np.clip(
            np.exp(-(((X + 0.20) / 0.11) ** 2 + ((Y - 0.35) / 0.045) ** 2)) +
            np.exp(-(((X - 0.20) / 0.11) ** 2 + ((Y - 0.35) / 0.045) ** 2)), 0.0, 1.0
        ) * np.clip(cos_t_grid, 0.0, 1.0)
        brow_w = np.clip(
            np.exp(-(((X + 0.20) / 0.14) ** 2 + ((Y - 0.44) / 0.065) ** 2)) +
            np.exp(-(((X - 0.20) / 0.14) ** 2 + ((Y - 0.44) / 0.065) ** 2)), 0.0, 1.0
        ) * np.clip(cos_t_grid, 0.0, 1.0)

    mouth_center_z = sample_front_z(mouth_center_x, mouth_slit_y)
    li_z = sample_front_z(li_x, li_y)
    ri_z = sample_front_z(ri_x, ri_y)
    jaw_pivot = [0.0, float(0.55 * nose_y + 0.45 * eye_y_avg), -0.04]
    neck_pivot = [0.0, float(chin_y - 0.16), -0.06]

    collar_y = chin_y - 0.14
    head_w = np.clip((Y - (collar_y - 0.18)) / 0.30, 0.0, 1.0)
    head_w = head_w * head_w * (3.0 - 2.0 * head_w)
    beard_p = np.exp(-((X / 0.32) ** 2 + np.maximum(0.0, collar_y - Y) ** 2 / 0.03)) * np.clip(cos_t_grid, 0.0, 1.0)
    head_w = np.clip(np.maximum(head_w, beard_p * 0.95), 0.0, 1.0)

    dist_below = y_slit_at_X - Y
    # Sharp parting strictly inside the actual mouth aperture; smooth non-tearing transition on outer cheeks
    in_mouth_x = np.exp(-((X - mouth_center_x) / (mouth_half_w * 0.96)) ** 6)
    hard_below = (dist_below >= 0.0).astype(np.float32)
    soft_below = 1.0 / (1.0 + np.exp(-np.clip(dist_below * 12.0, -20.0, 20.0)))
    below_mask = in_mouth_x * hard_below + (1.0 - in_mouth_x) * soft_below
    jaw_x_env = np.exp(-((X - mouth_center_x) / (mouth_half_w * 2.9)) ** 2)
    jaw_bot_env = np.clip((Y - (chin_y - 0.28)) / 0.18, 0.0, 1.0)
    jaw_bot_env = jaw_bot_env * jaw_bot_env * (3.0 - 2.0 * jaw_bot_env)
    jaw_front_env = np.clip((cos_t_grid + 0.15) / 0.65, 0.0, 1.0)
    jaw_w = np.clip(below_mask * jaw_x_env * jaw_bot_env * jaw_front_env, 0.0, 1.0)

    # Lip deformation weights strictly focused on lips & corners (taper cleanly to 0 on outer cheeks)
    dx_m = (X - mouth_center_x) / (mouth_half_w * 1.12)
    dy_m = (Y - y_slit_at_X) / (mouth_half_w * 0.68)
    lip_x_taper = np.exp(-((X - mouth_center_x) / (mouth_half_w * 1.15)) ** 4)
    lip_env = np.exp(-(dx_m * dx_m + dy_m * dy_m)) * lip_x_taper * np.clip(cos_t_grid, 0.0, 1.0)
    lip_sign = np.where(Y >= y_slit_at_X, 1.0, -1.0).astype(np.float32)
    lip_signed_w = np.clip(lip_env * lip_sign, -1.0, 1.0)

    edge_baked = np.clip(edge_baked + brow_w * 0.20 + np.abs(lip_signed_w) * 0.18, 0.0, 1.0)

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
        "sourceGlb": str(glb_path.resolve()),
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
            "noseTip": [round(nose_x, 5), round(nose_y, 5), round(sample_front_z(nose_x, nose_y), 5)],
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

    print(f"[import_glb] Exported {num_verts:,} structured vertices -> {bin_path}", file=sys.stderr)
    print(json.dumps({"status": "ok", "numVertices": num_verts, "landmarksDetected": lms is not None, "bin": str(bin_path), "meta": str(meta_path)}))


def main():
    parser = argparse.ArgumentParser(description="Import external .glb/.gltf/.obj bust into HoloMe format")
    parser.add_argument("--glb", required=True, help="Path to 3D bust file (.glb, .gltf, .obj)")
    parser.add_argument("--out", default="assets/bust_scenario", help="Output bust directory")
    parser.add_argument("--landmarker", default="assets/models/face_landmarker.task", help="MediaPipe face_landmarker.task path")
    parser.add_argument("--rows", type=int, default=260, help="Vertical contour rings")
    parser.add_argument("--cols", type=int, default=280, help="Circumferential columns")
    parser.add_argument("--yaw", type=float, default=None, help="Manual yaw rotation in degrees (auto-detected if omitted)")
    args = parser.parse_args()

    import_glb(
        Path(args.glb),
        Path(args.out),
        Path(args.landmarker),
        n_y=args.rows,
        n_theta=args.cols,
        yaw_deg=args.yaw,
    )


if __name__ == "__main__":
    main()
