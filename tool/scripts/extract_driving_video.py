#!/usr/bin/env python3
"""
HoloMe Live-Action Driving Video -> 3D Hologram Projection Extractor.

Takes any talking-head driving video (e.g. from Scenario LTX-2.5 Audio-to-Video or a real
webcam/iPhone clip of the speaker) and:
1. Extracts frames at exact target FPS aligned to the audio track.
2. Runs MediaPipe FaceLandmarker (478 landmarks + 52 ARKit blendshapes + 3D head pose) per frame.
3. Registers every frame's face via rigid upper-face landmarks (eyes + nose bridge) into the
   3D bust's canonical front-face UV projection space (384x384), preserving real rolling lips,
   teeth, tongue, mustache flex, cheeks, and natural blinks with zero drift!
4. Packs CLAHE + unsharp enhanced luminance (R) and multi-scale Sobel edge contrast (G)
   into compact per-frame JPEGs for GPU texture projection onto the 108,800-point 3D bust.
5. Combines real tracked 3D jaw/head motion with acoustic HUD spectrum/glitch into timeline.json.
"""

import argparse
import hashlib
import json
import math
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from scipy.io import wavfile
from scipy.ndimage import gaussian_filter1d


# Rigid upper-face landmarks for drift-free similarity registration (do not deform during speech!)
RIGID_LMS = [33, 133, 362, 263, 6, 1]

# Canonical front-face UV projection bounds in HoloMe bust world units
# Covers full forehead, brows, eyes, nose, cheeks, mustache, mouth, goatee, and chin!
UV_X_MIN = -0.58
UV_X_MAX = 0.58
UV_Y_MIN = -0.62
UV_Y_MAX = 0.78
TEX_SIZE = 384


def world_to_uv_px(wx: float, wy: float, size: int = TEX_SIZE) -> tuple[float, float]:
    u = (wx - UV_X_MIN) / (UV_X_MAX - UV_X_MIN)
    v = (UV_Y_MAX - wy) / (UV_Y_MAX - UV_Y_MIN)  # top (+Y) -> row 0
    return u * (size - 1), v * (size - 1)


def get_canonical_dst_points(bust_meta: dict, size: int = TEX_SIZE) -> np.ndarray:
    """Compute exact destination (u, v) pixel coordinates of the 6 rigid landmarks on our 3D bust."""
    piv = bust_meta["pivots"]
    le = piv["leftEye"]   # [cx, cy, halfW, halfH]
    re = piv["rightEye"]  # [cx, cy, halfW, halfH]
    nose = piv["noseTip"] # [x, y, z]

    # Approximate exact MediaPipe 6 rigid anchor positions in bust world coordinates:
    # 33: viewer-left outer eye corner, 133: viewer-left inner eye corner
    # 362: viewer-right inner eye corner, 263: viewer-right outer eye corner
    # 6: nose bridge (between eyes), 1: nose tip
    pts_w = [
        (le[0] - le[2], le[1]),
        (le[0] + le[2], le[1]),
        (re[0] - re[2], re[1]),
        (re[0] + re[2], re[1]),
        (0.5 * (le[0] + re[0]), 0.5 * (le[1] + re[1]) + 0.015),
        (nose[0], nose[1]),
    ]
    return np.array([world_to_uv_px(x, y, size) for x, y in pts_w], dtype=np.float32)


def decode_audio_features(audio_path: str, num_frames: int, fps: int, glitch_amount: float):
    """Extract HUD 8-band spectrum, loudness energy, and glitch impulses from audio."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
        subprocess.run([
            "ffmpeg", "-y", "-v", "error", "-i", audio_path,
            "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", tmp.name,
        ], check=True)
        sr, data = wavfile.read(tmp.name)
        samples = data.astype(np.float32) / 32768.0

    duration = len(samples) / float(sr)
    h = hashlib.sha256(samples[:min(len(samples), 32000)].tobytes()).digest()
    seed = int.from_bytes(h[:4], "little")
    rng = np.random.default_rng(seed)

    win_len = int(sr * 0.036)
    hop = sr / float(fps)
    freqs = np.fft.rfftfreq(win_len, d=1.0 / sr)
    hann = np.hanning(win_len).astype(np.float32)
    band_edges = [80, 220, 450, 800, 1400, 2400, 4000, 6000, 7900]
    band_masks = [(freqs >= band_edges[b]) & (freqs < band_edges[b + 1]) for b in range(8)]

    rms = np.zeros(num_frames, dtype=np.float32)
    spec_bands = np.zeros((num_frames, 8), dtype=np.float32)

    for i in range(num_frames):
        center = int(i * hop)
        s0 = center - win_len // 2
        buf = np.zeros(win_len, dtype=np.float32)
        src0, src1 = max(0, s0), min(len(samples), s0 + win_len)
        if src1 > src0:
            buf[src0 - s0 : src0 - s0 + (src1 - src0)] = samples[src0:src1]
        rms[i] = math.sqrt(float(np.mean(buf * buf)) + 1e-12)
        mag = np.abs(np.fft.rfft(buf * hann))
        for b in range(8):
            spec_bands[i, b] = float(np.mean(mag[band_masks[b]]))

    noise_floor = float(np.percentile(rms, 10))
    peak = max(float(np.percentile(rms, 96)), noise_floor + 1e-4)
    speech_env = np.tanh(np.clip((rms - noise_floor * 1.15) / (peak - noise_floor * 1.15 + 1e-6), 0.0, 1.5) * 1.3)

    for b in range(8):
        p95 = float(np.percentile(spec_bands[:, b], 95)) + 1e-6
        spec_bands[:, b] = gaussian_filter1d(np.clip(spec_bands[:, b] / p95, 0.0, 1.0), sigma=1.0)

    t = np.arange(num_frames, dtype=np.float32) / float(fps)
    onset = np.clip(np.maximum(0.0, np.diff(speech_env, prepend=speech_env[0])) * 4.5, 0.0, 1.0)
    glitch = np.zeros(num_frames, dtype=np.float32)
    if glitch_amount > 0.001:
        g_t = float(rng.uniform(0.7, 2.2))
        while g_t < duration:
            dur_g = float(rng.uniform(0.06, 0.16))
            amp_g = float(rng.uniform(0.35, 0.95)) * glitch_amount
            glitch[(t >= g_t) & (t <= g_t + dur_g)] = amp_g
            g_t += float(rng.uniform(2.0, 4.6))
        glitch = np.clip(glitch + onset * 0.20 * glitch_amount, 0.0, 1.0)

    return speech_env, spec_bands, glitch, duration, seed


def rotation_matrix_to_euler_yxz(R: np.ndarray) -> tuple[float, float, float]:
    """Extract (pitch, yaw, roll) in radians from 3x3 facial transform rotation matrix."""
    # MediaPipe face transform: +X right, +Y up, +Z out of face toward camera
    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    if sy > 1e-6:
        pitch = math.atan2(-R[2, 1], R[2, 2])
        yaw = math.atan2(R[2, 0], sy)
        roll = math.atan2(-R[1, 0], R[0, 0])
    else:
        pitch = math.atan2(R[1, 2], R[1, 1])
        yaw = math.atan2(R[2, 0], sy)
        roll = 0.0
    return pitch, yaw, roll


def encode_hologram_face_map(warped_bgr: np.ndarray, clahe, edge_gain: float = 1.0) -> np.ndarray:
    """
    Pack crisp holographic face detail into RGB JPEG (note OpenCV BGR order):
    - R channel (idx 2): High-contrast perceptual luminance (real dark mouth cavity, bright lips/teeth/eyes/skin!)
    - G channel (idx 1): Multi-scale Sobel edge contrast (eyelashes, brows, lip contours, beard hairs!)
    - B channel (idx 0): Raw normalized luminance for smooth midtone preservation
    """
    gray_u8 = cv2.cvtColor(warped_bgr, cv2.COLOR_BGR2GRAY)
    gray = gray_u8.astype(np.float32) / 255.0

    # Local CLAHE contrast + unsharp fine detail (beard stubble, teeth, lip texture, eyelashes!)
    enh = clahe.apply(gray_u8).astype(np.float32) / 255.0
    blur_usm = cv2.GaussianBlur(gray, (0, 0), 1.8)
    usm = np.clip(gray + (gray - blur_usm) * 1.65, 0.0, 1.0)

    # Preserve deep dark open mouth & pupils cleanly (don't let CLAHE wash out pitch-black open mouth!)
    dark_protect = np.clip(gray / 0.22, 0.0, 1.0)
    lum = np.clip((0.40 * gray + 0.34 * enh + 0.26 * usm) * (0.28 + 0.72 * dark_protect), 0.0, 1.0)

    # Multi-scale Sobel edges
    gx1 = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy1 = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    e1 = np.sqrt(gx1 * gx1 + gy1 * gy1)
    blur = cv2.GaussianBlur(gray, (5, 5), 1.2)
    gx2 = cv2.Sobel(blur, cv2.CV_32F, 1, 0, ksize=3)
    gy2 = cv2.Sobel(blur, cv2.CV_32F, 0, 1, ksize=3)
    e2 = np.sqrt(gx2 * gx2 + gy2 * gy2)
    # edge_gain comes from the bust (faceDetailGain): smooth-skinned faces get the same lift
    # here as in the baked edge channel, so driving mode is not dimmer than procedural mode.
    edge = np.clip((e1 * 2.1 + e2 * 1.2) * edge_gain, 0.0, 1.0)

    # Pack into BGR image -> R=lum, G=edge, B=raw gray
    out_bgr = np.zeros((warped_bgr.shape[0], warped_bgr.shape[1], 3), dtype=np.uint8)
    out_bgr[:, :, 2] = np.clip(lum * 255.0, 0, 255).astype(np.uint8)   # R -> texture.r
    out_bgr[:, :, 1] = np.clip(edge * 255.0, 0, 255).astype(np.uint8)  # G -> texture.g
    out_bgr[:, :, 0] = np.clip(gray * 255.0, 0, 255).astype(np.uint8)  # B -> texture.b
    return out_bgr


def extract_driving_video(
    video_path: Path,
    audio_path: Path,
    bust_meta_path: Path,
    landmarker_path: Path,
    out_dir: Path,
    fps: int = 30,
    glitch_amount: float = 0.35,
):
    out_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = out_dir / "driving_frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    with open(bust_meta_path) as f:
        bust_meta = json.load(f)
    dst_pts = get_canonical_dst_points(bust_meta, TEX_SIZE)

    # Resample driving video cleanly to exact target fps into a temp mp4 for deterministic frame reads
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp_vid:
        tmp_vid_path = Path(tmp_vid.name)

    try:
        subprocess.run([
            "ffmpeg", "-y", "-v", "error",
            "-i", str(video_path),
            "-vf", f"fps={fps}",
            "-c:v", "libx264", "-preset", "ultrafast", "-crf", "14",
            "-an", str(tmp_vid_path),
        ], check=True)

        cap = cv2.VideoCapture(str(tmp_vid_path))
        raw_frames = []
        while True:
            ok, frm = cap.read()
            if not ok:
                break
            raw_frames.append(frm)
        cap.release()
    finally:
        tmp_vid_path.unlink(missing_ok=True)

    if not raw_frames:
        raise RuntimeError(f"Could not decode any frames from driving video: {video_path}")

    num_frames = len(raw_frames)
    print(f"[extract_driving] Tracking & registering {num_frames} frames @ {fps}fps via MediaPipe...", file=sys.stderr)

    # Audio features for HUD + glitch
    speech_env, spec_bands, glitch, audio_dur, seed = decode_audio_features(
        str(audio_path), num_frames, fps, glitch_amount
    )

    options = mp.tasks.vision.FaceLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(landmarker_path)),
        running_mode=mp.tasks.vision.RunningMode.IMAGE,
        num_faces=1,
        output_face_blendshapes=True,
        output_facial_transformation_matrixes=True,
    )

    clahe = cv2.createCLAHE(clipLimit=2.6, tileGridSize=(8, 8))
    edge_gain = float(bust_meta.get("faceDetailGain", 1.0))
    prev_M = None
    smooth_M = None

    jaw_arr = np.zeros(num_frames, dtype=np.float32)
    funnel_arr = np.zeros(num_frames, dtype=np.float32)
    wide_arr = np.zeros(num_frames, dtype=np.float32)
    blink_arr = np.zeros(num_frames, dtype=np.float32)
    brow_arr = np.zeros(num_frames, dtype=np.float32)
    yaw_arr = np.zeros(num_frames, dtype=np.float32)
    pitch_arr = np.zeros(num_frames, dtype=np.float32)
    roll_arr = np.zeros(num_frames, dtype=np.float32)

    base_yaw = None
    base_pitch = None
    base_roll = None

    with mp.tasks.vision.FaceLandmarker.create_from_options(options) as landmarker:
        for i, bgr in enumerate(raw_frames):
            H, W = bgr.shape[:2]
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            res = landmarker.detect(mp_img)

            if res.face_landmarks:
                lms = res.face_landmarks[0]
                src_pts = np.array([[lms[idx].x * W, lms[idx].y * H] for idx in RIGID_LMS], dtype=np.float32)
                M, _ = cv2.estimateAffinePartial2D(src_pts, dst_pts, method=cv2.LMEDS)
                if M is not None:
                    if smooth_M is None:
                        smooth_M = M.copy()
                    else:
                        # Light temporal smoothing on registration matrix removes 1px landmark jitter!
                        smooth_M = 0.65 * M + 0.35 * smooth_M
                    prev_M = smooth_M.copy()

                # Extract real ARKit blendshapes from the live talking video!
                if res.face_blendshapes:
                    bs = {b.category_name: float(b.score) for b in res.face_blendshapes[0]}
                    jaw_arr[i] = bs.get("jawOpen", 0.0)
                    funnel_arr[i] = max(bs.get("mouthFunnel", 0.0), bs.get("mouthPucker", 0.0))
                    wide_arr[i] = 0.5 * (bs.get("mouthSmileLeft", 0.0) + bs.get("mouthSmileRight", 0.0))
                    blink_arr[i] = 0.5 * (bs.get("eyeBlinkLeft", 0.0) + bs.get("eyeBlinkRight", 0.0))
                    brow_arr[i] = bs.get("browInnerUp", 0.0)

                # Extract real 3D head orientation from MediaPipe transform matrix!
                if res.facial_transformation_matrixes:
                    mat = np.array(res.facial_transformation_matrixes[0], dtype=np.float32)
                    p_r, y_r, r_r = rotation_matrix_to_euler_yxz(mat[:3, :3])
                    if base_yaw is None:
                        base_yaw, base_pitch, base_roll = y_r, p_r, r_r
                    yaw_arr[i] = np.clip((y_r - base_yaw) * 0.72, -0.26, 0.26)
                    pitch_arr[i] = np.clip(-(p_r - base_pitch) * 0.72, -0.18, 0.18)
                    roll_arr[i] = np.clip((r_r - base_roll) * 0.65, -0.14, 0.14)

            if prev_M is not None:
                warped = cv2.warpAffine(
                    bgr, prev_M, (TEX_SIZE, TEX_SIZE),
                    flags=cv2.INTER_CUBIC,
                    borderMode=cv2.BORDER_REPLICATE,
                )
            else:
                warped = cv2.resize(bgr, (TEX_SIZE, TEX_SIZE), interpolation=cv2.INTER_AREA)

            packed = encode_hologram_face_map(warped, clahe, edge_gain)
            cv2.imwrite(str(frames_dir / f"f_{i:05d}.jpg"), packed, [cv2.IMWRITE_JPEG_QUALITY, 92])

    # Gentle temporal smoothing on tracked head pose & physical 3D jaw follow-through
    yaw_arr = gaussian_filter1d(yaw_arr, sigma=1.4)
    pitch_arr = gaussian_filter1d(pitch_arr, sigma=1.4)
    roll_arr = gaussian_filter1d(roll_arr, sigma=1.4)
    jaw_arr = gaussian_filter1d(jaw_arr, sigma=0.8)

    frames = []
    for i in range(num_frames):
        frames.append({
            "t": round(i / float(fps), 4),
            "energy": round(float(speech_env[i]), 4),
            "jawOpen": round(float(jaw_arr[i]), 4),
            "mouthFunnel": round(float(funnel_arr[i]), 4),
            "mouthWide": round(float(wide_arr[i]), 4),
            "mouthClose": 0.0,
            "mouthF": 0.0,
            "upperLipRaise": round(float(jaw_arr[i] * 0.3), 4),
            "blink": round(float(blink_arr[i]), 4),
            "browRaise": round(float(brow_arr[i]), 4),
            "headYaw": round(float(yaw_arr[i]), 5),
            "headPitch": round(float(pitch_arr[i]), 5),
            "headRoll": round(float(roll_arr[i]), 5),
            "glitch": round(float(glitch[i]), 4),
            "spectrum": [round(float(v), 3) for v in spec_bands[i]],
            "drivingFrame": f"f_{i:05d}.jpg",
        })

    timeline = {
        "version": 2,
        "mode": "driving_video",
        "drivingBounds": [UV_X_MIN, UV_X_MAX, UV_Y_MIN, UV_Y_MAX],
        "fps": fps,
        "duration": round(num_frames / float(fps), 4),
        "numFrames": num_frames,
        "seed": seed,
        "frames": frames,
    }
    tl_path = out_dir / "timeline.json"
    with open(tl_path, "w") as f:
        json.dump(timeline, f)

    print(json.dumps({
        "status": "ok",
        "mode": "driving_video",
        "numFrames": num_frames,
        "duration": timeline["duration"],
        "fps": fps,
        "framesDir": str(frames_dir),
        "out": str(tl_path),
    }))


def main():
    parser = argparse.ArgumentParser(description="Extract registered holographic driving video frames & 3D pose timeline")
    parser.add_argument("--video", required=True, help="Driving talking-head video (.mp4/.mov)")
    parser.add_argument("--audio", required=True, help="Audio track for HUD spectrum & sync")
    parser.add_argument("--bust-meta", required=True, help="Path to bust_meta.json")
    parser.add_argument("--landmarker", default="assets/models/face_landmarker.task", help="MediaPipe landmarker model")
    parser.add_argument("--out-dir", required=True, help="Output directory for timeline.json & driving_frames/")
    parser.add_argument("--fps", type=int, default=30, help="Target render FPS")
    parser.add_argument("--glitch", type=float, default=0.35, help="Glitch intensity")
    args = parser.parse_args()

    extract_driving_video(
        Path(args.video),
        Path(args.audio),
        Path(args.bust_meta),
        Path(args.landmarker),
        Path(args.out_dir),
        fps=args.fps,
        glitch_amount=args.glitch,
    )


if __name__ == "__main__":
    main()
