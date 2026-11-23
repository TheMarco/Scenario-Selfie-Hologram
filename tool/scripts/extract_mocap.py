#!/usr/bin/env python3
"""
HoloMe 3D Facial Motion Capture Extractor (--mocap-video).

Tracks a real talking-head video (e.g. iPhone .MOV or webcam .mp4) locally using
MediaPipe FaceLandmarker (52 ARKit blendshapes + 3D head transformation matrix)
and maps the speaker's real mouth articulation, blinks, brow raises, and 3D head
cadence directly onto HoloMe's crisp 108,800-point 3D hologram rig.

100% local, zero cloud APIs, zero video-pixel smearing.
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


def decode_audio_features(audio_path: str, fps: int, glitch_amount: float):
    """Extract duration, HUD 8-band spectrum, loudness energy, and glitch impulses from audio."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
        subprocess.run([
            "ffmpeg", "-y", "-v", "error", "-i", audio_path,
            "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", tmp.name,
        ], check=True)
        sr, data = wavfile.read(tmp.name)
        samples = data.astype(np.float32) / 32768.0

    duration = len(samples) / float(sr)
    num_frames = max(1, int(math.ceil(duration * fps)))

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
    speech_env = np.tanh(
        np.clip((rms - noise_floor * 1.15) / (peak - noise_floor * 1.15 + 1e-6), 0.0, 1.5) * 1.3
    )

    for b in range(8):
        p95 = float(np.percentile(spec_bands[:, b], 95)) + 1e-6
        spec_bands[:, b] = gaussian_filter1d(np.clip(spec_bands[:, b] / p95, 0.0, 1.0), sigma=1.0)

    t = np.arange(num_frames, dtype=np.float32) / float(fps)
    onset = np.clip(np.maximum(0.0, np.diff(speech_env, prepend=speech_env[0])) * 4.5, 0.0, 1.0)
    glitch = np.zeros(num_frames, dtype=np.float32)
    if glitch_amount > 0.01:
        glitch = np.clip(onset * 0.25 * glitch_amount, 0.0, 0.35)
        n_bursts = max(1, int(duration * 0.22 * glitch_amount))
        for _ in range(n_bursts):
            c_idx = int(rng.integers(0, max(1, num_frames - 4)))
            dur_f = int(rng.integers(2, 5))
            amp = float(rng.uniform(0.28, 0.68)) * glitch_amount
            glitch[c_idx : min(num_frames, c_idx + dur_f)] = np.maximum(
                glitch[c_idx : min(num_frames, c_idx + dur_f)], amp
            )
        glitch = gaussian_filter1d(glitch, sigma=0.7)

    return duration, num_frames, t, speech_env, spec_bands, glitch


def iter_video_frames_rgb(video_path: str, fps: int, target_width: int = 540):
    """Stream auto-rotated CFR frames from any video (incl. iPhone HEVC .MOV) via ffmpeg MJPEG pipe."""
    cmd = [
        "ffmpeg", "-v", "error",
        "-i", video_path,
        "-vf", f"fps={fps},scale={target_width}:-2",
        "-f", "image2pipe",
        "-vcodec", "mjpeg",
        "-q:v", "4",
        "-",
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    buf = bytearray()
    try:
        while True:
            chunk = proc.stdout.read(65536)
            if not chunk:
                break
            buf.extend(chunk)
            while True:
                soi = buf.find(b"\xff\xd8")
                if soi == -1:
                    buf.clear()
                    break
                if soi > 0:
                    del buf[:soi]
                eoi = buf.find(b"\xff\xd9", 2)
                if eoi == -1:
                    break
                jpg = bytes(buf[: eoi + 2])
                del buf[: eoi + 2]
                arr = np.frombuffer(jpg, dtype=np.uint8)
                bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if bgr is not None:
                    yield cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    finally:
        proc.stdout.close()
        proc.wait()


def rot_matrix_to_euler_yxz(R: np.ndarray) -> tuple[float, float, float]:
    """Extract (pitch, yaw, roll) in radians from MediaPipe 3x3 facial rotation matrix."""
    # MediaPipe face matrix: +X right, +Y up, +Z toward camera
    sy = math.sqrt(float(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0]))
    if sy > 1e-6:
        pitch = math.atan2(float(R[2, 1]), float(R[2, 2]))
        yaw = math.atan2(float(-R[2, 0]), sy)
        roll = math.atan2(float(R[1, 0]), float(R[0, 0]))
    else:
        pitch = math.atan2(float(-R[1, 2]), float(R[1, 1]))
        yaw = math.atan2(float(-R[2, 0]), sy)
        roll = 0.0
    return pitch, yaw, roll


def interpolate_missing(arr: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Linearly interpolate any frames where face tracking temporarily dropped."""
    if valid.all():
        return arr
    if not valid.any():
        return np.zeros_like(arr)
    idx = np.arange(len(arr))
    return np.interp(idx, idx[valid], arr[valid]).astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description="Extract 3D facial mocap timeline from a talking-head video")
    parser.add_argument("--video", required=True, help="Path to talking-head video (.MOV, .mp4, etc.)")
    parser.add_argument("--audio", default=None, help="Optional separate audio track (defaults to --video)")
    parser.add_argument("--landmarker", required=True, help="Path to MediaPipe face_landmarker.task")
    parser.add_argument("--out", required=True, help="Output timeline JSON path")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--glitch", type=float, default=0.25)
    parser.add_argument("--head-scale", type=float, default=0.85, help="Scale multiplier for tracked 3D head motion")
    args = parser.parse_args()

    audio_src = args.audio or args.video

    # 1. Run local Whisper + CMUdict ARPABET phoneme lip-sync on the exact audio track
    #    for rock-solid, syllable-locked lip articulation (eliminates beard/mustache tracker dropouts
    #    and false jaw opens during silent pauses!)
    print(f"[extract_mocap] Running local Whisper phoneme lip-sync alignment...", file=sys.stderr)
    analyze_script = str(Path(__file__).resolve().parent / "analyze_audio.py")
    with tempfile.NamedTemporaryFile(suffix=".json", delete=True) as tmp_au:
        subprocess.run([
            sys.executable, analyze_script,
            "--audio", audio_src,
            "--out", tmp_au.name,
            "--fps", str(args.fps),
            "--glitch", str(args.glitch),
        ], check=True, stdout=subprocess.DEVNULL)
        with open(tmp_au.name, "r") as f:
            au_tl = json.load(f)

    au_frames = au_tl["frames"]
    duration = float(au_tl["duration"])
    num_frames = int(au_tl["numFrames"])
    t_arr = np.array([f["t"] for f in au_frames], dtype=np.float32)
    speech_env = np.array([f["energy"] for f in au_frames], dtype=np.float32)
    w_jaw = np.array([f["jawOpen"] for f in au_frames], dtype=np.float32)
    w_funnel = np.array([f["mouthFunnel"] for f in au_frames], dtype=np.float32)
    w_wide = np.array([f["mouthWide"] for f in au_frames], dtype=np.float32)
    w_close = np.array([f["mouthClose"] for f in au_frames], dtype=np.float32)
    w_f = np.array([f["mouthF"] for f in au_frames], dtype=np.float32)
    w_upper = np.array([f["upperLipRaise"] for f in au_frames], dtype=np.float32)
    glitch = np.array([f["glitch"] for f in au_frames], dtype=np.float32)
    spec_bands = np.array([f["spectrum"] for f in au_frames], dtype=np.float32)

    # Configure MediaPipe FaceLandmarker in VIDEO mode for smooth temporal tracking
    BaseOptions = mp.tasks.BaseOptions
    FaceLandmarker = mp.tasks.vision.FaceLandmarker
    FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=args.landmarker),
        running_mode=VisionRunningMode.VIDEO,
        output_face_blendshapes=True,
        output_facial_transformation_matrixes=True,
        num_faces=1,
    )

    jaw_open = np.zeros(num_frames, dtype=np.float32)
    mouth_funnel = np.zeros(num_frames, dtype=np.float32)
    mouth_wide = np.zeros(num_frames, dtype=np.float32)
    mouth_close = np.zeros(num_frames, dtype=np.float32)
    mouth_f = np.zeros(num_frames, dtype=np.float32)
    upper_lip = np.zeros(num_frames, dtype=np.float32)
    blink = np.zeros(num_frames, dtype=np.float32)
    brow_raise = np.zeros(num_frames, dtype=np.float32)
    head_yaw = np.zeros(num_frames, dtype=np.float32)
    head_pitch = np.zeros(num_frames, dtype=np.float32)
    head_roll = np.zeros(num_frames, dtype=np.float32)
    valid = np.zeros(num_frames, dtype=bool)

    print(f"[extract_mocap] Tracking {num_frames} frames ({duration:.2f}s @ {args.fps}fps)...", file=sys.stderr)
    tracked_count = 0
    frame_idx = 0

    with FaceLandmarker.create_from_options(options) as landmarker:
        for rgb in iter_video_frames_rgb(args.video, args.fps, target_width=540):
            if frame_idx >= num_frames:
                break
            ts_ms = int(round((frame_idx / float(args.fps)) * 1000.0))
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            res = landmarker.detect_for_video(mp_img, ts_ms)

            if res.face_blendshapes and len(res.face_blendshapes) > 0:
                bs = {b.category_name: float(b.score) for b in res.face_blendshapes[0]}
                jaw_open[frame_idx] = bs.get("jawOpen", 0.0)
                mouth_funnel[frame_idx] = max(
                    bs.get("mouthFunnel", 0.0),
                    bs.get("mouthPucker", 0.0) * 0.92,
                )
                smile = 0.5 * (bs.get("mouthSmileLeft", 0.0) + bs.get("mouthSmileRight", 0.0))
                stretch = 0.5 * (bs.get("mouthStretchLeft", 0.0) + bs.get("mouthStretchRight", 0.0))
                mouth_wide[frame_idx] = max(smile, stretch * 1.15)
                press = 0.5 * (bs.get("mouthPressLeft", 0.0) + bs.get("mouthPressRight", 0.0))
                mouth_close[frame_idx] = max(bs.get("mouthClose", 0.0), press)
                mouth_f[frame_idx] = bs.get("mouthRollLower", 0.0)
                upper_lip[frame_idx] = 0.5 * (
                    bs.get("mouthUpperUpLeft", 0.0) + bs.get("mouthUpperUpRight", 0.0)
                )
                blink[frame_idx] = 0.5 * (
                    bs.get("eyeBlinkLeft", 0.0) + bs.get("eyeBlinkRight", 0.0)
                )
                brow_outer = 0.5 * (
                    bs.get("browOuterUpLeft", 0.0) + bs.get("browOuterUpRight", 0.0)
                )
                brow_raise[frame_idx] = max(bs.get("browInnerUp", 0.0), brow_outer)

                if res.facial_transformation_matrixes and len(res.facial_transformation_matrixes) > 0:
                    mat = np.array(res.facial_transformation_matrixes[0], dtype=np.float32).reshape(4, 4)
                    pitch, yaw, roll = rot_matrix_to_euler_yxz(mat[:3, :3])
                    head_pitch[frame_idx] = pitch
                    head_yaw[frame_idx] = yaw
                    head_roll[frame_idx] = roll

                valid[frame_idx] = True
                tracked_count += 1

            frame_idx += 1
            if frame_idx % 300 == 0:
                print(f"[extract_mocap] Tracked {frame_idx}/{num_frames} frames...", file=sys.stderr)

    # Interpolate any dropped frames
    jaw_open = interpolate_missing(jaw_open, valid)
    mouth_funnel = interpolate_missing(mouth_funnel, valid)
    mouth_wide = interpolate_missing(mouth_wide, valid)
    mouth_close = interpolate_missing(mouth_close, valid)
    mouth_f = interpolate_missing(mouth_f, valid)
    upper_lip = interpolate_missing(upper_lip, valid)
    blink = interpolate_missing(blink, valid)
    brow_raise = interpolate_missing(brow_raise, valid)
    head_yaw = interpolate_missing(head_yaw, valid)
    head_pitch = interpolate_missing(head_pitch, valid)
    head_roll = interpolate_missing(head_roll, valid)

    # Baseline-calibrate & map ARKit scores into HoloMe rig ranges
    jaw_base = float(np.percentile(jaw_open, 6))
    jaw_cal = np.clip((jaw_open - jaw_base) * 1.80, 0.0, 1.0)
    # Bilabial closure cleanly suppresses tiny residual jawOpen during M/B/P presses
    close_cal = np.clip(mouth_close * 1.45, 0.0, 1.0)
    jaw_cal = np.clip(jaw_cal * (1.0 - 0.55 * close_cal), 0.0, 1.0)

    funnel_cal = np.clip(mouth_funnel * 1.45, 0.0, 1.0)
    wide_base = float(np.percentile(mouth_wide, 8))
    wide_cal = np.clip((mouth_wide - wide_base) * 1.45, 0.0, 1.0)
    f_cal = np.clip(mouth_f * 1.65, 0.0, 1.0)
    upper_cal = np.clip(upper_lip * 1.50 + jaw_cal * 0.32, 0.0, 1.0)

    # Blinks: subtract resting eyelid droop so normal eyes stay crisp & open, snap true blinks to 1.0
    blink_base = float(np.percentile(blink, 15))
    blink_cal = np.clip((blink - blink_base - 0.07) * 2.15, 0.0, 1.0)
    blink_cal = np.where(blink_cal < 0.08, 0.0, blink_cal)

    brow_base = float(np.percentile(brow_raise, 12))
    brow_cal = np.clip((brow_raise - brow_base) * 1.45, 0.0, 1.0)

    # Center 3D head pose around speaker's median resting gaze & apply smooth anatomical clamps
    # Positive pitch rotates chin DOWN in shaders.js rotX, matching MediaPipe's +pitch sign directly!
    hs = float(args.head_scale)
    yaw_cal = np.clip((head_yaw - float(np.median(head_yaw))) * hs, -0.36, 0.36)
    pitch_cal = np.clip((head_pitch - float(np.median(head_pitch))) * hs, -0.22, 0.22)
    roll_cal = np.clip((head_roll - float(np.median(head_roll))) * hs, -0.22, 0.22)

    # Light temporal smoothing on tracked mocap channels
    jaw_cal = gaussian_filter1d(jaw_cal, sigma=0.75)
    funnel_cal = gaussian_filter1d(funnel_cal, sigma=0.85)
    wide_cal = gaussian_filter1d(wide_cal, sigma=0.95)
    f_cal = gaussian_filter1d(f_cal, sigma=0.75)
    upper_cal = gaussian_filter1d(upper_cal, sigma=0.85)
    blink_cal = gaussian_filter1d(blink_cal, sigma=0.55)
    brow_cal = gaussian_filter1d(brow_cal, sigma=1.15)
    yaw_cal = gaussian_filter1d(yaw_cal, sigma=1.8)
    pitch_cal = gaussian_filter1d(pitch_cal, sigma=1.8)
    roll_cal = gaussian_filter1d(roll_cal, sigma=1.8)

    # Fuse syllable-locked Whisper ARPABET lip-sync with speech-gated mocap dynamics:
    # - Guarantees 0ms drift, crisp consonants, and clean silence closures (no beard/mustache jitter)
    # - Preserves 100% real tracked 3D head turns/nods/tilts, real blinks, real eyebrows, and real smiles!
    speech_gate = np.clip((speech_env - 0.05) / 0.18, 0.0, 1.0)
    speech_gate = speech_gate * speech_gate * (3.0 - 2.0 * speech_gate)

    fused_jaw = np.clip(0.82 * w_jaw + 0.22 * jaw_cal * speech_gate, 0.0, 1.0)
    fused_funnel = np.clip(np.maximum(w_funnel, 0.55 * funnel_cal * speech_gate), 0.0, 1.0)
    fused_wide = np.clip(np.maximum(w_wide, 0.70 * wide_cal), 0.0, 1.0)
    fused_close = np.clip(w_close, 0.0, 1.0)
    fused_f = np.clip(np.maximum(w_f, 0.45 * f_cal * speech_gate), 0.0, 1.0)
    fused_upper = np.clip(0.75 * w_upper + 0.25 * upper_cal * speech_gate, 0.0, 1.0)

    frames = []
    for i in range(num_frames):
        frames.append({
            "t": round(float(t_arr[i]), 4),
            "energy": round(float(speech_env[i]), 4),
            "jawOpen": round(float(fused_jaw[i]), 4),
            "mouthFunnel": round(float(fused_funnel[i]), 4),
            "mouthWide": round(float(fused_wide[i]), 4),
            "mouthClose": round(float(fused_close[i]), 4),
            "mouthF": round(float(fused_f[i]), 4),
            "upperLipRaise": round(float(fused_upper[i]), 4),
            "blink": round(float(blink_cal[i]), 4),
            "browRaise": round(float(brow_cal[i]), 4),
            "headYaw": round(float(yaw_cal[i]), 4),
            "headPitch": round(float(pitch_cal[i]), 4),
            "headRoll": round(float(roll_cal[i]), 4),
            "glitch": round(float(glitch[i]), 4),
            "spectrum": [round(float(v), 3) for v in spec_bands[i]],
        })

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    timeline = {
        "version": 1,
        "mode": "mocap",
        "fps": args.fps,
        "duration": round(duration, 4),
        "numFrames": num_frames,
        "trackedFrames": tracked_count,
        "frames": frames,
    }
    with open(out_path, "w") as f:
        json.dump(timeline, f)

    summary = {
        "status": "ok",
        "mode": "mocap",
        "duration": round(duration, 4),
        "numFrames": num_frames,
        "trackedFrames": tracked_count,
        "fps": args.fps,
        "timeline": str(out_path.resolve()),
    }
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
