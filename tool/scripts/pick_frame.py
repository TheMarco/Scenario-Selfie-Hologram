#!/usr/bin/env python3
"""
Pick the best still of a person from a selfie video: facing the camera, eyes open, mouth
closed, sharp and well lit. The still is the identity reference for the character sheet.

Writes the chosen frame at full resolution (auto-rotated, as the phone recorded it) and
prints a JSON report with its time, pose and quality scores, plus warnings an agent can
relay to the user (face too small, too dark, never frontal, never closed-mouth).
"""

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from extract_mocap import iter_video_frames_rgb, rot_matrix_to_euler_yxz  # noqa: E402

ANALYSIS_FPS = 6
ANALYSIS_WIDTH = 540


def probe_duration(video: str) -> float:
    out = subprocess.run([
        "ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", video,
    ], capture_output=True, text=True, check=True).stdout.strip()
    return float(out or 0.0)


def extract_full_frame(video: str, t: float, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([
        "ffmpeg", "-y", "-v", "error", "-ss", f"{t:.3f}", "-i", video,
        "-frames:v", "1", "-q:v", "2", str(out_path),
    ], check=True)


def main():
    parser = argparse.ArgumentParser(description="Pick the best frontal, neutral still from a selfie video")
    parser.add_argument("--video", required=True)
    parser.add_argument("--out", required=True, help="Output JPEG path for the chosen frame")
    parser.add_argument("--landmarker", required=True, help="Path to MediaPipe face_landmarker.task")
    args = parser.parse_args()

    video = str(Path(args.video).resolve())
    duration = probe_duration(video)

    options = mp.tasks.vision.FaceLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=args.landmarker),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_faces=1,
        output_face_blendshapes=True,
        output_facial_transformation_matrixes=True,
    )

    rows = []
    with mp.tasks.vision.FaceLandmarker.create_from_options(options) as landmarker:
        for idx, rgb in enumerate(iter_video_frames_rgb(video, ANALYSIS_FPS, target_width=ANALYSIS_WIDTH)):
            t = idx / float(ANALYSIS_FPS)
            res = landmarker.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), int(t * 1000))
            if not res.face_landmarks:
                continue
            H, W = rgb.shape[:2]
            lm = np.array([[p.x * W, p.y * H] for p in res.face_landmarks[0]], dtype=np.float32)
            bs = {c.category_name: c.score for c in res.face_blendshapes[0]} if res.face_blendshapes else {}
            pitch = yaw = roll = 0.0
            if res.facial_transformation_matrixes:
                pitch, yaw, roll = rot_matrix_to_euler_yxz(np.array(res.facial_transformation_matrixes[0])[:3, :3])
            x0, y0 = np.maximum(lm.min(axis=0).astype(int), 0)
            x1, y1 = lm.max(axis=0).astype(int)
            face = cv2.cvtColor(rgb[y0:max(y1, y0 + 2), x0:max(x1, x0 + 2)], cv2.COLOR_RGB2GRAY)
            rows.append({
                "t": t,
                "yaw": float(yaw),
                "pitch": float(pitch),
                "roll": float(roll),
                "jaw": float(bs.get("jawOpen", 0.0)),
                "blink": 0.5 * float(bs.get("eyeBlinkLeft", 0.0) + bs.get("eyeBlinkRight", 0.0)),
                "smile": 0.5 * float(bs.get("mouthSmileLeft", 0.0) + bs.get("mouthSmileRight", 0.0)),
                "sharp": float(cv2.Laplacian(face, cv2.CV_32F).var()) if face.size else 0.0,
                "bright": float(face.mean() / 255.0) if face.size else 0.0,
                "faceH": float((y1 - y0) / float(H)),
            })

    if not rows:
        print(json.dumps({"status": "error", "error": "No face found in the video.", "durationSec": round(duration, 2)}))
        sys.exit(2)

    sharp = np.array([r["sharp"] for r in rows])
    sharp_rank = sharp.argsort().argsort() / max(1, len(rows) - 1)
    scored = []
    for r, sr in zip(rows, sharp_rank):
        pose = math.hypot(r["yaw"], r["pitch"]) + 0.5 * abs(r["roll"])
        score = (
            0.35 * sr
            + 0.25 * (1.0 - min(1.0, pose / 0.30))
            + 0.20 * (1.0 - min(1.0, r["jaw"] / 0.25))
            + 0.12 * (1.0 - min(1.0, r["blink"] / 0.45))
            + 0.08 * (1.0 - min(1.0, r["smile"] / 0.6))
        )
        # Hard preferences: frontal and eyes open, when the video has such frames at all
        ok = pose < 0.22 and r["blink"] < 0.45
        scored.append((ok, score, r))
    pool = [x for x in scored if x[0]] or scored
    _, best_score, best = max(pool, key=lambda x: x[1])

    warnings = []
    if not any(x[0] for x in scored):
        warnings.append("no frame faces the camera with both eyes open; ask for a take that starts looking straight into the lens")
    if best["jaw"] > 0.2:
        warnings.append("mouth is open in the best frame; a take that starts with 2 seconds of closed-mouth neutral face gives a better likeness")
    if best["faceH"] < 0.22:
        warnings.append(f"face fills only {best['faceH']:.0%} of the frame height; hold the phone closer")
    if best["bright"] < 0.28:
        warnings.append("face is dark; record facing a window or lamp")
    tracked_ratio = len(rows) / max(1.0, duration * ANALYSIS_FPS)
    if tracked_ratio < 0.7:
        warnings.append(f"face was found in only {tracked_ratio:.0%} of sampled frames; mocap will interpolate the gaps")

    out = Path(args.out).resolve()
    extract_full_frame(video, best["t"], out)
    frame = cv2.imread(str(out))
    print(json.dumps({
        "status": "ok",
        "frame": str(out),
        "frameSize": [int(frame.shape[1]), int(frame.shape[0])] if frame is not None else None,
        "timeSec": round(best["t"], 2),
        "durationSec": round(duration, 2),
        "yawDeg": round(math.degrees(best["yaw"]), 1),
        "pitchDeg": round(math.degrees(best["pitch"]), 1),
        "jawOpen": round(best["jaw"], 3),
        "eyeBlink": round(best["blink"], 3),
        "faceHeightFrac": round(best["faceH"], 3),
        "score": round(best_score, 3),
        "framesAnalyzed": len(rows),
        "warnings": warnings,
    }))


if __name__ == "__main__":
    main()
