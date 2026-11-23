#!/usr/bin/env python3
"""
Split spoken-word audio into segments no longer than --max-sec, cutting in the quietest
pause available, so each segment fits audio-to-video models with a length cap
(Scenario LTX-2.5 Fast Audio to Video accepts up to 20 s at 720p/1080p).

Writes 48 kHz mono 16-bit WAV segments plus segments.json, and prints the same JSON.
"""

import argparse
import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from scipy.io import wavfile

FRAME_SEC = 0.02  # 20 ms analysis frames


def decode(audio_path: Path, sr: int) -> np.ndarray:
    with tempfile.NamedTemporaryFile(suffix=".wav") as tmp:
        subprocess.run([
            "ffmpeg", "-y", "-v", "error", "-i", str(audio_path),
            "-ac", "1", "-ar", str(sr), "-c:a", "pcm_s16le", tmp.name,
        ], check=True)
        _, data = wavfile.read(tmp.name)
    return data


def frame_rms(samples: np.ndarray, sr: int) -> np.ndarray:
    hop = int(sr * FRAME_SEC)
    n = len(samples) // hop
    x = samples[: n * hop].astype(np.float32).reshape(n, hop) / 32768.0
    return np.sqrt((x * x).mean(axis=1) + 1e-12)


def choose_cuts(rms: np.ndarray, total_sec: float, max_sec: float, min_sec: float) -> list[float]:
    """Greedy cuts: each cut lands in the quietest 300 ms window between 55% and 100% of max_sec."""
    win = max(1, int(round(0.30 / FRAME_SEC)))
    smooth = np.convolve(rms, np.ones(win) / win, mode="same")
    cuts = [0.0]
    start = 0.0
    while total_sec - start > max_sec:
        lo = int((start + max(min_sec, 0.55 * max_sec)) / FRAME_SEC)
        hi = int((start + max_sec) / FRAME_SEC)
        hi = min(hi, len(smooth) - 1)
        if hi <= lo:
            cut = start + max_sec
        else:
            cut = (lo + int(np.argmin(smooth[lo:hi]))) * FRAME_SEC
        cuts.append(round(cut, 3))
        start = cut
    cuts.append(round(total_sec, 3))
    return cuts


def main():
    parser = argparse.ArgumentParser(description="Split audio at pauses into segments of at most --max-sec seconds")
    parser.add_argument("--audio", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--max-sec", type=float, default=19.5)
    parser.add_argument("--min-sec", type=float, default=3.0)
    args = parser.parse_args()

    sr = 48000
    audio_path = Path(args.audio).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    samples = decode(audio_path, sr)
    total_sec = len(samples) / float(sr)
    cuts = choose_cuts(frame_rms(samples, sr), total_sec, args.max_sec, args.min_sec)

    segments = []
    for i in range(len(cuts) - 1):
        s0 = int(round(cuts[i] * sr))
        s1 = int(round(cuts[i + 1] * sr))
        seg_path = out_dir / f"segment_{i + 1:02d}.wav"
        wavfile.write(str(seg_path), sr, samples[s0:s1])
        segments.append({
            "index": i + 1,
            "path": str(seg_path),
            "start": round(s0 / sr, 4),
            "end": round(s1 / sr, 4),
            "duration": round((s1 - s0) / sr, 4),
        })

    result = {
        "status": "ok",
        "source": str(audio_path),
        "durationSec": round(total_sec, 4),
        "maxSec": args.max_sec,
        "segments": segments,
    }
    (out_dir / "segments.json").write_text(json.dumps(result, indent=2))
    result["plan"] = str(out_dir / "segments.json")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
