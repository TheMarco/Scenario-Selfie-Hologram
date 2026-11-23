#!/usr/bin/env python3
"""
Make the opening frame for an audio-to-video talking-head model from a built bust's
front panel: a 9:16 (or 16:9) crop centered on the face, on the panel's own background.
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def main():
    parser = argparse.ArgumentParser(description="Crop the bust's front panel into an opening frame for LTX")
    parser.add_argument("--bust", required=True, help="Bust directory (contains panel_front.png and bust_meta.json)")
    parser.add_argument("--out", required=True, help="Output PNG path")
    parser.add_argument("--aspect", default="9:16", choices=["9:16", "16:9"])
    parser.add_argument("--height", type=int, default=1280, help="Output height in pixels")
    args = parser.parse_args()

    bust = Path(args.bust)
    hires = bust / "panel_front_hires.png"
    panel = cv2.imread(str(hires if hires.exists() else bust / "panel_front.png"))
    if panel is None:
        raise SystemExit(f"panel_front.png not found in {bust}")
    meta = json.loads((bust / "bust_meta.json").read_text())
    H, W = panel.shape[:2]
    front_meta = meta.get("frontPanel", {})
    face = front_meta.get("faceBox")
    # faceBox is in the coordinates of the (resampled) panel used for rigging
    box_w = float(front_meta.get("size", [W, H])[0])
    cx = 0.5 * (face[0] + face[2]) * (W / box_w) if face else W / 2.0

    aw, ah = (9, 16) if args.aspect == "9:16" else (16, 9)
    out_h = args.height
    out_w = int(round(out_h * aw / ah / 2.0)) * 2

    # Fit the full panel height, then crop or pad horizontally around the face center.
    scale = out_h / float(H)
    resized = cv2.resize(panel, (int(round(W * scale)), out_h), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC)
    rw = resized.shape[1]
    bg = np.median(np.concatenate([resized[:10, :10].reshape(-1, 3), resized[:10, -10:].reshape(-1, 3)]), axis=0)
    canvas = np.empty((out_h, out_w, 3), np.uint8)
    canvas[:] = bg.astype(np.uint8)
    src_cx = cx * scale
    x_src0 = int(round(src_cx - out_w / 2.0))
    # Copy the overlapping column range
    dst0 = max(0, -x_src0)
    src0 = max(0, x_src0)
    n = min(rw - src0, out_w - dst0)
    if n > 0:
        canvas[:, dst0:dst0 + n] = resized[:, src0:src0 + n]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), canvas)
    print(json.dumps({"status": "ok", "frame": str(out.resolve()), "width": out_w, "height": out_h, "aspect": args.aspect}))


if __name__ == "__main__":
    main()
