#!/usr/bin/env python3
"""
Draw an empty 3-panel character sheet (FRONT / LEFT / RIGHT) used as a layout reference
for image models. The filled sheet should keep this layout so build_bust.py can split it.

The template contains no person, so it can ship with the tool.
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

WIDTH = 2048
HEIGHT = 1152
MARGIN_X = 26
GUTTER = 20
PANEL_TOP = 112
PANEL_BOTTOM = 1056
BORDER = (190, 190, 190)
INK = (40, 40, 40)
MUTED = (120, 120, 120)


def panel_boxes() -> list[tuple[int, int, int, int]]:
    pw = (WIDTH - 2 * MARGIN_X - 2 * GUTTER) // 3
    boxes = []
    for i in range(3):
        x0 = MARGIN_X + i * (pw + GUTTER)
        boxes.append((x0, PANEL_TOP, x0 + pw, PANEL_BOTTOM))
    return boxes


def centered_text(img, text, cx, y, scale, color, thickness):
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, _), _ = cv2.getTextSize(text, font, scale, thickness)
    cv2.putText(img, text, (int(cx - tw / 2), y), font, scale, color, thickness, cv2.LINE_AA)


def draw_template() -> np.ndarray:
    img = np.full((HEIGHT, WIDTH, 3), 255, np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, "CHARACTER SHEET", (60, 72), font, 1.35, INK, 3, cv2.LINE_AA)
    cv2.line(img, (560, 60), (1560, 60), BORDER, 2)
    cv2.putText(img, "SUBJECT", (1720, 72), font, 0.9, MUTED, 2, cv2.LINE_AA)
    for (x0, y0, x1, y1), label in zip(panel_boxes(), ["FRONT", "LEFT", "RIGHT"]):
        cv2.rectangle(img, (x0, y0), (x1, y1), BORDER, 2)
        centered_text(img, label, (x0 + x1) / 2, y1 + 52, 1.05, INK, 2)
    return img


def main():
    parser = argparse.ArgumentParser(description="Write the empty 3-panel character sheet template")
    parser.add_argument("--out", required=True, help="Output PNG path")
    args = parser.parse_args()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), draw_template())
    print(json.dumps({"status": "ok", "template": str(out.resolve()), "width": WIDTH, "height": HEIGHT}))


if __name__ == "__main__":
    main()
