#!/usr/bin/env python3
"""Generate an AprilTag 36h11 GridBoard (2×2) for use as a spatial anchor.

AprilTag 36h11 has a 6×6 data bit grid and minimum Hamming distance 11 — the
most robust AprilTag family: largest cells for a given physical size, fewest
false detections.  Using a GridBoard (4 tags) instead of a single tag gives a
more stable and accurate pose estimate because all visible tags contribute to
one joint solvePnP call.

Writes:
  tag_board.png       — preview image
  tag_board.pdf       — A4 print-ready PDF at exact physical size
  tag_satellite.json  — board params loaded by qr_target_node.py

Usage:
  python apriltag_generate.py                        # defaults below
  python apriltag_generate.py <base_id> <size_m> <spacing_m>

Defaults:
  base_id   = 0    → tag ids 0, 1, 2, 3
  size_m    = 0.08 → 80 mm per-tag active area
  spacing_m = 0.02 → 20 mm gap between tags  (board total: 180 × 180 mm)

size_m is each tag's individual active area (corner-to-corner of the outer black
border, quiet zone excluded).  Red marks in the PDF show each tag's active-area
boundary.  After printing at 100% scale, measure one tag and update size_m in
tag_satellite.json if it differs.
"""

import io
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image as PILImage
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas

DB_DIR   = Path("tinynav_db/qrcode")
OUT_PNG  = DB_DIR / "tag_board.png"
OUT_PDF  = DB_DIR / "tag_board.pdf"
OUT_JSON = DB_DIR / "tag_satellite.json"

A4_W, A4_H = A4

ARUCO_DICT  = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
FAMILY_NAME = "DICT_APRILTAG_36h11"

TAG_PX   = 300   # pixels per tag active area (sets image resolution)
QUIET_PX = 20    # quiet-zone margin in pixels around the whole board


def _make_board(tag_ids: list[int], size_m: float, spacing_m: float) -> cv2.aruco.GridBoard:
    return cv2.aruco.GridBoard(
        size=(2, 2),
        markerLength=size_m,
        markerSeparation=spacing_m,
        dictionary=ARUCO_DICT,
        ids=np.array(tag_ids, dtype=np.int32),
    )


def _board_image(board: cv2.aruco.GridBoard, size_m: float, spacing_m: float) -> tuple[np.ndarray, int, int]:
    """Return (board_image, active_board_px, margin_px)."""
    spacing_px       = int(round(spacing_m / size_m * TAG_PX))
    active_board_px  = 2 * TAG_PX + spacing_px
    margin_px        = QUIET_PX
    total_px         = active_board_px + 2 * margin_px
    img = board.generateImage((total_px, total_px), marginSize=margin_px, borderBits=1)
    return img, active_board_px, margin_px


def _save_pdf(board_img: np.ndarray, tag_ids: list[int],
              size_m: float, spacing_m: float,
              active_board_px: int, margin_px: int) -> None:
    active_board_mm = (2 * size_m + spacing_m) * 1000
    active_board_pt = active_board_mm * mm

    # Scale entire image so active board area = active_board_pt
    scale        = active_board_pt / active_board_px
    total_img_pt = board_img.shape[0] * scale
    margin_pt    = margin_px * scale

    cx, cy = A4_W / 2, A4_H / 2
    img_x  = cx - total_img_pt / 2
    img_y  = cy - total_img_pt / 2

    c = canvas.Canvas(str(OUT_PDF), pagesize=A4)

    buf = io.BytesIO()
    PILImage.fromarray(board_img).convert("RGB").save(buf, format="PNG")
    buf.seek(0)
    c.drawImage(ImageReader(buf), img_x, img_y,
                width=total_img_pt, height=total_img_pt, preserveAspectRatio=True)

    # Active-area box starts after the margin
    ax0 = img_x + margin_pt
    ay0 = img_y + margin_pt
    ax1 = ax0   + active_board_pt
    ay1 = ay0   + active_board_pt

    # Red corner marks at each tag's active area
    size_pt    = size_m    * 1000 * mm
    spacing_pt = spacing_m * 1000 * mm
    tick = 4 * mm
    c.setStrokeColorRGB(0.9, 0, 0)
    c.setLineWidth(0.5)
    for row in range(2):
        for col in range(2):
            # Tag active-area corner positions inside the board
            tx0 = ax0 + col * (size_pt + spacing_pt)
            ty0 = ay0 + row * (size_pt + spacing_pt)
            tx1 = tx0 + size_pt
            ty1 = ty0 + size_pt
            tag_cx = (tx0 + tx1) / 2
            tag_cy = (ty0 + ty1) / 2
            for sx, sy in [(tx0, ty0), (tx1, ty0), (tx0, ty1), (tx1, ty1)]:
                dx = tick if sx < tag_cx else -tick
                dy = tick if sy < tag_cy else -tick
                c.line(sx, sy, sx + dx, sy)
                c.line(sx, sy, sx, sy + dy)

    # Grey board boundary
    c.setStrokeColorRGB(0.6, 0.6, 0.6)
    c.setLineWidth(0.3)
    c.rect(ax0, ay0, active_board_pt, active_board_pt)

    # Dimension line below board
    dim_y = ay0 - 8 * mm
    c.setStrokeColorRGB(0, 0, 0)
    c.setLineWidth(0.5)
    c.line(ax0, dim_y, ax1, dim_y)
    for bx in (ax0, ax1):
        c.line(bx, dim_y - 2 * mm, bx, dim_y + 2 * mm)

    size_mm    = size_m    * 1000
    spacing_mm = spacing_m * 1000
    c.setFillColorRGB(0, 0, 0)
    c.setFont("Helvetica-Bold", 9)
    c.drawCentredString(cx, dim_y - 5 * mm,
                        f"board: {active_board_mm:.0f} mm  |  per tag: {size_mm:.0f} mm  |  gap: {spacing_mm:.0f} mm")
    c.setFont("Helvetica", 8)
    c.drawCentredString(cx, dim_y - 11 * mm,
                        f"AprilTag 36h11 (6×6)  |  ids: {tag_ids}")
    c.setFillColorRGB(0.4, 0.4, 0.4)
    c.drawCentredString(cx, dim_y - 16 * mm,
                        "Print at 100% scale (no 'fit to page').  "
                        "Measure one tag's active area between its red marks.  "
                        f"Update size_m in {OUT_JSON} if it differs.")
    c.save()


def generate(base_id: int, size_m: float, spacing_m: float) -> None:
    DB_DIR.mkdir(parents=True, exist_ok=True)
    tag_ids = [base_id + i for i in range(4)]
    board   = _make_board(tag_ids, size_m, spacing_m)
    img, active_px, margin_px = _board_image(board, size_m, spacing_m)

    cv2.imwrite(str(OUT_PNG), img)
    _save_pdf(img, tag_ids, size_m, spacing_m, active_px, margin_px)
    OUT_JSON.write_text(json.dumps(
        {"tag_family": FAMILY_NAME, "tag_ids": tag_ids,
         "size_m": size_m, "spacing_m": spacing_m},
        indent=2,
    ))

    size_mm, spacing_mm = size_m * 1000, spacing_m * 1000
    board_mm = 2 * size_mm + spacing_mm
    print(f"Family    : AprilTag 36h11 (6×6 data bits, Hamming distance 11)")
    print(f"Tag IDs   : {tag_ids}")
    print(f"Per tag   : {size_mm:.0f} mm active area")
    print(f"Gap       : {spacing_mm:.0f} mm")
    print(f"Board     : {board_mm:.0f} × {board_mm:.0f} mm total")
    print(f"PNG       : {OUT_PNG}")
    print(f"PDF       : {OUT_PDF}  ← print on A4 at 100% scale")
    print(f"JSON      : {OUT_JSON}")
    print()
    print("After printing: measure one tag's active area between its red corner marks.")
    print(f"Should be {size_mm:.0f} mm.  Update size_m in {OUT_JSON} if it differs.")


def main() -> None:
    base_id   = int(sys.argv[1])   if len(sys.argv) > 1 else 0
    size_m    = float(sys.argv[2]) if len(sys.argv) > 2 else 0.08
    spacing_m = float(sys.argv[3]) if len(sys.argv) > 3 else 0.02
    generate(base_id, size_m, spacing_m)


if __name__ == "__main__":
    main()
