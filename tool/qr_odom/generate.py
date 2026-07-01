#!/usr/bin/env python3
"""Generate a print-ready PDF for an NxN AprilTag 36h11 GridBoard.

Writes to tinynav_db/qrcode/:
  tag_grid_NxN.pdf   — A4 / A3 PDF at exact physical size
  tag_grid_NxN.json  — board params loaded by odom_node.py / record_node.py

Usage:
  python tool/qr_odom/generate.py --grid 2
  python tool/qr_odom/generate.py --grid 3
  python tool/qr_odom/generate.py --grid 4
  python tool/qr_odom/generate.py --grid 4 --size 0.10 --spacing 0.02

Page selection (auto, based on board size):
  fits on A4  → A4 portrait
  fits on A3  → A3 portrait
"""

import argparse
import io
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image as PILImage
from reportlab.lib.units import mm
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas

DB_DIR      = Path("tinynav_db/qrcode")
ARUCO_DICT  = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
FAMILY_NAME = "DICT_APRILTAG_36h11"

TAG_PX   = 300   # pixels per tag active area (image resolution)
QUIET_PX = 24    # quiet-zone margin in pixels

A4_MM = (210.0, 297.0)   # portrait (width, height)
A3_MM = (297.0, 420.0)


def _page_for(grid: int) -> tuple[float, float]:
    """Grid 1–2 → A4 portrait, grid 3–4 → A3 portrait."""
    return A4_MM if grid <= 2 else A3_MM


def generate(grid: int, size_m: float, spacing_m: float) -> None:
    n          = grid
    tag_ids    = list(range(n * n))
    size_mm    = size_m    * 1000
    spacing_mm = spacing_m * 1000
    board_mm   = n * size_mm + (n - 1) * spacing_mm

    board = cv2.aruco.GridBoard(
        size=(n, n),
        markerLength=size_m,
        markerSeparation=spacing_m,
        dictionary=ARUCO_DICT,
        ids=np.array(tag_ids, dtype=np.int32),
    )

    # Render board image
    spacing_px      = int(round(spacing_m / size_m * TAG_PX))
    active_board_px = n * TAG_PX + (n - 1) * spacing_px
    total_px        = active_board_px + 2 * QUIET_PX
    img = board.generateImage((total_px, total_px), marginSize=QUIET_PX, borderBits=1)

    # Page and scaling
    pw_mm, ph_mm = _page_for(n)
    margin_mm = (pw_mm - board_mm) / 2
    if margin_mm < 0:
        print(f"WARNING: board {board_mm:.0f} mm exceeds {pw_mm:.0f} mm paper width by "
              f"{-margin_mm*2:.0f} mm. Print with 'fit to page' or use smaller --size.")
    elif margin_mm < 10:
        print(f"WARNING: only {margin_mm:.1f} mm margin per side on "
              f"{pw_mm:.0f}×{ph_mm:.0f} mm paper.")

    PW = pw_mm * mm
    PH = ph_mm * mm
    active_board_pt = board_mm * mm
    scale           = active_board_pt / active_board_px
    total_img_pt    = total_px * scale
    margin_pt       = QUIET_PX * scale

    cx    = PW / 2
    cy    = PH / 2
    img_x = cx - total_img_pt / 2
    img_y = cy - total_img_pt / 2
    ax0   = img_x + margin_pt
    ay0   = img_y + margin_pt
    ax1   = ax0 + active_board_pt

    _smm     = int(round(size_m * 1000))
    out_pdf  = DB_DIR / f"tag_grid_{n}x{n}_s{_smm}mm.pdf"
    out_json = DB_DIR / f"tag_grid_{n}x{n}_s{_smm}mm.json"

    c = canvas.Canvas(str(out_pdf), pagesize=(PW, PH))

    # Board image
    buf = io.BytesIO()
    PILImage.fromarray(img).convert("RGB").save(buf, format="PNG")
    buf.seek(0)
    c.drawImage(ImageReader(buf), img_x, img_y,
                width=total_img_pt, height=total_img_pt, preserveAspectRatio=True)

    # Red corner marks per tag
    size_pt    = size_mm    * mm
    spacing_pt = spacing_mm * mm
    tick       = 3.5 * mm
    c.setStrokeColorRGB(0.85, 0, 0)
    c.setLineWidth(0.5)
    for row in range(n):
        for col in range(n):
            tx0 = ax0 + col * (size_pt + spacing_pt)
            ty0 = ay0 + row * (size_pt + spacing_pt)
            tx1 = tx0 + size_pt
            ty1 = ty0 + size_pt
            tcx = (tx0 + tx1) / 2
            tcy = (ty0 + ty1) / 2
            for sx, sy in [(tx0, ty0), (tx1, ty0), (tx0, ty1), (tx1, ty1)]:
                dx = tick if sx < tcx else -tick
                dy = tick if sy < tcy else -tick
                c.line(sx, sy, sx + dx, sy)
                c.line(sx, sy, sx, sy + dy)

    # Board outline
    c.setStrokeColorRGB(0.65, 0.65, 0.65)
    c.setLineWidth(0.3)
    c.rect(ax0, ay0, active_board_pt, active_board_pt)

    # Dimension line
    dim_y = ay0 - 8 * mm
    c.setStrokeColorRGB(0, 0, 0)
    c.setLineWidth(0.5)
    c.line(ax0, dim_y, ax1, dim_y)
    for bx in (ax0, ax1):
        c.line(bx, dim_y - 2 * mm, bx, dim_y + 2 * mm)

    c.setFillColorRGB(0, 0, 0)
    c.setFont("Helvetica-Bold", 9)
    c.drawCentredString(cx, dim_y - 5 * mm,
        f"board: {board_mm:.0f} mm  |  per tag: {size_mm:.0f} mm  |  gap: {spacing_mm:.0f} mm")
    c.setFont("Helvetica", 8)
    c.drawCentredString(cx, dim_y - 11 * mm,
        f"AprilTag 36h11  |  {n}×{n} grid  |  ids: 0 – {n*n - 1}")
    c.setFillColorRGB(0.4, 0.4, 0.4)
    c.drawCentredString(cx, dim_y - 17 * mm,
        f"Print at 100% on {pw_mm:.0f}×{ph_mm:.0f} mm paper (no 'fit to page').  "
        f"Measure one tag between red marks — should be {size_mm:.0f} mm.  "
        f"Update size_m in {out_json.name} if it differs.")
    c.save()

    # JSON
    out_json.write_text(json.dumps({
        "tag_family": FAMILY_NAME,
        "grid":       f"{n}x{n}",
        "tag_ids":    tag_ids,
        "size_m":     size_m,
        "spacing_m":  spacing_m,
    }, indent=2))

    print(f"Grid     : {n}×{n}  ({n*n} tags, ids 0–{n*n-1})")
    print(f"Tag size : {size_mm:.0f} mm  gap: {spacing_mm:.0f} mm  board: {board_mm:.0f}×{board_mm:.0f} mm")
    print(f"Paper    : {pw_mm:.0f}×{ph_mm:.0f} mm")
    print(f"PDF      : {out_pdf}")
    print(f"JSON     : {out_json}")
    print()
    print(f"After printing: measure one tag between its red corner marks.")
    print(f"Should be {size_mm:.0f} mm. Update size_m in {out_json.name} if it differs.")


def main():
    parser = argparse.ArgumentParser(
        description="Generate a print-ready AprilTag GridBoard PDF."
    )
    parser.add_argument("--grid",    type=int,   required=True,
                        help="Grid size N for NxN board (e.g. 2, 3, 4)")
    parser.add_argument("--size",    type=float, default=0.076,
                        help="Tag marker size in meters (default: 0.076)")
    parser.add_argument("--spacing", type=float, default=0.019,
                        help="Tag spacing in meters (default: 0.019)")
    args = parser.parse_args()

    DB_DIR.mkdir(parents=True, exist_ok=True)
    generate(args.grid, args.size, args.spacing)


if __name__ == "__main__":
    main()
