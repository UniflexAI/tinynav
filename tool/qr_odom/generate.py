#!/usr/bin/env python3
"""Generate a satellite QR code to use as a spatial anchor.

Writes:
  qr_satellite.png  — preview image
  qr_satellite.pdf  — A4 print-ready PDF (QR active area at exact physical size)
  qr_satellite.json — parameters loaded by qr_target_node.py record mode

Usage:
  python qr_generate.py                         # random UUID, 0.20 m
  python qr_generate.py <content> <size_m>      # e.g. python qr_generate.py dock-A 0.15

size_m is the physical side length of the QR active area (corner-to-corner of the
finder patterns, quiet zone excluded).  Red corner marks in the PDF indicate the
active area boundary.  After printing at 100% scale, measure between the marks
and update size_m in qr_satellite.json if it differs from the intended value.
"""

import io
import json
import sys
import uuid
from pathlib import Path

import cv2
import numpy as np
from PIL import Image as PILImage
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas

OUT_PNG = Path("qr_satellite.png")
OUT_PDF = Path("qr_satellite.pdf")
OUT_JSON = Path("qr_satellite.json")

A4_W, A4_H = A4   # points (595.28 × 841.89)


def _encode_qr(content: str) -> np.ndarray:
    """Return the raw QR module grid as a uint8 grayscale array (0=black, 255=white)."""
    return cv2.QRCodeEncoder.create().encode(content)


def _save_png(qr_modules: np.ndarray, border_px: int = 20, module_px: int = 10) -> None:
    n = qr_modules.shape[0]
    scaled = cv2.resize(qr_modules, (n * module_px, n * module_px),
                        interpolation=cv2.INTER_NEAREST)
    with_border = cv2.copyMakeBorder(
        scaled, border_px, border_px, border_px, border_px,
        cv2.BORDER_CONSTANT, value=255,
    )
    cv2.imwrite(str(OUT_PNG), with_border)


def _save_pdf(qr_modules: np.ndarray, content: str, size_m: float) -> None:
    n = qr_modules.shape[0]
    size_mm = size_m * 1000
    active_pt = size_mm * mm          # active area in PDF points
    quiet_pt = 4 * active_pt / n      # standard 4-module quiet zone

    # Page centre
    cx, cy = A4_W / 2, A4_H / 2

    # Active-area bounding box (bottom-left origin, as reportlab uses)
    ax0 = cx - active_pt / 2
    ay0 = cy - active_pt / 2

    c = canvas.Canvas(str(OUT_PDF), pagesize=A4)

    # ── QR image (active area only, no quiet zone) ──────────────────────
    pil = PILImage.fromarray(qr_modules).convert("RGB")
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    buf.seek(0)
    c.drawImage(ImageReader(buf), ax0, ay0, width=active_pt, height=active_pt,
                preserveAspectRatio=True)

    # ── Red corner registration marks ────────────────────────────────────
    tick = 5 * mm
    c.setStrokeColorRGB(0.9, 0, 0)
    c.setLineWidth(0.6)
    for sx, sy in [
        (ax0,              ay0),
        (ax0 + active_pt,  ay0),
        (ax0,              ay0 + active_pt),
        (ax0 + active_pt,  ay0 + active_pt),
    ]:
        dx = tick if sx < cx else -tick
        dy = tick if sy < cy else -tick
        c.line(sx, sy, sx + dx, sy)
        c.line(sx, sy, sx, sy + dy)

    # ── Horizontal dimension line below the QR ───────────────────────────
    dim_y = ay0 - 8 * mm
    c.setStrokeColorRGB(0, 0, 0)
    c.setLineWidth(0.5)
    c.line(ax0, dim_y, ax0 + active_pt, dim_y)
    c.line(ax0, dim_y - 2 * mm, ax0, dim_y + 2 * mm)
    c.line(ax0 + active_pt, dim_y - 2 * mm, ax0 + active_pt, dim_y + 2 * mm)

    # ── Labels ────────────────────────────────────────────────────────────
    c.setFillColorRGB(0, 0, 0)
    c.setFont("Helvetica-Bold", 10)
    c.drawCentredString(cx, dim_y - 5 * mm, f"{size_mm:.0f} mm")

    c.setFont("Helvetica", 8)
    c.drawCentredString(cx, ay0 - 17 * mm,
                        f"Satellite QR  |  active area: {size_mm:.0f} × {size_mm:.0f} mm  |  {content}")
    c.setFillColorRGB(0.4, 0.4, 0.4)
    c.drawCentredString(cx, ay0 - 22 * mm,
                        "Print at 100% scale (no 'fit to page'). "
                        "Measure between red corner marks. "
                        f"Update size_m in {OUT_JSON} if it differs.")

    c.save()


def generate(content: str, size_m: float) -> None:
    qr_modules = _encode_qr(content)
    _save_png(qr_modules)
    _save_pdf(qr_modules, content, size_m)
    OUT_JSON.write_text(json.dumps({"content": content, "size_m": size_m}, indent=2))

    size_mm = size_m * 1000
    print(f"Content : {content}")
    print(f"size_m  : {size_m} m  ({size_mm:.0f} mm)  — active area, quiet zone excluded")
    print(f"PNG     : {OUT_PNG}")
    print(f"PDF     : {OUT_PDF}  ← print this on A4 at 100% scale")
    print(f"JSON    : {OUT_JSON}")
    print()
    print("After printing: measure between the red corner marks.")
    print(f"They should be {size_mm:.0f} mm apart. If not, update size_m in {OUT_JSON}.")


def main() -> None:
    content = sys.argv[1] if len(sys.argv) > 1 else str(uuid.uuid4())
    size_m = float(sys.argv[2]) if len(sys.argv) > 2 else 0.20
    generate(content, size_m)


if __name__ == "__main__":
    main()
