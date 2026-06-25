#!/usr/bin/env python3
"""Extract PIDNet dataset frames from a ROS 2 bag and edit masks in a web UI."""

from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import os
import random
import shutil
import sys
import threading
import webbrowser
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse


DEFAULT_IMAGE_TOPIC = "/camera/camera/infra1/image_rect_raw"
DEFAULT_MASK_TOPIC = "/segmentation/floor_prob_stable"
DEFAULT_OVERLAY_TOPIC = "/segmentation/floor_overlay"


@dataclass(frozen=True)
class FrameItem:
    frame_id: str
    stamp_ns: int
    image: str
    mask_current: str | None
    mask_edit: str
    overlay: str | None


def _lazy_cv2_np():
    import cv2  # type: ignore
    import numpy as np  # type: ignore

    return cv2, np


def _read_image_msg(msg: Any) -> Any:
    cv2, np = _lazy_cv2_np()
    h = int(msg.height)
    w = int(msg.width)
    enc = str(msg.encoding).lower()
    data = np.frombuffer(msg.data, dtype=np.uint8)

    if enc in ("mono8", "8uc1"):
        return data.reshape((h, w))
    if enc in ("bgr8", "rgb8"):
        arr = data.reshape((h, w, 3))
        if enc == "rgb8":
            arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        return arr
    if enc in ("bgra8", "rgba8"):
        arr = data.reshape((h, w, 4))
        code = cv2.COLOR_RGBA2BGR if enc == "rgba8" else cv2.COLOR_BGRA2BGR
        return cv2.cvtColor(arr, code)
    if enc in ("16uc1", "mono16"):
        arr16 = data.view(np.uint16).reshape((h, w))
        maxv = max(1, int(arr16.max()))
        return ((arr16.astype(np.float32) / maxv) * 255.0).astype(np.uint8)

    raise ValueError(f"unsupported image encoding: {msg.encoding}")


def _nearest(items: list[tuple[int, Any]], stamp_ns: int, max_delta_ns: int) -> tuple[int, Any] | None:
    if not items:
        return None
    best = min(items, key=lambda item: abs(item[0] - stamp_ns))
    if abs(best[0] - stamp_ns) > max_delta_ns:
        return None
    return best


def _load_manifest(workspace: Path) -> list[FrameItem]:
    manifest_path = workspace / "manifest.json"
    if not manifest_path.exists():
        return []
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    return [FrameItem(**item) for item in data.get("frames", [])]


def _write_manifest(workspace: Path, frames: list[FrameItem]) -> None:
    payload = {"frames": [frame.__dict__ for frame in frames]}
    (workspace / "manifest.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _prepare_workspace(workspace: Path) -> None:
    for name in ("images", "masks_current", "masks_edit", "overlays", "exports"):
        (workspace / name).mkdir(parents=True, exist_ok=True)


def extract_bag(
    bag: Path,
    workspace: Path,
    image_topic: str,
    mask_topic: str,
    overlay_topic: str,
    stride: int,
    max_frames: int,
    max_delta_ms: float,
) -> list[FrameItem]:
    try:
        import rosbag2_py  # type: ignore
        from rclpy.serialization import deserialize_message  # type: ignore
        from rosidl_runtime_py.utilities import get_message  # type: ignore
    except ImportError as exc:
        raise SystemExit(
            "This extractor must run in a ROS 2 Python environment. "
            "Run it inside the tinynav container or source the ROS setup first."
        ) from exc

    cv2, np = _lazy_cv2_np()
    _prepare_workspace(workspace)

    storage_options = rosbag2_py.StorageOptions(uri=str(bag), storage_id="sqlite3")
    converter_options = rosbag2_py.ConverterOptions(
        input_serialization_format="cdr",
        output_serialization_format="cdr",
    )
    reader = rosbag2_py.SequentialReader()
    reader.open(storage_options, converter_options)

    topic_types = {topic.name: topic.type for topic in reader.get_all_topics_and_types()}
    wanted = {image_topic, mask_topic, overlay_topic}
    missing = sorted(topic for topic in wanted if topic not in topic_types)
    if missing:
        raise SystemExit(f"bag is missing topic(s): {', '.join(missing)}")

    msg_types = {topic: get_message(topic_types[topic]) for topic in wanted}
    images: list[tuple[int, Any]] = []
    masks: list[tuple[int, Any]] = []
    overlays: list[tuple[int, Any]] = []

    print(f"Reading bag: {bag}")
    while reader.has_next():
        topic, data, stamp_ns = reader.read_next()
        if topic not in wanted:
            continue
        msg = deserialize_message(data, msg_types[topic])
        arr = _read_image_msg(msg)
        if topic == image_topic:
            images.append((stamp_ns, arr))
        elif topic == mask_topic:
            masks.append((stamp_ns, arr))
        elif topic == overlay_topic:
            overlays.append((stamp_ns, arr))

    max_delta_ns = int(max_delta_ms * 1_000_000)
    selected = images[:: max(1, stride)]
    if max_frames > 0:
        selected = selected[:max_frames]

    frames: list[FrameItem] = []
    for idx, (stamp_ns, image) in enumerate(selected, start=1):
        frame_id = f"{idx:06d}"
        image_rel = f"images/{frame_id}.png"
        mask_current_rel = f"masks_current/{frame_id}.png"
        mask_edit_rel = f"masks_edit/{frame_id}.png"
        overlay_rel = f"overlays/{frame_id}.png"

        cv2.imwrite(str(workspace / image_rel), image)

        current_match = _nearest(masks, stamp_ns, max_delta_ns)
        if current_match is not None:
            mask = current_match[1]
            if mask.ndim == 3:
                mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
            binary = (mask >= 128).astype(np.uint8)
            edit_mask = np.where(binary > 0, 1, 2).astype(np.uint8)
            cv2.imwrite(str(workspace / mask_current_rel), binary * 255)
        else:
            edit_mask = np.zeros(image.shape[:2], dtype=np.uint8)
            mask_current_rel = None

        cv2.imwrite(str(workspace / mask_edit_rel), edit_mask)

        overlay_match = _nearest(overlays, stamp_ns, max_delta_ns)
        if overlay_match is not None:
            cv2.imwrite(str(workspace / overlay_rel), overlay_match[1])
        else:
            overlay_rel = None

        frames.append(
            FrameItem(
                frame_id=frame_id,
                stamp_ns=stamp_ns,
                image=image_rel,
                mask_current=mask_current_rel,
                mask_edit=mask_edit_rel,
                overlay=overlay_rel,
            )
        )

    _write_manifest(workspace, frames)
    print(f"Extracted {len(frames)} frame(s) into {workspace}")
    return frames


def export_dataset(workspace: Path, val_ratio: float) -> Path:
    frames = _load_manifest(workspace)
    if not frames:
        raise ValueError("no frames found; extract a bag first")

    export_dir = workspace / "exports" / "tinynav_floor_dataset"
    if export_dir.exists():
        shutil.rmtree(export_dir)
    for split in ("train", "val"):
        (export_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (export_dir / "masks" / split).mkdir(parents=True, exist_ok=True)

    rng = random.Random(42)
    shuffled = list(frames)
    rng.shuffle(shuffled)
    val_count = int(round(len(shuffled) * max(0.0, min(0.9, val_ratio))))
    val_ids = {frame.frame_id for frame in shuffled[:val_count]}

    for frame in frames:
        split = "val" if frame.frame_id in val_ids else "train"
        shutil.copy2(workspace / frame.image, export_dir / "images" / split / f"{frame.frame_id}.png")
        shutil.copy2(workspace / frame.mask_edit, export_dir / "masks" / split / f"{frame.frame_id}.png")

    labels = {
        "0": "ignore",
        "1": "floor",
        "2": "non_floor",
    }
    (export_dir / "labels.json").write_text(json.dumps(labels, indent=2), encoding="utf-8")
    return export_dir


INDEX_HTML = r"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>TinyNav PIDNet Dataset Tool</title>
  <style>
    :root { color-scheme: dark; font-family: system-ui, -apple-system, sans-serif; }
    body { margin: 0; background: #14171a; color: #e7edf3; }
    header { height: 52px; display: flex; align-items: center; gap: 12px; padding: 0 16px; background: #20262d; border-bottom: 1px solid #303842; }
    button, input, select { background: #2a323b; color: #e7edf3; border: 1px solid #45505c; border-radius: 6px; padding: 7px 10px; }
    button.active { outline: 2px solid #78d28b; }
    main { display: grid; grid-template-columns: 300px 1fr; min-height: calc(100vh - 52px); }
    aside { border-right: 1px solid #303842; overflow: auto; padding: 12px; }
    .thumb { display: grid; grid-template-columns: 72px 1fr; gap: 10px; align-items: center; padding: 8px; border-radius: 6px; cursor: pointer; }
    .thumb.active { background: #313a44; }
    .thumb img { width: 72px; height: 48px; object-fit: cover; border-radius: 4px; }
    section { padding: 14px; overflow: auto; }
    .toolbar { display: flex; align-items: center; flex-wrap: wrap; gap: 8px; margin-bottom: 12px; }
    .stage { display: grid; grid-template-columns: repeat(2, minmax(320px, 1fr)); gap: 12px; }
    .panel { background: #1b2026; border: 1px solid #303842; border-radius: 8px; padding: 10px; }
    .panel h3 { font-size: 13px; margin: 0 0 8px; color: #aeb8c2; font-weight: 600; }
    .canvasWrap { position: relative; width: 100%; background: #090b0d; }
    canvas, .canvasWrap img { display: block; width: 100%; height: auto; image-rendering: auto; }
    #editImage, #editMask { position: absolute; left: 0; top: 0; }
    .legend { display: flex; gap: 12px; align-items: center; color: #aeb8c2; font-size: 13px; }
    .dot { width: 12px; height: 12px; border-radius: 2px; display: inline-block; margin-right: 5px; vertical-align: -2px; }
  </style>
</head>
<body>
<header>
  <strong>PIDNet Dataset Tool</strong>
  <button id="prevBtn">Prev</button>
  <button id="nextBtn">Next</button>
  <span id="counter"></span>
  <span style="flex:1"></span>
  <label>Val ratio <input id="valRatio" type="number" min="0" max="0.9" step="0.05" value="0.15" style="width:70px"></label>
  <button id="exportBtn">Export dataset</button>
</header>
<main>
  <aside id="list"></aside>
  <section>
    <div class="toolbar">
      <button data-class="1" class="brush active">Floor</button>
      <button data-class="2" class="brush">Non-floor</button>
      <button data-class="0" class="brush">Ignore</button>
      <label>Brush <input id="brushSize" type="range" min="2" max="80" value="18"></label>
      <button id="saveBtn">Save mask</button>
      <span id="status"></span>
      <span class="legend">
        <span><span class="dot" style="background:#24d65f"></span>floor</span>
        <span><span class="dot" style="background:#f05050"></span>non-floor</span>
        <span><span class="dot" style="background:#20242a"></span>ignore</span>
      </span>
    </div>
    <div class="stage">
      <div class="panel"><h3>Raw image</h3><img id="rawImage"></div>
      <div class="panel"><h3>Current overlay</h3><img id="overlayImage"></div>
      <div class="panel"><h3>Current PIDNet mask</h3><img id="currentMask"></div>
      <div class="panel">
        <h3>Editable training mask</h3>
        <div class="canvasWrap" id="editWrap">
          <canvas id="editBase"></canvas>
          <canvas id="editMask"></canvas>
        </div>
      </div>
    </div>
  </section>
</main>
<script>
let frames = [];
let index = 0;
let brushClass = 1;
let maskData = null;
let maskWidth = 0;
let maskHeight = 0;
let drawing = false;

const $ = (id) => document.getElementById(id);

function rel(path) { return path ? `/file/${path}` : ""; }

async function loadFrames() {
  const res = await fetch("/api/frames");
  frames = await res.json();
  renderList();
  await selectFrame(0);
}

function renderList() {
  $("list").innerHTML = "";
  frames.forEach((frame, i) => {
    const item = document.createElement("div");
    item.className = "thumb" + (i === index ? " active" : "");
    item.onclick = () => selectFrame(i);
    item.innerHTML = `<img src="${rel(frame.image)}"><div><strong>${frame.frame_id}</strong><br><small>${new Date(frame.stamp_ns / 1000000).toISOString()}</small></div>`;
    $("list").appendChild(item);
  });
}

async function selectFrame(i) {
  if (!frames.length) return;
  index = Math.max(0, Math.min(frames.length - 1, i));
  const frame = frames[index];
  $("counter").textContent = `${index + 1} / ${frames.length}`;
  $("rawImage").src = rel(frame.image);
  $("overlayImage").src = frame.overlay ? rel(frame.overlay) : rel(frame.image);
  $("currentMask").src = frame.mask_current ? rel(frame.mask_current) : rel(frame.mask_edit);
  await loadEditableMask();
  renderList();
}

async function loadEditableMask() {
  const res = await fetch(`/api/mask?index=${index}`);
  const payload = await res.json();
  maskWidth = payload.width;
  maskHeight = payload.height;
  const bin = atob(payload.data);
  maskData = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) maskData[i] = bin.charCodeAt(i);

  const base = $("editBase");
  const mask = $("editMask");
  base.width = mask.width = maskWidth;
  base.height = mask.height = maskHeight;
  const ctx = base.getContext("2d");
  const img = new Image();
  img.onload = () => ctx.drawImage(img, 0, 0, maskWidth, maskHeight);
  img.src = rel(frames[index].image);
  drawMask();
}

function drawMask() {
  const canvas = $("editMask");
  const ctx = canvas.getContext("2d");
  const image = ctx.createImageData(maskWidth, maskHeight);
  for (let i = 0; i < maskData.length; i++) {
    const p = i * 4;
    const cls = maskData[i];
    if (cls === 1) {
      image.data[p] = 36; image.data[p + 1] = 214; image.data[p + 2] = 95; image.data[p + 3] = 105;
    } else if (cls === 2) {
      image.data[p] = 240; image.data[p + 1] = 80; image.data[p + 2] = 80; image.data[p + 3] = 105;
    } else {
      image.data[p] = 32; image.data[p + 1] = 36; image.data[p + 2] = 42; image.data[p + 3] = 80;
    }
  }
  ctx.putImageData(image, 0, 0);
}

function paint(ev) {
  if (!drawing || !maskData) return;
  const rect = $("editMask").getBoundingClientRect();
  const x = Math.round((ev.clientX - rect.left) * maskWidth / rect.width);
  const y = Math.round((ev.clientY - rect.top) * maskHeight / rect.height);
  const r = Number($("brushSize").value);
  for (let yy = y - r; yy <= y + r; yy++) {
    if (yy < 0 || yy >= maskHeight) continue;
    for (let xx = x - r; xx <= x + r; xx++) {
      if (xx < 0 || xx >= maskWidth) continue;
      if ((xx - x) * (xx - x) + (yy - y) * (yy - y) <= r * r) {
        maskData[yy * maskWidth + xx] = brushClass;
      }
    }
  }
  drawMask();
}

async function saveMask() {
  let bin = "";
  for (let i = 0; i < maskData.length; i++) bin += String.fromCharCode(maskData[i]);
  const res = await fetch("/api/mask", {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({index, width: maskWidth, height: maskHeight, data: btoa(bin)}),
  });
  $("status").textContent = res.ok ? "Saved" : "Save failed";
}

async function exportDataset() {
  const res = await fetch("/api/export", {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({val_ratio: Number($("valRatio").value)}),
  });
  const payload = await res.json();
  $("status").textContent = res.ok ? `Exported: ${payload.path}` : payload.error;
}

document.querySelectorAll(".brush").forEach((button) => {
  button.onclick = () => {
    document.querySelectorAll(".brush").forEach((b) => b.classList.remove("active"));
    button.classList.add("active");
    brushClass = Number(button.dataset.class);
  };
});

$("prevBtn").onclick = () => selectFrame(index - 1);
$("nextBtn").onclick = () => selectFrame(index + 1);
$("saveBtn").onclick = saveMask;
$("exportBtn").onclick = exportDataset;
$("editMask").addEventListener("pointerdown", (ev) => { drawing = true; paint(ev); });
$("editMask").addEventListener("pointermove", paint);
window.addEventListener("pointerup", () => { drawing = false; });
window.addEventListener("keydown", (ev) => {
  if (ev.key === "ArrowLeft") selectFrame(index - 1);
  if (ev.key === "ArrowRight") selectFrame(index + 1);
  if (ev.key === "1") document.querySelector('[data-class="1"]').click();
  if (ev.key === "2") document.querySelector('[data-class="2"]').click();
  if (ev.key === "0") document.querySelector('[data-class="0"]').click();
  if ((ev.metaKey || ev.ctrlKey) && ev.key.toLowerCase() === "s") { ev.preventDefault(); saveMask(); }
});

loadFrames();
</script>
</body>
</html>
"""


class DatasetHandler(BaseHTTPRequestHandler):
    workspace: Path

    def _send(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _json(self, status: int, payload: Any) -> None:
        self._send(status, json.dumps(payload).encode("utf-8"), "application/json")

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self._send(200, INDEX_HTML.encode("utf-8"), "text/html; charset=utf-8")
            return
        if parsed.path == "/api/frames":
            self._json(200, [frame.__dict__ for frame in _load_manifest(self.workspace)])
            return
        if parsed.path == "/api/mask":
            self._handle_get_mask(parsed.query)
            return
        if parsed.path.startswith("/file/"):
            rel = parsed.path.removeprefix("/file/")
            path = (self.workspace / rel).resolve()
            if not path.is_relative_to(self.workspace.resolve()) or not path.exists():
                self._json(404, {"error": "file not found"})
                return
            content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
            self._send(200, path.read_bytes(), content_type)
            return
        self._json(404, {"error": "not found"})

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        length = int(self.headers.get("Content-Length", "0"))
        payload = json.loads(self.rfile.read(length).decode("utf-8") or "{}")
        try:
            if parsed.path == "/api/mask":
                self._handle_post_mask(payload)
                return
            if parsed.path == "/api/export":
                out = export_dataset(self.workspace, float(payload.get("val_ratio", 0.15)))
                self._json(200, {"path": str(out)})
                return
        except Exception as exc:  # pragma: no cover - server error reporting
            self._json(500, {"error": str(exc)})
            return
        self._json(404, {"error": "not found"})

    def _handle_get_mask(self, query: str) -> None:
        cv2, _ = _lazy_cv2_np()
        frames = _load_manifest(self.workspace)
        index = int(parse_qs(query).get("index", ["0"])[0])
        frame = frames[index]
        mask = cv2.imread(str(self.workspace / frame.mask_edit), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError(f"failed to read mask: {frame.mask_edit}")
        data = base64.b64encode(mask.tobytes()).decode("ascii")
        self._json(200, {"width": int(mask.shape[1]), "height": int(mask.shape[0]), "data": data})

    def _handle_post_mask(self, payload: dict[str, Any]) -> None:
        cv2, np = _lazy_cv2_np()
        frames = _load_manifest(self.workspace)
        frame = frames[int(payload["index"])]
        width = int(payload["width"])
        height = int(payload["height"])
        raw = base64.b64decode(payload["data"])
        mask = np.frombuffer(raw, dtype=np.uint8).reshape((height, width))
        cv2.imwrite(str(self.workspace / frame.mask_edit), mask)
        self._json(200, {"ok": True})

    def log_message(self, fmt: str, *args: Any) -> None:
        sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))


def serve(workspace: Path, host: str, port: int, open_browser: bool) -> None:
    if not _load_manifest(workspace):
        raise SystemExit(f"no manifest found in {workspace}; pass --bag first or use an extracted workspace")

    DatasetHandler.workspace = workspace.resolve()
    server = ThreadingHTTPServer((host, port), DatasetHandler)
    url = f"http://{host}:{port}"
    print(f"Dataset editor: {url}")
    print(f"Workspace: {workspace}")
    if open_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    server.serve_forever()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bag", type=Path, help="ROS 2 bag directory to extract")
    parser.add_argument("--workspace", type=Path, default=Path("pidnet_dataset_workspace"))
    parser.add_argument("--image-topic", default=DEFAULT_IMAGE_TOPIC)
    parser.add_argument("--mask-topic", default=DEFAULT_MASK_TOPIC)
    parser.add_argument("--overlay-topic", default=DEFAULT_OVERLAY_TOPIC)
    parser.add_argument("--stride", type=int, default=5, help="take one raw image every N frames")
    parser.add_argument("--max-frames", type=int, default=0, help="0 means no limit")
    parser.add_argument("--max-delta-ms", type=float, default=150.0, help="max timestamp gap for mask/overlay pairing")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-serve", action="store_true", help="extract only")
    parser.add_argument("--open-browser", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    workspace = args.workspace.resolve()
    _prepare_workspace(workspace)

    if args.bag:
        extract_bag(
            bag=args.bag,
            workspace=workspace,
            image_topic=args.image_topic,
            mask_topic=args.mask_topic,
            overlay_topic=args.overlay_topic,
            stride=args.stride,
            max_frames=args.max_frames,
            max_delta_ms=args.max_delta_ms,
        )

    if not args.no_serve:
        serve(workspace, args.host, args.port, args.open_browser)


if __name__ == "__main__":
    main()
