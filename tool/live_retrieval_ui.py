#!/usr/bin/env python3
"""Live top-1 VLAD image-retrieval viewer.

Replays a query rosbag frame by frame, embeds each color frame with the same
DINOv2 + VLAD pipeline used at map-build time, and serves a small web page
showing the live query frame next to the current top-1 matching keyframe
from a pre-built TinyNav map.

Usage:
    uv run python tool/live_retrieval_ui.py \\
        --tinynav_map_path /tinynav/output/map_day_20260716 \\
        --query_bag /tinynav/dataset/202601718/rosbags/bag_2026_07_17_08_38_17
Then open http://127.0.0.1:8642 in a browser.
"""
import argparse
import asyncio
import base64
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import uvicorn
from cv_bridge import CvBridge
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from rclpy.serialization import deserialize_message
from rosbag2_py import ConverterOptions, SequentialReader, StorageFilter, StorageOptions
from rosidl_runtime_py.utilities import get_message

from tinynav.core.build_map_node import TinyNavDB
from tinynav.core.models_trt import Dinov2TRT
from tinynav.core.vlad_retrieval import compute_vlad


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Live top-1 VLAD retrieval viewer")
    parser.add_argument("--tinynav_map_path", required=True, help="Pre-built map directory (from build_map_node.py)")
    parser.add_argument("--query_bag", required=True, help="Rosbag2 directory to replay as the live query stream")
    parser.add_argument("--image_topic", default="/camera/camera/color/image_rect_raw/compressed")
    parser.add_argument("--play_rate", type=float, default=1.0, help="Bag playback speed multiplier")
    parser.add_argument("--query_hz", type=float, default=2.0, help="Max rate at which frames are embedded and matched")
    parser.add_argument("--similarity_threshold", type=float, default=0.5, help="Score above which a match is shown as confident")
    parser.add_argument("--loop", action=argparse.BooleanOptionalAction, default=True, help="Replay the bag on completion")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8642)
    args = parser.parse_args()
    if args.play_rate <= 0.0:
        raise ValueError(f"--play_rate must be > 0, got {args.play_rate}")
    return args


def load_map(map_path: str):
    map_path = Path(map_path)
    poses = np.load(map_path / "poses.npy", allow_pickle=True).item()
    codebook = np.load(map_path / "vlad_codebook.npy")
    db = TinyNavDB(str(map_path), is_scratch=False)
    idx_to_timestamp = {idx: ts for idx, ts in enumerate(poses.keys())}
    embeddings = np.stack([db.get_embedding(ts) for ts in idx_to_timestamp.values()])
    return db, codebook, embeddings, idx_to_timestamp


def encode_jpeg_b64(image: np.ndarray, max_edge_px: int = 640, quality: int = 80) -> str:
    h, w = image.shape[:2]
    scale = min(1.0, max_edge_px / max(h, w))
    if scale < 1.0:
        image = cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("JPEG encode failed")
    return base64.b64encode(buf.tobytes()).decode("ascii")


class RetrievalState:
    """Holds the latest retrieval result; readers only ever see the newest frame."""

    def __init__(self):
        self._lock = threading.Lock()
        self._seq = 0
        self._payload: dict = {"status": "starting"}

    def publish(self, payload: dict) -> None:
        with self._lock:
            self._seq += 1
            self._payload = payload

    def snapshot(self) -> tuple[int, dict]:
        with self._lock:
            return self._seq, self._payload


def bag_message_iterator(bag_path: str, topic: str):
    reader = SequentialReader()
    reader.open(
        StorageOptions(uri=bag_path, storage_id="sqlite3"),
        ConverterOptions(input_serialization_format="cdr", output_serialization_format="cdr"),
    )
    topic_types = {t.name: t.type for t in reader.get_all_topics_and_types()}
    if topic not in topic_types:
        raise ValueError(f"Topic {topic!r} not found in bag {bag_path}; available: {sorted(topic_types)}")
    try:
        reader.set_filter(StorageFilter(topics=[topic]))
    except Exception:
        pass
    msg_type = get_message(topic_types[topic])
    while reader.has_next():
        read_topic, data, timestamp_ns = reader.read_next()
        if read_topic != topic:
            continue
        yield deserialize_message(data, msg_type), int(timestamp_ns)


def retrieval_loop(args: argparse.Namespace, state: RetrievalState) -> None:
    db, codebook, map_embeddings, idx_to_timestamp = load_map(args.tinynav_map_path)
    dinov2_model = Dinov2TRT()
    bridge = CvBridge()
    min_query_interval_ns = int(1e9 / args.query_hz) if args.query_hz > 0 else 0
    frame_index = 0

    while True:
        timestamps = [ts for _, ts in bag_message_iterator(args.query_bag, args.image_topic)]
        if not timestamps:
            state.publish({"status": "error", "message": f"No messages on {args.image_topic} in {args.query_bag}"})
            return
        bag_start_ts_ns, bag_end_ts_ns = timestamps[0], timestamps[-1]

        playback_start_wall_s = None
        playback_start_ts_ns = None
        last_query_ts_ns = None

        for msg, timestamp_ns in bag_message_iterator(args.query_bag, args.image_topic):
            if playback_start_wall_s is None:
                playback_start_wall_s = time.monotonic()
                playback_start_ts_ns = timestamp_ns
            elapsed_bag_s = (timestamp_ns - playback_start_ts_ns) * 1e-9
            sleep_s = playback_start_wall_s + elapsed_bag_s / args.play_rate - time.monotonic()
            if sleep_s > 0:
                time.sleep(sleep_s)

            if last_query_ts_ns is not None and (timestamp_ns - last_query_ts_ns) < min_query_interval_ns:
                continue
            last_query_ts_ns = timestamp_ns

            query_image = bridge.compressed_imgmsg_to_cv2(msg)
            patch_tokens = asyncio.run(dinov2_model.infer_patch_tokens(query_image))
            query_embedding = compute_vlad(patch_tokens, codebook)

            similarities = map_embeddings @ query_embedding
            best_idx = int(np.argmax(similarities))
            best_score = float(similarities[best_idx])
            match_timestamp = idx_to_timestamp[best_idx]
            _, _, _, rgb_loader, infra1_loader = db.get_depth_embedding_features_images(match_timestamp)
            match_image = rgb_loader()
            if match_image is None:
                match_image = infra1_loader()
                if match_image is not None and match_image.ndim == 2:
                    match_image = cv2.cvtColor(match_image, cv2.COLOR_GRAY2BGR)

            frame_index += 1
            state.publish({
                "status": "ok",
                "frame_index": frame_index,
                "progress_pct": 100.0 * (timestamp_ns - bag_start_ts_ns) / max(1, bag_end_ts_ns - bag_start_ts_ns),
                "query_timestamp_ns": timestamp_ns,
                "match_timestamp_ns": int(match_timestamp),
                "similarity": best_score,
                "is_confident": best_score >= args.similarity_threshold,
                "query_image_b64": encode_jpeg_b64(query_image),
                "match_image_b64": encode_jpeg_b64(match_image) if match_image is not None else None,
            })

        if not args.loop:
            return


HTML_PAGE = """<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>TinyNav Live Retrieval</title>
<style>
  body { font-family: -apple-system, system-ui, sans-serif; background: #111; color: #eee; margin: 0; padding: 16px; }
  h1 { font-size: 16px; font-weight: 600; margin: 0 0 12px; color: #ccc; }
  #status { font-size: 13px; color: #999; margin-bottom: 12px; }
  #status.live { color: #4caf50; }
  #status.dead { color: #e05a4e; }
  .row { display: flex; gap: 16px; flex-wrap: wrap; }
  .panel { flex: 1 1 380px; min-width: 300px; background: #1a1a1a; border-radius: 8px; padding: 12px; box-sizing: border-box; }
  .panel h2 { font-size: 13px; text-transform: uppercase; letter-spacing: 0.05em; color: #888; margin: 0 0 8px; }
  .panel img { width: 100%; height: auto; border-radius: 4px; background: #000; display: block; }
  .meta { font-size: 12px; color: #aaa; margin-top: 8px; display: flex; justify-content: space-between; }
  #score { font-size: 28px; font-weight: 700; text-align: center; margin: 12px 0; }
  #score.confident { color: #4caf50; }
  #score.weak { color: #e0a34e; }
  #bar-track { height: 4px; background: #333; border-radius: 2px; overflow: hidden; margin-top: 12px; }
  #bar-fill { height: 100%; background: #4caf50; width: 0%; transition: width 0.2s linear; }
</style>
</head>
<body>
  <h1>TinyNav Live Retrieval &mdash; top-1 VLAD match</h1>
  <div id="status">connecting&hellip;</div>
  <div id="score"></div>
  <div class="row">
    <div class="panel">
      <h2>Query (live)</h2>
      <img id="query-img">
      <div class="meta"><span id="query-ts"></span><span id="frame-idx"></span></div>
    </div>
    <div class="panel">
      <h2>Top-1 match (map keyframe)</h2>
      <img id="match-img">
      <div class="meta"><span id="match-ts"></span></div>
    </div>
  </div>
  <div id="bar-track"><div id="bar-fill"></div></div>

<script>
function fmtNs(ns) {
  if (!ns) return "";
  return (ns / 1e9).toFixed(3) + " s";
}

function connect() {
  const statusEl = document.getElementById("status");
  const ws = new WebSocket(`ws://${location.host}/ws/live`);

  ws.onopen = () => { statusEl.textContent = "live"; statusEl.className = "live"; };
  ws.onclose = () => {
    statusEl.textContent = "disconnected, retrying…";
    statusEl.className = "dead";
    setTimeout(connect, 1000);
  };
  ws.onerror = () => ws.close();

  ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    if (data.status === "error") {
      statusEl.textContent = "error: " + data.message;
      statusEl.className = "dead";
      return;
    }
    document.getElementById("query-img").src = "data:image/jpeg;base64," + data.query_image_b64;
    if (data.match_image_b64) {
      document.getElementById("match-img").src = "data:image/jpeg;base64," + data.match_image_b64;
    }
    document.getElementById("query-ts").textContent = fmtNs(data.query_timestamp_ns);
    document.getElementById("match-ts").textContent = fmtNs(data.match_timestamp_ns);
    document.getElementById("frame-idx").textContent = "frame " + data.frame_index;
    document.getElementById("bar-fill").style.width = data.progress_pct + "%";

    const scoreEl = document.getElementById("score");
    scoreEl.textContent = "similarity " + data.similarity.toFixed(3);
    scoreEl.className = data.is_confident ? "confident" : "weak";
  };
}
connect();
</script>
</body>
</html>
"""


def build_app(state: RetrievalState) -> FastAPI:
    app = FastAPI()

    @app.get("/", response_class=HTMLResponse)
    def index():
        return HTML_PAGE

    @app.websocket("/ws/live")
    async def ws_live(ws: WebSocket):
        await ws.accept()
        last_seq_sent = -1
        try:
            while True:
                seq, payload = state.snapshot()
                if seq != last_seq_sent:
                    last_seq_sent = seq
                    await ws.send_json(payload)
                await asyncio.sleep(0.03)
        except WebSocketDisconnect:
            pass

    return app


def main() -> None:
    args = parse_args()
    state = RetrievalState()
    threading.Thread(target=retrieval_loop, args=(args, state), daemon=True).start()
    uvicorn.run(build_app(state), host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
