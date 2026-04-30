import json
import os
import time

import av
import numpy as np


class VideoDB:
    def __init__(
        self,
        dir_path: str,
        mode: str,
        fps: int = 30,
    ):
        if mode not in {"read", "write"}:
            raise ValueError(f"Invalid mode: {mode}")
        self.dir_path = dir_path
        self.video_path = os.path.join(dir_path, "video.mp4")
        self.meta_path = os.path.join(dir_path, "meta.json")
        self.mode = mode
        self.fps = fps
        self.ts_to_idx = {}
        self.frame_count = 0
        self._stream = None
        self._container = None
        self._video_reader = None
        self.is_gray = None
        self.write_count = 0
        self.write_total_s = 0.0
        self.read_count = 0
        self.read_total_s = 0.0

        if self.mode == "write":
            os.makedirs(self.dir_path, exist_ok=True)
            if os.path.exists(self.video_path):
                os.remove(self.video_path)
            if os.path.exists(self.meta_path):
                os.remove(self.meta_path)
            self._container = av.open(self.video_path, mode="w")
        else:
            self.ts_to_idx = self._load_meta()
            try:
                import decord
            except ImportError as e:
                raise ImportError(
                    "decord is required for VideoDB.read(). Install decord first."
                ) from e
            if os.path.exists(self.video_path):
                self._video_reader = decord.VideoReader(self.video_path)

    def _ensure_writer_stream(self, image: np.ndarray):
        h, w = image.shape[:2]
        stream = self._stream
        if stream is not None:
            return stream
        stream = self._container.add_stream("libx264", rate=self.fps)
        stream.width = int(w)
        stream.height = int(h)
        stream.pix_fmt = "yuv420p"
        stream.options = {"preset": "veryfast", "crf": "18", "tune": "zerolatency", "bf": "0"}
        self._stream = stream
        return stream

    def write(self, timestamp: int, image: np.ndarray):
        if self.mode != "write":
            raise RuntimeError("VideoDB write() requires mode='write'")
        t0 = time.perf_counter()
        frame_np = np.asarray(image)
        curr_is_gray = frame_np.ndim == 2
        if self.is_gray is None:
            self.is_gray = curr_is_gray
        elif self.is_gray != curr_is_gray:
            raise ValueError(
                f"Inconsistent image ndim for {self.dir_path}: expected "
                f"{'gray(2D)' if self.is_gray else 'color(3D)'}, got ndim={frame_np.ndim}"
            )
        frame_format = "gray" if self.is_gray else "bgr24"
        frame = av.VideoFrame.from_ndarray(frame_np, format=frame_format)
        stream = self._ensure_writer_stream(frame_np)
        frame.pts = self.frame_count
        self.ts_to_idx[int(timestamp)] = int(self.frame_count)
        self.frame_count += 1
        for pkt in stream.encode(frame):
            self._container.mux(pkt)
        self.write_count += 1
        self.write_total_s += time.perf_counter() - t0

    def _write_meta(self):
        payload = {
            "ts_to_idx": {str(k): int(v) for k, v in self.ts_to_idx.items()},
            "is_gray": bool(self.is_gray) if self.is_gray is not None else False,
        }
        with open(self.meta_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, separators=(",", ":"))

    def _load_meta(self) -> dict:
        if not os.path.exists(self.meta_path):
            return {}
        with open(self.meta_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict) or "ts_to_idx" not in data:
            return {}
        self.is_gray = bool(data.get("is_gray", False))
        ts_to_idx = data["ts_to_idx"]
        return {int(k): int(v) for k, v in ts_to_idx.items()}

    def _decode_frame_by_index(self, frame_idx: int):
        if frame_idx < 0 or not os.path.exists(self.video_path):
            return None
        if self._video_reader is None:
            return None
        if frame_idx >= len(self._video_reader):
            return None
        frame = self._video_reader[frame_idx].asnumpy()
        if self.is_gray:
            if frame.ndim == 3:
                frame = frame[..., 0]
            return frame
        # decord returns RGB; convert to BGR to keep OpenCV-style behavior.
        if frame.ndim == 3:
            return frame[..., ::-1]
        return frame

    def read(self, timestamp: int):
        if self.mode != "read":
            raise RuntimeError("VideoDB read() requires mode='read'")
        t0 = time.perf_counter()
        key = int(timestamp)
        if key not in self.ts_to_idx:
            self.read_count += 1
            self.read_total_s += time.perf_counter() - t0
            return None
        image = self._decode_frame_by_index(self.ts_to_idx[key])
        self.read_count += 1
        self.read_total_s += time.perf_counter() - t0
        return image

    def close(self):
        if self.mode == "write":
            if self._stream is not None:
                for pkt in self._stream.encode(None):
                    self._container.mux(pkt)
            if self._container is not None:
                self._container.close()
            self._write_meta()
        write_avg_ms = (self.write_total_s / self.write_count * 1000.0) if self.write_count > 0 else 0.0
        read_avg_ms = (self.read_total_s / self.read_count * 1000.0) if self.read_count > 0 else 0.0
        print(
            f"[VideoDB] dir={self.dir_path} mode={self.mode} "
            f"write_count={self.write_count} write_avg_ms={write_avg_ms:.3f} "
            f"read_count={self.read_count} read_avg_ms={read_avg_ms:.3f}"
        )
