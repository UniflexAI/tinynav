import json
import os
import time

import av
import numpy as np
import decord


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
        self._remux_pts_offset = 0
        self._encoded_via_write = False
        self._remuxed = False
        self._first_src_time_base = None

        if self.mode == "write":
            os.makedirs(self.dir_path, exist_ok=True)
            if os.path.exists(self.video_path):
                os.remove(self.video_path)
            if os.path.exists(self.meta_path):
                os.remove(self.meta_path)
            self._container = av.open(self.video_path, mode="w")
        else:
            self.ts_to_idx = self._load_meta()
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
        if self._remuxed:
            raise RuntimeError("VideoDB.write() cannot follow remux_from_dir() on the same instance")
        self._encoded_via_write = True
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

    def remux_from_dir(self, src_dir_path: str) -> int:
        """Append another VideoDB directory's entire video via packet-level stream
        copy (no decode/re-encode). Much cheaper than read()-ing every frame through
        decord and re-encoding it with write(), which is what makes merge_maps.py's
        keyframe fusion step slow/memory-heavy on constrained devices (e.g. Jetson
        Nano). Only valid when this VideoDB has never had write() called on it.

        Any frames present in the source video but absent from its own ts_to_idx
        (e.g. left over from a keyframe-pruning pass) are still copied byte-for-byte
        along with the rest of the stream, since a compressed inter-frame stream
        can't be selectively dropped without decoding; they simply remain unreferenced
        by any timestamp in the merged map.

        Returns the number of frames appended (0 if the source has no video).
        """
        if self.mode != "write":
            raise RuntimeError("remux_from_dir requires mode='write'")
        if self._encoded_via_write:
            raise RuntimeError("remux_from_dir() cannot follow write() on the same instance")
        src_video_path = os.path.join(src_dir_path, "video.mp4")
        if not os.path.exists(src_video_path):
            return 0
        src_meta_path = os.path.join(src_dir_path, "meta.json")
        with open(src_meta_path, "r", encoding="utf-8") as f:
            src_meta = json.load(f)
        src_ts_to_idx = {int(k): int(v) for k, v in src_meta.get("ts_to_idx", {}).items()}
        src_is_gray = bool(src_meta.get("is_gray", False))
        if self.is_gray is not None and self.is_gray != src_is_gray:
            raise ValueError(f"remux_from_dir: gray/color mismatch with {src_dir_path}")
        idx_to_ts = {v: k for k, v in src_ts_to_idx.items()}

        src_container = av.open(src_video_path)
        src_stream = src_container.streams.video[0]
        if self._stream is None:
            self._stream = self._container.add_stream_from_template(src_stream)
            self.is_gray = src_is_gray
            self._first_src_time_base = src_stream.time_base
        else:
            # Note: self._stream.time_base is the *output* container's muxer-assigned
            # time_base (only finalized after the first packet is muxed, and not
            # necessarily equal to any source's time_base even when compatible) — the
            # meaningful comparison is between the raw source streams themselves, since
            # packet.pts/dts below are offset in units of the *source's own* time_base
            # (each packet keeps its originating stream's time_base attached, which
            # av.mux() uses to rescale into the output's time_base automatically).
            dst_ctx, src_ctx = self._stream.codec_context, src_stream.codec_context
            if (
                (dst_ctx.width, dst_ctx.height, str(dst_ctx.pix_fmt)) != (src_ctx.width, src_ctx.height, str(src_ctx.pix_fmt))
                or self._first_src_time_base != src_stream.time_base
            ):
                src_container.close()
                raise ValueError(
                    f"remux_from_dir: video codec/resolution/time_base mismatch with {src_dir_path}; "
                    "cannot concatenate into a single stream without re-encoding."
                )

        pts_offset = self._remux_pts_offset
        max_end = 0
        n_frames = 0
        for packet in src_container.demux(src_stream):
            if packet.dts is None:
                continue  # flush/EOF marker packet
            packet.pts += pts_offset
            packet.dts += pts_offset
            packet.stream = self._stream
            self._container.mux(packet)
            ts = idx_to_ts.get(n_frames)
            if ts is not None:
                self.ts_to_idx[ts] = self.frame_count
            self.frame_count += 1
            n_frames += 1
            max_end = max(max_end, packet.dts + (packet.duration or 0))
        src_container.close()

        self._remux_pts_offset = pts_offset + max_end
        self._remuxed = True
        if n_frames and n_frames != len(src_ts_to_idx):
            print(
                f"  warning: remux_from_dir({src_dir_path}): {n_frames} video frames but "
                f"{len(src_ts_to_idx)} entries in its ts_to_idx; some frames are orphaned "
                "(harmless, but bloats the merged video)."
            )
        return n_frames

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
            if self._stream is not None and self._encoded_via_write:
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
