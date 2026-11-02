"""Read-only check of a recorded TinyNav bag, and kernel USB events lined up with its gaps.

Detects whether the bag came from a Looper or a RealSense, then checks what map
building needs from that sensor: the topics, their stream gaps, the frames the
pipeline pairs, and colour frames close enough to be matched to keyframes.

Runs with Python's standard library, preferably on the recording host.
Journal access may require sudo. No camera or service is restarted.
"""
from __future__ import annotations

import argparse
import bisect
from collections import defaultdict
from datetime import datetime
import json
import math
import os
from pathlib import Path
import re
import sqlite3
import struct
import subprocess
import sys
from zoneinfo import ZoneInfo

CAM = "/camera/camera"
# What each sensor's map build reads (tool/looper_bridge_node.py or
# tinynav/core/perception_node.py, then tinynav/core/build_map_node.py).
# `pair` is what the source node synchronizes, first topic as the base;
# `pair_slop_ms` is its synchronizer's slop (0 = exact TimeSynchronizer).
MODES = {
    "looper": {
        "streams": [f"{CAM}/infra1/image_rect_raw", f"{CAM}/depth/image_rect_raw",
                    f"{CAM}/vio_image", f"{CAM}/color/image_rect_raw/compressed"],
        "info": [f"{CAM}/infra1/camera_info", f"{CAM}/color/camera_info"],
        "pair": [f"{CAM}/depth/image_rect_raw", f"{CAM}/infra1/image_rect_raw",
                 f"{CAM}/vio_image"],
        "pair_slop_ms": 0,
        "color": f"{CAM}/color/image_rect_raw/compressed",
    },
    "realsense": {
        "streams": [f"{CAM}/infra1/image_rect_raw", f"{CAM}/infra2/image_rect_raw",
                    f"{CAM}/imu", f"{CAM}/color/image_raw"],
        "info": [f"{CAM}/infra2/camera_info", f"{CAM}/color/camera_info"],
        "pair": [f"{CAM}/infra1/image_rect_raw", f"{CAM}/infra2/image_rect_raw"],
        "pair_slop_ms": 20,
        "color": f"{CAM}/color/image_raw",
    },
}
# build_map_node.py matches a keyframe (an infra1 frame) to colour within 20 ms.
COLOR_SLOP_MS = 20
COLOR_BASE = f"{CAM}/infra1/image_rect_raw"
STREAMS = {t for m in MODES.values() for t in m["streams"]}
INTERFACE = "enx020000000a01"
PORT = re.compile(r"(?:usb |cdc_ncm )(\d+-[\d.]+)(?=[:\s])")


def header_stamp(data):
    """Header.stamp is first in plain CDR Image, CompressedImage, Imu and PoseStamped."""
    if len(data) < 12 or data[:2] not in (b"\x00\x00", b"\x00\x01"):
        return None
    sec, ns = struct.unpack("<iI" if data[1] else ">iI", data[4:12])
    return sec * 1_000_000_000 + ns if ns < 1_000_000_000 else None


def default_database():
    if os.environ.get("TINYNAV_DB_PATH"):
        return Path(os.environ["TINYNAV_DB_PATH"])
    checkout = Path(__file__).resolve().parents[1] / "tinynav_db"
    return checkout if checkout.is_dir() else Path("/tinynav/tinynav_db")


def resolve_bag(value, database):
    path = Path(value).expanduser()
    if path.exists():
        return path.resolve()
    path = database / "rosbags" / value
    if not path.exists():
        raise ValueError(f"Bag not found: {value}")
    return path.resolve()


def detect_mode(counts):
    """Which sensor recorded the bag, judged from the topics only that sensor publishes."""
    has = lambda name: counts.get(name, 0) > 0
    if has(f"{CAM}/vio_image"):
        return "looper", "has /camera/camera/vio_image (Looper VIO)"
    if any(n.startswith("/insight/") and c for n, c in counts.items()):
        return "looper", "has /insight/* but no /camera/camera/vio_image (older Looper firmware)"
    if has(MODES["looper"]["color"]):
        return "looper", "has compressed colour but no VIO"
    if has(f"{CAM}/infra2/image_rect_raw") and has(f"{CAM}/imu"):
        return "realsense", "has stereo infrared and IMU, no Looper VIO"
    return "unknown", "neither Looper VIO nor RealSense stereo + IMU recorded"


def matched(base, others, slop_ns):
    """How many base stamps have a stamp within slop in every other sorted list."""
    def near(values, t):
        i = bisect.bisect_left(values, t - slop_ns)
        return i < len(values) and values[i] <= t + slop_ns
    return sum(all(near(o, t) for o in others) for t in base)


def inspect_bag(path, gap_ms=250.0, dropout_seconds=1.0, mode=None):
    directory = path if path.is_dir() else path.parent
    files = sorted(path.glob("*.db3")) if path.is_dir() else [path]
    if not files or any(f.suffix != ".db3" for f in files):
        raise ValueError("Expected a SQLite .db3 file or rosbag directory")
    topics = defaultdict(lambda: {"count": 0, "empty": 0, "records": []})
    integrity, start, end = {}, None, None
    for file in files:
        # mode=ro prevents accidental creation of an empty DB for a bad path.
        with sqlite3.connect(file.resolve().as_uri() + "?mode=ro", uri=True) as db:
            integrity[file.name] = [r[0] for r in db.execute("PRAGMA quick_check")]
            if integrity[file.name] != ["ok"]:
                raise ValueError(f"SQLite integrity failed: {integrity[file.name]}")
            names = dict(db.execute("SELECT id, name FROM topics"))
            for name in names.values():
                topics[name]
            for tid, timestamp, size, head in db.execute(
                "SELECT topic_id, timestamp, length(data), substr(data,1,12) FROM messages"
            ):
                name = names[tid]
                topic = topics[name]
                topic["count"] += 1
                topic["empty"] += int(not size)
                start = timestamp if start is None else min(start, timestamp)
                end = timestamp if end is None else max(end, timestamp)
                if name in STREAMS:
                    topic["records"].append((timestamp, header_stamp(head or b"")))
    if start is None:
        raise ValueError("Bag contains no messages")
    counts = {n: t["count"] for n, t in topics.items()}
    detected, reason = detect_mode(counts)
    mode = mode or detected
    spec = MODES.get(mode)
    warnings, detail, stamps = [], {}, {}
    if spec is None:
        warnings.append(f"Unknown sensor: {reason}")
    elif mode != detected:
        warnings.append(f"Checked as {mode}, but the topics look like {detected}: {reason}")
    streams = spec["streams"] if spec else []
    for name, values in topics.items():
        records = sorted(values.pop("records"), key=lambda r: r[0])
        info = dict(values)
        if values["empty"]:
            warnings.append(f"{name}: {values['empty']} empty messages")
        if records and name in streams:
            valid = sorted(s for _, s in records if s is not None)
            unique = sorted(set(valid))
            stamps[name] = unique
            header_gaps = [(b - a) / 1e9 for a, b in zip(unique, unique[1:])]
            span = (records[-1][0] - records[0][0]) / 1e9
            gaps = [
                {"start_ns": a[0], "end_ns": b[0],
                 "offset_s": round((a[0] - start) / 1e9, 3),
                 "duration_s": round((b[0] - a[0]) / 1e9, 6)}
                for a, b in zip(records, records[1:])
                if b[0] - a[0] > gap_ms * 1e6
            ]
            for left, right, boundary in (
                (start, records[0][0], "leading"),
                (records[-1][0], end, "trailing"),
            ):
                if right - left > gap_ms * 1e6:
                    gaps.append({"start_ns": left, "end_ns": right,
                                 "offset_s": round((left-start)/1e9, 3),
                                 "duration_s": round((right-left)/1e9, 6),
                                 "boundary": boundary})
            gaps.sort(key=lambda g: g["start_ns"])
            in_order = [s for _, s in records if s is not None]
            info.update(
                span_s=round(span, 3),
                hz=round((len(records) - 1) / span, 3) if span else 0,
                invalid_headers=len(records) - len(valid),
                duplicates=len(valid) - len(unique),
                max_header_gap_s=max(header_gaps, default=0),
                header_backwards_over_1s=sum(
                    b < a - 1_000_000_000 for a, b in zip(in_order, in_order[1:])),
                leading_missing_s=round((records[0][0] - start) / 1e9, 3),
                trailing_missing_s=round((end - records[-1][0]) / 1e9, 3),
                gaps=gaps,
            )
            if info["invalid_headers"] or info["header_backwards_over_1s"]:
                warnings.append(f"{name}: malformed headers or camera timestamp reset")
            if (any(g["duration_s"] >= dropout_seconds for g in gaps)
                    or info["leading_missing_s"] >= dropout_seconds
                    or info["trailing_missing_s"] >= dropout_seconds):
                warnings.append(f"{name}: missing interval >= {dropout_seconds:g}s")
        detail[name] = info
    tf_count = {n: counts.get(n, 0) for n in ("/tf_static", "/tf")}
    report = {
        "bag": str(path), "sqlite_integrity": integrity,
        "metadata_present": (directory / "metadata.yaml").exists(),
        "bytes": sum(f.stat().st_size for f in files),
        "start_ns": start, "end_ns": end, "duration_s": round((end - start) / 1e9, 3),
        "mode": mode, "mode_detected": detected, "mode_reason": reason,
        "topics": detail, "streams": streams, "missing": [], "pairing": None,
        "color": None, "tf_count": tf_count, "warnings": warnings,
    }
    if spec is None:
        return report
    missing = [n for n in spec["streams"] + spec["info"] if not counts.get(n)]
    report["missing"] = missing
    if missing:
        warnings.append(f"{mode}: missing required topics: {missing}")
        for name in missing:
            detail.setdefault(name, {"count": 0, "empty": 0})
            if name in spec["streams"]:
                detail[name]["gaps"] = [{
                    "start_ns": start, "end_ns": end, "offset_s": 0,
                    "duration_s": round((end-start)/1e9, 6), "boundary": "whole_bag",
                }]
    base, *others = [stamps.get(n, []) for n in spec["pair"]]
    paired = matched(base, others, spec["pair_slop_ms"] * 1_000_000)
    report["pairing"] = {
        "topics": spec["pair"], "slop_ms": spec["pair_slop_ms"], "paired": paired,
        "base_count": len(base), "pct": round(100 * paired / len(base), 3) if base else 0,
    }
    if base and paired / len(base) < .99:
        warnings.append(f"{mode}: synchronized pairing below 99%")
    color, keyframes = stamps.get(spec["color"], []), stamps.get(COLOR_BASE, [])
    hits = matched(keyframes, [color], COLOR_SLOP_MS * 1_000_000)
    report["color"] = {
        "topic": spec["color"], "count": counts.get(spec["color"], 0),
        "hz": detail.get(spec["color"], {}).get("hz", 0), "slop_ms": COLOR_SLOP_MS,
        "matched": hits, "base_count": len(keyframes),
        "pct": round(100 * hits / len(keyframes), 3) if keyframes else 0,
    }
    if keyframes and color and hits / len(keyframes) < .99:
        warnings.append(f"colour within {COLOR_SLOP_MS} ms of infra1 below 99%")
    if not any(tf_count.values()):
        warnings.append("/tf_static and /tf: no recorded messages")
    return report


def read_kernel(start_ns, end_ns, log_file=None, journal_dir=None):
    if log_file:
        return {"status": "supplied", "source": str(log_file),
                "text": log_file.read_text(), "note": ""}
    command = [
        "journalctl", "--quiet", "--no-pager", "_TRANSPORT=kernel",
        "--since", f"@{start_ns // 1_000_000_000 - 2}",
        "--until", f"@{math.ceil(end_ns / 1e9) + 2}", "-o", "json",
    ]
    if journal_dir:
        command[1:1] = ["--directory", str(journal_dir)]
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"status": "unavailable", "source": "journalctl",
                "text": "", "note": str(exc)}
    status = ("unavailable" if result.returncode else
              "partial" if result.stderr.strip() else
              "available" if result.stdout.strip() else "no_records")
    return {"status": status, "source": "journalctl", "text": result.stdout,
            "note": result.stderr.strip()}


def usb_kind(message):
    lower = message.lower()
    if "usb disconnect" in lower or "unregister 'cdc_ncm'" in lower:
        return "disconnect"
    if re.search(r"reset (?:super|high|full|low)", lower):
        return "reset"
    if any(s in lower for s in ("failed", "timed out", "not accepting address",
                                "not responding", "netdev watchdog", "error -")):
        return "error"
    if any(s in lower for s in ("new superspeed", "register 'cdc_ncm'",
                                "link becomes ready", "authorized to connect")):
        return "connect"
    return None


def correlate_usb(report, kernel, port=None, match_window_s=2.0):
    """Kernel USB events on the camera's port, and the stream gaps they fall near."""
    device = Path("/sys/class/net") / INTERFACE / "device"
    if port is None and device.exists():
        port = device.resolve().name.split(":")[0]
    entries, invalid = [], 0
    for line in kernel["text"].splitlines():
        try:
            item = json.loads(line)
            ns = int(item["__REALTIME_TIMESTAMP"]) * 1000
            message = item["MESSAGE"]
            if not isinstance(message, str):
                raise ValueError("Non-text kernel message")
        except (ValueError, KeyError, TypeError):
            invalid += 1
            continue
        if report["start_ns"] - 2e9 <= ns <= report["end_ns"] + 2e9:
            entries.append((ns, message))
            found = PORT.search(message)
            if found and INTERFACE in message:
                port = found.group(1)
    events = []
    for ns, message in sorted(entries):
        kind = usb_kind(message)
        if kind is None:
            continue
        found = PORT.search(message)
        event_port = found.group(1) if found else None
        if event_port is None:
            continue
        events.append({"time_ns": ns, "offset_s": round((ns-report["start_ns"])/1e9, 3),
                       "camera": INTERFACE in message or event_port == port,
                       "port": event_port, "kind": kind, "message": message})
    matches = []
    for topic in report["streams"]:
        for gap in report["topics"].get(topic, {}).get("gaps", []):
            evidence = [
                e for e in events if e["camera"]
                and e["kind"] in ("disconnect", "reset", "error")
                and gap["start_ns"] - match_window_s * 1e9 <= e["time_ns"]
                <= gap["end_ns"] + match_window_s * 1e9
            ]
            if evidence:
                matches.append({"topic": topic, "gap": gap, "usb_events": evidence})
    return {"status": kernel["status"], "source": kernel["source"],
            "note": kernel["note"], "invalid_log_lines": invalid,
            "port": port, "events": events, "gap_correlations": matches,
            "limitation": "Temporal association does not establish cable, power or firmware root cause. "
                          "No returned records does not prove no USB incident occurred."}


def show(report, timezone):
    zone = ZoneInfo(timezone)
    fmt = lambda ns: datetime.fromtimestamp(ns/1e9, zone).isoformat(timespec="milliseconds")
    print(f"Bag: {report['bag']}")
    print(f"SQLite: OK | {report['bytes']/1e9:.3f} GB | {report['duration_s']:.3f} seconds")
    print(f"Time: {fmt(report['start_ns'])} -> {fmt(report['end_ns'])}")
    print(f"Sensor: {report['mode']} ({report['mode_reason']})")
    for topic in report["streams"]:
        data = report["topics"].get(topic, {})
        print(f"  {topic}: {data.get('count', 0)} messages, {data.get('hz', 0):g} Hz")
        if "max_header_gap_s" in data:
            print(f"    max sorted header gap {data['max_header_gap_s']:.3f}s; "
                  f"duplicates {data['duplicates']}; invalid headers {data['invalid_headers']}")
        for gap in data.get("gaps", []):
            print(f"    gap +{gap['offset_s']:.3f}s, {gap['duration_s']:.3f}s "
                  f"({fmt(gap['start_ns'])} -> {fmt(gap['end_ns'])}) "
                  f"{gap.get('boundary', '')}")
    if report["missing"]:
        print(f"Missing for {report['mode']} map building: {', '.join(report['missing'])}")
    pairing = report["pairing"]
    if pairing:
        how = "exact stamps" if not pairing["slop_ms"] else f"within {pairing['slop_ms']} ms"
        print(f"Synchronized ({how}): {pairing['paired']}/{pairing['base_count']} "
              f"{pairing['topics'][0]} paired ({pairing['pct']:.3f}%)")
    color = report["color"]
    if color:
        print(f"Colour {color['topic']}: {color['count']} messages, {color['hz']:g} Hz; "
              f"{color['matched']}/{color['base_count']} infra1 frames have colour within "
              f"{color['slop_ms']} ms ({color['pct']:.3f}%)")
    print("/tf_static: {} messages, /tf: {} messages".format(*report["tf_count"].values()))
    print("Pairing is conditional on RECEIVED frames; it excludes entirely missing intervals.")
    usb = report["usb"]
    if usb["status"] != "skipped":
        print(f"USB log: {usb['status']} ({usb['source']}) {usb['note']}")
        if usb["invalid_log_lines"]:
            print(f"USB log: {usb['invalid_log_lines']} unparseable lines; expected journal JSON lines.")
        for event in usb["events"]:
            print(f"  {fmt(event['time_ns'])} +{event['offset_s']:.3f}s "
                  f"{'camera' if event['camera'] else 'other'} port {event['port']} "
                  f"{event['kind']}: {event['message']}")
        for match in usb["gap_correlations"]:
            print(f"USB/GAP overlap: {match['topic']} "
                  f"+{match['gap']['offset_s']:.3f}s for {match['gap']['duration_s']:.3f}s")
        print(usb["limitation"])
    for warning in report["warnings"]:
        print("WARNING:", warning)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bag", help="Bag name, directory or SQLite .db3 path")
    parser.add_argument("--db-path", type=Path, default=default_database())
    parser.add_argument("--mode", choices=sorted(MODES), help="Sensor to check against (default: detect)")
    parser.add_argument("--gap-ms", type=float, default=250)
    parser.add_argument("--dropout-seconds", type=float, default=1)
    parser.add_argument("--timezone", default="Asia/Shanghai")
    parser.add_argument("--kernel-log", type=Path, help="Offline journalctl -o json export")
    parser.add_argument("--journal-dir", type=Path, help="Host journal directory (read-only mount)")
    parser.add_argument("--skip-usb", action="store_true")
    parser.add_argument("--usb-port", help="Camera USB port, e.g. 2-1.2 (default: from the camera NIC)")
    parser.add_argument("--json", type=Path, dest="json_path", help="Write full JSON report")
    args = parser.parse_args(argv)
    try:
        if (not math.isfinite(args.gap_ms) or not math.isfinite(args.dropout_seconds)
                or args.gap_ms <= 0 or args.dropout_seconds < args.gap_ms / 1000):
            raise ValueError("Finite positive thresholds required; dropout must be >= gap-ms/1000")
        report = inspect_bag(resolve_bag(args.bag, args.db_path),
                             args.gap_ms, args.dropout_seconds, args.mode)
        kernel = ({"status": "skipped", "source": "none", "text": "", "note": ""}
                  if args.skip_usb else read_kernel(
                      report["start_ns"], report["end_ns"], args.kernel_log, args.journal_dir))
        report["usb"] = correlate_usb(report, kernel, args.usb_port)
        show(report, args.timezone)
        if args.json_path:
            # A report must never overwrite a bag, manifest or supplied kernel log.
            target = args.json_path.resolve()
            bag = Path(report["bag"])
            root = bag if bag.is_dir() else bag.parent
            if (target.exists() or target == args.kernel_log
                    or target == root or root in target.parents):
                raise ValueError("JSON output must be a new file outside the bag directory")
            with target.open("x") as output:
                output.write(json.dumps(report, indent=2) + "\n")
        return 1 if report["warnings"] else 0
    except (OSError, ValueError, KeyError, sqlite3.Error) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
