"""Read-only SQLite camera bag checks and timestamp-aligned kernel USB evidence.

Runs with Python's standard library, preferably on the recording host.
Journal access may require sudo. No camera or service is restarted.
"""
from __future__ import annotations

import argparse
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

SUFFIXES = ("infra1/image_rect_raw", "depth/image_rect_raw", "vio_image")
INTERFACES = {"front": "enx020000000a01", "rear": "enx020000000b01"}
PORT = re.compile(r"(?:usb |cdc_ncm )(\d+-[\d.]+)(?=[:\s])")


def header_stamp(data):
    """Header.stamp is first in the rig's plain CDR Image and PoseStamped."""
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


def camera_prefixes(directory, topic_names):
    manifest = directory / "cameras.json"
    if manifest.exists():
        data = json.loads(manifest.read_text())
        cameras = [c for c in data["cameras"] if c["enabled"]]
        if not cameras:
            raise ValueError("Bag camera manifest has no enabled camera")
        return {c["id"]: c["prefix"] for c in cameras}
    prefixes = {"front": "/camera/camera"}
    if any(t.startswith("/camera/camera1/") for t in topic_names):
        prefixes["rear"] = "/camera/camera1"
    return prefixes


def inspect_bag(path, gap_ms=250.0, dropout_seconds=1.0):
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
                if any(name.endswith("/" + s) for s in SUFFIXES):
                    topic["records"].append((timestamp, header_stamp(head or b"")))
    if start is None:
        raise ValueError("Bag contains no messages")
    warnings, detail, stamp_sets = [], {}, {}
    for name, values in topics.items():
        records = sorted(values.pop("records"))
        info = dict(values)
        if values["empty"]:
            warnings.append(f"{name}: {values['empty']} empty messages")
        if records:
            stamps = [s for _, s in records if s is not None]
            unique = sorted(set(stamps))
            stamp_sets[name] = set(unique)
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
            info.update(
                span_s=round(span, 3),
                hz=round((len(records) - 1) / span, 3) if span else 0,
                invalid_headers=len(records) - len(stamps),
                duplicates=len(stamps) - len(unique),
                max_header_gap_s=max(header_gaps, default=0),
                header_backwards_over_1s=sum(
                    b < a - 1_000_000_000 for a, b in zip(stamps, stamps[1:])),
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
    cameras = {}
    for camera, prefix in camera_prefixes(directory, topics).items():
        names = [prefix + "/" + suffix for suffix in SUFFIXES]
        missing = [n for n in names if not detail.get(n, {}).get("count")]
        if missing:
            warnings.append(f"{camera}: missing required topics: {missing}")
            for name in missing:
                detail.setdefault(name, {"count": 0, "empty": 0})
                detail[name]["gaps"] = [{
                    "start_ns": start, "end_ns": end, "offset_s": 0,
                    "duration_s": round((end-start)/1e9, 6), "boundary": "whole_bag",
                }]
        sets = [stamp_sets.get(n, set()) for n in names]
        paired = len(set.intersection(*sets))
        depth = len(sets[1])
        coverage = paired / depth if depth else 0
        cameras[camera] = {
            "prefix": prefix, "topics": names, "paired": paired,
            "received_unique_depth": depth,
            "received_depth_pair_pct": round(100 * coverage, 3),
        }
        if depth and coverage < .99:
            warnings.append(f"{camera}: received-depth pairing below 99%")
    tf_count = detail.get("/tf_static", {}).get("count", 0)
    if not tf_count:
        warnings.append("/tf_static: no recorded messages")
    return {
        "bag": str(path), "sqlite_integrity": integrity,
        "metadata_present": (directory / "metadata.yaml").exists(),
        "bytes": sum(f.stat().st_size for f in files),
        "start_ns": start, "end_ns": end, "duration_s": round((end - start) / 1e9, 3),
        "topics": detail, "cameras": cameras, "tf_static_count": tf_count,
        "warnings": warnings,
    }


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


def correlate_usb(report, kernel, ports=None, match_window_s=2.0):
    ports = dict(ports or {})
    for camera, interface in INTERFACES.items():
        device = Path("/sys/class/net") / interface / "device"
        if camera not in ports and device.exists():
            ports[camera] = device.resolve().name.split(":")[0]
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
            port = PORT.search(message)
            for camera, interface in INTERFACES.items():
                if port and interface in message:
                    ports[camera] = port.group(1)
    events = []
    for ns, message in sorted(entries):
        kind = usb_kind(message)
        if kind is None:
            continue
        port_match = PORT.search(message)
        port = port_match.group(1) if port_match else None
        camera = next((c for c in report["cameras"] if
                       INTERFACES.get(c, "\0") in message
                       or (port is not None and ports.get(c) == port)), None)
        if camera is None and port is None:
            continue
        events.append({"time_ns": ns, "offset_s": round((ns-report["start_ns"])/1e9, 3),
                       "camera": camera, "port": port, "kind": kind, "message": message})
    matches = []
    for camera, info in report["cameras"].items():
        for topic in info["topics"]:
            for gap in report["topics"].get(topic, {}).get("gaps", []):
                evidence = [
                    e for e in events if e["camera"] == camera
                    and e["kind"] in ("disconnect", "reset", "error")
                    and gap["start_ns"] - match_window_s * 1e9 <= e["time_ns"]
                    <= gap["end_ns"] + match_window_s * 1e9
                ]
                if evidence:
                    matches.append({"camera": camera, "topic": topic, "gap": gap,
                                    "usb_events": evidence})
    return {"status": kernel["status"], "source": kernel["source"],
            "note": kernel["note"], "invalid_log_lines": invalid,
            "ports": ports, "events": events, "gap_correlations": matches,
            "limitation": "Temporal association does not establish cable, power or firmware root cause. "
                          "No returned records does not prove no USB incident occurred."}


def show(report, timezone):
    zone = ZoneInfo(timezone)
    fmt = lambda ns: datetime.fromtimestamp(ns/1e9, zone).isoformat(timespec="milliseconds")
    print(f"Bag: {report['bag']}")
    print(f"SQLite: OK | {report['bytes']/1e9:.3f} GB | {report['duration_s']:.3f} seconds")
    print(f"Time: {fmt(report['start_ns'])} -> {fmt(report['end_ns'])}")
    for camera, info in report["cameras"].items():
        print(f"{camera}: {info['paired']}/{info['received_unique_depth']} received depth "
              f"paired ({info['received_depth_pair_pct']:.3f}%)")
        for topic in info["topics"]:
            data = report["topics"].get(topic, {})
            print(f"  {topic}: {data.get('count', 0)} messages, {data.get('hz', 0):g} Hz")
            if "max_header_gap_s" in data:
                print(f"    max sorted header gap {data['max_header_gap_s']:.3f}s; "
                      f"duplicates {data['duplicates']}; invalid headers {data['invalid_headers']}")
            for gap in data.get("gaps", []):
                print(f"    gap +{gap['offset_s']:.3f}s, {gap['duration_s']:.3f}s "
                      f"({fmt(gap['start_ns'])} -> {fmt(gap['end_ns'])}) "
                      f"{gap.get('boundary', '')}")
    print(f"/tf_static: {report['tf_static_count']} messages")
    print("Pairing is conditional on RECEIVED depth; it excludes entirely missing intervals.")
    usb = report["usb"]
    if usb["status"] != "skipped":
        print(f"USB log: {usb['status']} ({usb['source']}) {usb['note']}")
        if usb["invalid_log_lines"]:
            print(f"USB log: {usb['invalid_log_lines']} unparseable lines; expected journal JSON lines.")
        for event in usb["events"]:
            print(f"  {fmt(event['time_ns'])} +{event['offset_s']:.3f}s "
                  f"{event['camera'] or 'unassigned'} {event['kind']}: {event['message']}")
        for match in usb["gap_correlations"]:
            print(f"USB/GAP overlap: {match['camera']} {match['topic']} "
                  f"+{match['gap']['offset_s']:.3f}s for {match['gap']['duration_s']:.3f}s")
        print(usb["limitation"])
    for warning in report["warnings"]:
        print("WARNING:", warning)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bag", help="Bag name, directory or SQLite .db3 path")
    parser.add_argument("--db-path", type=Path, default=default_database())
    parser.add_argument("--gap-ms", type=float, default=250)
    parser.add_argument("--dropout-seconds", type=float, default=1)
    parser.add_argument("--timezone", default="Asia/Shanghai")
    parser.add_argument("--kernel-log", type=Path, help="Offline journalctl -o json export")
    parser.add_argument("--journal-dir", type=Path, help="Host journal directory (read-only mount)")
    parser.add_argument("--skip-usb", action="store_true")
    parser.add_argument("--usb-port", action="append", default=[], metavar="CAMERA=PORT")
    parser.add_argument("--json", type=Path, dest="json_path", help="Write full JSON report")
    args = parser.parse_args(argv)
    try:
        if (not math.isfinite(args.gap_ms) or not math.isfinite(args.dropout_seconds)
                or args.gap_ms <= 0 or args.dropout_seconds < args.gap_ms / 1000):
            raise ValueError("Finite positive thresholds required; dropout must be >= gap-ms/1000")
        ports = dict(item.split("=", 1) for item in args.usb_port)
        report = inspect_bag(resolve_bag(args.bag, args.db_path),
                             args.gap_ms, args.dropout_seconds)
        kernel = ({"status": "skipped", "source": "none", "text": "", "note": ""}
                  if args.skip_usb else read_kernel(
                      report["start_ns"], report["end_ns"], args.kernel_log, args.journal_dir))
        report["usb"] = correlate_usb(report, kernel, ports)
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
