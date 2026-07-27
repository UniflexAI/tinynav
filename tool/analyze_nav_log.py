#!/usr/bin/env python3
"""Summarise a map_node run from its ROS log.

Answers, in order: did the executor stall, what did the planner spend its time
on, did anything fail, and did the route actually progress.

Usage
  python3 tool/analyze_nav_log.py                 # newest map_node log
  python3 tool/analyze_nav_log.py <path/to.log>
  python3 tool/analyze_nav_log.py --list          # show candidate logs
"""
from __future__ import annotations

import glob
import os
import re
import sys

STAMP = re.compile(r"^\[\w+\] \[(\d+\.\d+)\]")
KV = re.compile(r"(\w+)=(-?[\d.]+)")


def find_logs():
    """map_node logs, newest first. ROS names them python3_<pid>_<ms>.log."""
    out = []
    for path in glob.glob(os.path.expanduser("~/.ros/log/python3_*.log")):
        try:
            with open(path, errors="ignore") as f:
                head = f.read(20000)
        except OSError:
            continue
        if "[map_node]" in head:
            out.append((os.path.getmtime(path), path))
    return [p for _, p in sorted(out, reverse=True)]


def pct(values, q):
    if not values:
        return float("nan")
    s = sorted(values)
    return s[min(len(s) - 1, int(len(s) * q / 100.0))]


def describe(name, values, unit="", prec=0):
    if not values:
        print(f"  {name:<22} (none)")
        return
    print(f"  {name:<22} n={len(values):<5} 中位={pct(values,50):>8.{prec}f}{unit} "
          f"90%={pct(values,90):>8.{prec}f}{unit} 最大={max(values):>8.{prec}f}{unit}")


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if "--list" in sys.argv:
        for p in find_logs():
            print(p)
        return
    if args:
        path = args[0]
    else:
        logs = find_logs()
        if not logs:
            print("no map_node log found under ~/.ros/log/")
            return
        path = logs[0]
    print(f"log: {path}\n")

    lines = open(path, errors="ignore").readlines()
    t0 = t1 = None
    for line in lines:
        m = STAMP.match(line)
        if m:
            t = float(m.group(1))
            t0 = t if t0 is None else t0
            t1 = t
    if t0 is not None:
        print(f"时长: {t1 - t0:.0f}s   行数: {len(lines)}\n")

    # ---- executor stalls -------------------------------------------------
    gaps = {}
    for line in lines:
        if "cb_gap" in line:
            d = dict(KV.findall(line))
            nm = re.search(r"name=(\w+)", line)
            if nm and "gap_s" in d:
                gaps.setdefault(nm.group(1), []).append(float(d["gap_s"]))
    print("== 执行器阻塞 (cb_gap: 回调比周期晚多少) ==")
    if not gaps:
        print("  无 —— 没有回调被显著延迟 ✅")
    for name, g in sorted(gaps.items()):
        total = sum(x for x in g)
        print(f"  {name:<20} 次数={len(g):<5} 中位={pct(g,50):.2f}s "
              f"90%={pct(g,90):.2f}s 最大={max(g):.2f}s  累计滞后={total:.0f}s")
        if t0 is not None and t1 and t1 > t0:
            print(f"  {'':<20} -> 约占运行时长的 {100*total/(t1-t0):.0f}%")
    print()

    # ---- planner ---------------------------------------------------------
    ok = [ln for ln in lines if "nav_path_profile ok" in ln]
    bad = [ln for ln in lines if "nav_path_profile failed" in ln]
    fields = {}
    for line in ok:
        for k, v in KV.findall(line):
            fields.setdefault(k, []).append(float(v))
    print(f"== 全局规划 (成功 {len(ok)} / 失败 {len(bad)}) ==")
    for key, unit in [("total_ms", "ms"), ("sdf_search_ms", "ms"),
                      ("start_snap_ms", "ms"), ("shortcut_ms", "ms")]:
        describe(key, fields.get(key, []), unit)
    for key in ("expanded", "z_span", "sdf_path_len", "pruned_path_len"):
        describe(key, fields.get(key, []))
    if bad:
        reasons = {}
        for line in bad:
            r = re.search(r"failed=(\w+)", line)
            if r:
                reasons[r.group(1)] = reasons.get(r.group(1), 0) + 1
        print(f"  失败原因: {reasons}")
    print()

    # ---- is the search crawling the 3D volume? ---------------------------
    exp = fields.get("expanded", [])
    sdf = fields.get("sdf_search_ms", [])
    if exp and sdf and len(exp) == len(sdf):
        slow = [(e, s, z) for e, s, z in zip(exp, sdf, fields.get("z_span", [0] * len(exp))) if s > 1000]
        print(f"== 慢规划 (sdf_search > 1s): {len(slow)}/{len(sdf)} ==")
        if slow:
            print(f"  这些的 expanded 中位 = {pct([x[0] for x in slow],50):.0f} 节点, "
                  f"z_span 中位 = {pct([x[2] for x in slow],50):.0f} 层")
            fast = [(e, s, z) for e, s, z in zip(exp, sdf, fields.get("z_span", [0] * len(exp))) if s <= 1000]
            if fast:
                print(f"  对比快的  expanded 中位 = {pct([x[0] for x in fast],50):.0f} 节点, "
                      f"z_span 中位 = {pct([x[2] for x in fast],50):.0f} 层")
            print("  -> z_span 明显 >0 就说明 A* 在 3D 体积里爬,压成单层可直接消掉")
        print()

    # ---- route progress --------------------------------------------------
    print("== 路线进展 ==")
    for pat, label in [(r"Generated (\d+) nav subgoals", "子目标生成"),
                       (r"Advanced nav subgoal: (\d+/\d+)", "子目标推进"),
                       (r"All POIs have been visited", "导航完成"),
                       (r"Failed to regenerate global path", "重规划失败沿用旧路径"),
                       (r"RTK ground-z lookup built from (\d+)", "z查表(我们的新代码)"),
                       (r"Replanning global path: deviation", "偏离触发重规划(force_replan下应为0)")]:
        hits = [ln for ln in lines if re.search(pat, ln)]
        extra = ""
        if hits and "(" in pat:
            last = re.search(pat, hits[-1])
            if last and last.groups():
                extra = f"  最后={last.group(1)}"
        print(f"  {label:<32} {len(hits)}{extra}")
    print()

    # ---- RTK -------------------------------------------------------------
    rtk = [float(STAMP.match(ln).group(1)) for ln in lines
           if "Using /rtk/map_pose" in ln and STAMP.match(ln)]
    if len(rtk) > 1:
        d = [rtk[i + 1] - rtk[i] for i in range(len(rtk) - 1)]
        print("== RTK 定位 (该日志节流 5s,间隔应≈5.0s) ==")
        print(f"  条数={len(rtk)}  中位间隔={pct(d,50):.1f}s  最大={max(d):.1f}s")
        over = [x for x in d if x > 6.0]
        if over:
            print(f"  ⚠️ {len(over)}/{len(d)} 次间隔 >6s -> 这期间 map_node 被堵住,定位没更新")
    print()

    # ---- z ---------------------------------------------------------------
    zc = [dict(KV.findall(ln)) for ln in lines if "nav_z_clamp" in ln]
    if zc:
        diffs = [float(d["diff"]) for d in zc if "diff" in d]
        describe("nav_z_clamp diff", diffs, "m", prec=2)
        srcs = {}
        for ln in lines:
            m = re.search(r"nav_z_clamp source=(\w+)", ln)
            if m:
                srcs[m.group(1)] = srcs.get(m.group(1), 0) + 1
        print(f"  z 来源分布: {srcs}")


if __name__ == "__main__":
    main()
