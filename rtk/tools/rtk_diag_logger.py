#!/usr/bin/env python3
"""Record every UM982 line from the bridge's raw pty, plus a 1 Hz summary CSV.

Reads stdin (see rtk_diag_logger.sh). Output: rtk/log/rtk_YYYYMMDD.csv and
rtk/log/rtk_YYYYMMDD_raw.log.
"""
import os
import sys
import time

LOGDIR = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "log"))
HDR = ("wall,quality,sats,hdop,diff_age,calc,ion,postype,adr,"
       "corr_gps,corr_bds,corr_glo,corr_gal,corr_qzs,"
       "tracked,max_cn0,el_lt15,el_lt20,agnss,lat,lon")

st = dict.fromkeys(("quality sats hdop diff_age lat lon calc ion postype adr "
                    "cg cb cr ce cq agnss").split(), "")
gsv = {}
last = 0
day = out = raw = None


def open_day(now):
    global day, out, raw
    d = time.strftime("%Y%m%d", time.localtime(now))
    if d == day:
        return
    day = d
    os.makedirs(LOGDIR, exist_ok=True)
    out = open("%s/rtk_%s.csv" % (LOGDIR, d), "a", buffering=1)
    raw = open("%s/rtk_%s_raw.log" % (LOGDIR, d), "a", buffering=1)
    if out.tell() == 0:
        out.write(HDR + "\n")


def nbits(h):
    try:
        return bin(int(h, 16)).count("1")
    except ValueError:
        return None


def nsum(*hs):
    v = [nbits(h) for h in hs]
    return "" if None in v else sum(v)


def flush(now):
    global gsv
    cn0 = [c for c, e in gsv.values() if c is not None]
    els = [e for c, e in gsv.values() if e is not None]
    out.write(",".join(str(x) for x in (
        time.strftime("%H:%M:%S", time.localtime(now)),
        st["quality"], st["sats"], st["hdop"], st["diff_age"],
        st["calc"], st["ion"], st["postype"], st["adr"],
        st["cg"], st["cb"], st["cr"], st["ce"], st["cq"],
        len(gsv), max(cn0) if cn0 else "",
        sum(1 for e in els if e < 15), sum(1 for e in els if e < 20),
        st["agnss"], st["lat"], st["lon"])) + "\n")
    gsv = {}


for line in sys.stdin:
    line = line.strip()
    now = time.time()
    open_day(now)
    if line:
        raw.write("%.2f %s\n" % (now, line))
    # Talker follows the constellations in the fix: $GBGGA when only BeiDou is used.
    if line.startswith("$G") and line[3:7] == "GGA,":
        f = line.split("*")[0].split(",")
        if len(f) > 13:
            st["lat"], st["lon"] = f[2], f[4]
            st["quality"], st["sats"], st["hdop"], st["diff_age"] = f[6], f[7], f[8], f[13]
    elif line.startswith("#RTKSTATUSA"):
        b = line.split(";")
        f = b[1].split(",") if len(b) > 1 else []
        if len(f) > 15:
            # 15 = ADR Number (carrier observations); only manual R1.6+ documents it.
            st["postype"], st["calc"], st["ion"], st["adr"] = f[11], f[12], f[13], f[15]
            st["cg"], st["cb"], st["cr"] = nsum(f[0]), nsum(f[2], f[3]), nsum(f[5])
            st["ce"], st["cq"] = nsum(f[7], f[8]), nsum(f[9])
    elif line.startswith("#AGNSSSTATUSA"):
        b = line.split(";")
        f = b[1].split(",") if len(b) > 1 else []
        if len(f) > 5:
            st["agnss"] = f[5]
    elif "GSV," in line[:7] and line.startswith("$"):
        f = line.split("*")[0].split(",")
        t = line[1:3]
        # The len-3 bound keeps the trailing NMEA 4.11 signal id out of the satellites.
        for i in range(4, len(f) - 3, 4):
            if not f[i]:
                continue
            try:
                el = int(f[i + 1]) if f[i + 1] else None
                cn = int(f[i + 3]) if f[i + 3] else None
            except ValueError:
                continue
            gsv[(t, f[i])] = (cn, el)
    if now - last >= 1.0:
        last = now
        flush(now)
