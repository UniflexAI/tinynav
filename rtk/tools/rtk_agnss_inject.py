#!/usr/bin/env python3
"""Feed the UM982 AGNSS engine an aided position and UTC time at boot.

Each command is confirmed from the receiver echo that rtk_diag_logger.py writes
to rtk/log/rtk_YYYYMMDD_raw.log. `touch rtk/log/agnss_off` skips injection (A/B).
"""
import datetime
import os
import subprocess
import time

PORT = "/dev/serial/by-id/usb-1a86_USB_Serial-if00-port0"
LOGDIR = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "log"))
# Fixed site: the receiver only needs to be within 10 km. Same point as the
# NTRIP initial GGA in rtk/.ntrip.env; height is ellipsoidal, from a real fix.
AID_LAT, AID_LON, AID_HGT = "3947.002938", "11633.671097", "17.0"


def log(*a):
    print(time.strftime("%Y-%m-%d %H:%M:%S"), *a, flush=True)


def raw_path():
    return "%s/rtk_%s_raw.log" % (LOGDIR, time.strftime("%Y%m%d"))


def raw_size():
    try:
        return os.path.getsize(raw_path())
    except OSError:
        return None


def receiver_alive():
    try:
        with open(raw_path(), "rb") as f:
            f.seek(max(0, os.path.getsize(raw_path()) - 4000))
            tail = f.read().decode("latin1").splitlines()
        return any("GGA," in ln[:30] and time.time() - float(ln.split()[0]) < 3 for ln in tail[-60:])
    except (OSError, ValueError, IndexError):
        return False


def ntp_synced():
    r = subprocess.run(["timedatectl", "show", "-p", "NTPSynchronized", "--value"],
                       capture_output=True, text=True)
    return r.stdout.strip() == "yes"


def wait_for(cond, what, limit):
    # Monotonic: at boot the wall clock starts in 1970 and NTP jumps it ~56 years.
    t0 = time.monotonic()
    while not cond():
        if time.monotonic() - t0 > limit:
            log("gave up waiting for", what)
            return False
        time.sleep(2)
    return True


def send_confirmed(make_line, tag):
    """make_line runs right before each write so AIDTIME is fresh."""
    for attempt in range(1, 4):
        start = raw_size()
        line = make_line()
        fd = os.open(PORT, os.O_WRONLY | os.O_NOCTTY | os.O_NONBLOCK)
        try:
            os.write(fd, (line + "\r\n").encode())
        finally:
            os.close(fd)
        time.sleep(3)
        if start is None:
            log("sent (unconfirmed, no logger)", line)
            return None
        with open(raw_path(), "rb") as f:
            f.seek(start)
            new = f.read().decode("latin1")
        echo = [ln for ln in new.splitlines() if "$command,$" + tag in ln]
        if echo and "response: OK" in echo[-1]:
            log("OK", line)
            return True
        log("attempt %d no OK for %s: %s" % (attempt, tag, echo[-1][:120] if echo else "no echo"))
        # Back-to-back commands drop the second one; give the parser room.
        time.sleep(2)
    return False


def aidtime():
    t = datetime.datetime.now(datetime.timezone.utc)
    # 18 = GPS-UTC leap seconds. No "*hh" checksum: the firmware silently ignores it.
    return "$AIDTIME,%d,%d,%d,%d,%d,%d,%d,18" % (
        t.year, t.month, t.day, t.hour, t.minute, t.second, t.microsecond // 1000)


def main():
    if os.path.exists(os.path.join(LOGDIR, "agnss_off")):
        log("agnss_off present, injection skipped")
        return
    wait_for(lambda: os.path.exists(PORT), "serial port", 600)
    # The bridge (and so the echo) comes up ~80 s after boot.
    if not wait_for(receiver_alive, "receiver output in raw log", 900):
        log("logger not running; sends will be unconfirmed")
    send_confirmed(lambda: "$AIDPOS,%s,N,%s,E,%s" % (AID_LAT, AID_LON, AID_HGT), "AIDPOS")
    time.sleep(2)
    # A wrong time misleads the satellite search more than no time at all.
    if wait_for(ntp_synced, "NTP sync", 600):
        send_confirmed(aidtime, "AIDTIME")
    else:
        log("AIDTIME skipped: clock never synced")


if __name__ == "__main__":
    main()
