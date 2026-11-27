"""Read-only probe of Go2 motion ownership and SDK low-command support."""

import argparse
import json
import threading
import time


def inspect_sdk():
    from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowCmd_
    from unitree_sdk2py.utils.crc import CRC

    packet = unitree_go_msg_dds__LowCmd_()
    try:
        CRC().Crc(packet)
    except (ImportError, OSError, RuntimeError) as exc:
        return {"ok": False, "error": str(exc)}
    return {"ok": True, "motor_slots": len(packet.motor_cmd)}


def inspect_owner():
    try:
        from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient
    except ImportError as exc:
        return {"state": "unknown", "error": str(exc)}
    client = MotionSwitcherClient()
    client.SetTimeout(1.0)
    client.Init()
    status, mode = client.CheckMode()
    if status != 0 or not isinstance(mode, dict) or not isinstance(mode.get("name"), str):
        return {"state": "unknown", "rpc_status": status}
    return {"state": "occupied" if mode["name"] else "released", "mode": mode}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--iface", required=True)
    args = ap.parse_args()
    from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
    from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_

    ChannelFactoryInitialize(0, args.iface)
    lock = threading.Lock()
    distinct_ticks = 0
    last_tick = None
    last_received = None

    def on_state(msg):
        nonlocal distinct_ticks, last_tick, last_received
        with lock:
            if msg.tick != last_tick:
                distinct_ticks += 1
                last_tick = msg.tick
                last_received = time.monotonic()

    sub = ChannelSubscriber("rt/lowstate", LowState_)
    sub.Init(on_state, 10)
    crc = inspect_sdk()
    owner = inspect_owner()
    time.sleep(2.0)
    with lock:
        age = time.monotonic() - last_received if last_received is not None else None
        count = distinct_ticks
    print(json.dumps({
        "mode": "read_only",
        "iface": args.iface,
        "motion_owner": owner,
        "sdk_lowcmd_crc": crc,
        "lowstate_distinct_ticks": count,
        "lowstate_age_ms": age * 1000 if age is not None else None,
        "independent_physical_stop": "unverified",
        "motor_write_enabled": False,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
