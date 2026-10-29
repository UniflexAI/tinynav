"""Robot profiles: one declarative table for every supported Unitree chassis.

This module is deliberately import-pure: no ROS, no SDK import at module
level (client factories lazy-import on call), no environment reads. The
driver (unitree_control.py) and pilot's charge watch both look up the model
here instead of growing `is_quadruped`-style flags.

Facts are audit-backed (docs/multi-robot-sim-testbench.md §2, as2 audit
2026-10-08): AS2's lowstate is the hg layout with NO bms field, its battery
lives on rt/lf/bmsstate, and it has no robot_odom publisher at all -- the
odometry is republished from sportmodestate. The b2 probe (hangzhou rig,
2026-10-15) pins the b2 row: rt/lf/lowstate 19.1 Hz with a populated
bms_state, rt/lf/sportmodestate 19.9 Hz with real position/velocity but zero
stamps (the driver's all-zero filter is load-bearing), and likewise no
robot_odom publisher; b2w is still unprobed and keeps the go2 surface.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Callable, Literal, Protocol


class RobotType(str, Enum):
    # values are the existing ROBOT_TYPE strings -- .env / compose / the
    # {TYPE}_CONFIG lookup in robot_specs all keep working
    GO2 = "go2"
    GO2W = "go2w"
    B2 = "b2"
    B2W = "b2w"
    G1 = "g1"
    AS2 = "as2"


class Family(str, Enum):
    QUADRUPED = "quadruped"
    QUADRUPED_WHEELED = "quadruped_wheeled"
    HUMANOID = "humanoid"


class ChargingPolicy(Protocol):
    """How a model's BMS fields map to charge state / external power."""

    def state_from(self, status: int | None, median_ma: float | None) -> str: ...

    def externally_powered(self, state: str, fresh: bool, spans: bool,
                           median_ma: float | None) -> bool | None: ...


class StatusBitPolicy:
    """go family: bms_state.status is Unitree's enum; external power is a
    current median above the measured threshold (docs/pilot-dock.md §7)."""

    needs_median = False     # judges the status bit alone

    def __init__(self, threshold_ma: float) -> None:
        self._threshold_ma = threshold_ma

    def state_from(self, status, median_ma):
        return _STATES.get(status, UNKNOWN)

    def externally_powered(self, state, fresh, spans, median_ma):
        if not fresh:
            return None
        if state == CHARGING:
            return True
        return median_ma > self._threshold_ma if spans else None


class CurrentSignPolicy:
    """AS2: the hg BmsState has no status field -- the current sign/threshold
    IS the judgement. Threshold is a conservative default (-100 mA sits well
    above the measured -970 mA resting draw); calibrate on the pad before
    trusting dock decisions."""

    needs_median = True      # the current median IS the state

    def __init__(self, threshold_ma: float) -> None:
        self._threshold_ma = threshold_ma

    def state_from(self, status, median_ma):
        if median_ma is None:
            return UNKNOWN
        return CHARGING if median_ma > self._threshold_ma else DISCHARGING

    def externally_powered(self, state, fresh, spans, median_ma):
        if not fresh:
            return None
        return median_ma > self._threshold_ma if spans else None


# charge states (pilot/backend/charge_watch.py owns the canonical names)
CHARGING = "charging"
DISCHARGING = "discharging"
IDLE = "idle"
UNKNOWN = "unknown"

#: `bms_state.status`, Unitree's enum (unlisted -- alarms included -- UNKNOWN)
SAFE, WAKE_UP, PRECHG, CHG, DCHG, SELF_DCHG = 0, 1, 6, 7, 8, 9
_STATES = {
    PRECHG: CHARGING, CHG: CHARGING,
    DCHG: DISCHARGING, SELF_DCHG: DISCHARGING,
    SAFE: IDLE, WAKE_UP: IDLE,
}


@dataclass(frozen=True)
class BatterySpec:
    topic: str                       # where the BMS fields arrive
    idl: Literal["unitree_go", "unitree_hg"]
    has_status: bool                 # go family carries bms_state.status
    policy: ChargingPolicy


@dataclass(frozen=True)
class RobotProfile:
    type: RobotType
    family: Family
    motion_api: Literal["sport", "loco"]     # client family: sport (go2/b2/as2) | loco (g1)
    client_factory: Callable[[], object]     # returns an UNINITIALISED client
    lowstate_idl: Literal["unitree_go", "unitree_hg"]
    battery: BatterySpec | None              # g1: None -- no battery reporting
    odom: Literal["robot_odom", "sportmodestate"] | None
    gait_assertion: Literal["classic_walk"] | None
    needs_gait_worker: bool                  # reply RPC slower than the reader thread tolerates
    stand_steps: tuple[str, ...]
    sit_steps: tuple[str, ...]


def _sport_client(pkg: str) -> Callable[[], object]:
    def _make():
        mod = __import__(f"unitree_sdk2py.{pkg}.sport.sport_client",
                         fromlist=["SportClient"])
        return mod.SportClient()
    return _make


def _g1_client() -> object:
    from unitree_sdk2py.g1.loco.g1_loco_client import LocoClient
    return LocoClient()


#: AS2 resting discharge current, measured 2026-10-08 (audit). The go-family
#: threshold (-1000 mA) must NOT be reused: it sits inside the AS2 rest draw.
AS2_EXTERNAL_POWER_THRESHOLD_MA = -100.0
#: go family, measured on the rigs (docs/pilot-dock.md §7)
GO_EXTERNAL_POWER_THRESHOLD_MA = -1000.0


def _battery_go() -> BatterySpec:
    return BatterySpec("rt/lf/lowstate", "unitree_go", True,
                       StatusBitPolicy(GO_EXTERNAL_POWER_THRESHOLD_MA))


PROFILES: dict[RobotType, RobotProfile] = {
    RobotType.GO2: RobotProfile(
        RobotType.GO2, Family.QUADRUPED, "sport", _sport_client("go2"),
        "unitree_go", _battery_go(), "robot_odom",
        gait_assertion="classic_walk", needs_gait_worker=True,
        stand_steps=("StandUp", "BalanceStand"),
        sit_steps=("StandUp", "StandDown")),
    RobotType.GO2W: RobotProfile(
        RobotType.GO2W, Family.QUADRUPED_WHEELED, "sport",
        _sport_client("go2"), "unitree_go", _battery_go(), "robot_odom",
        gait_assertion=None, needs_gait_worker=False,
        stand_steps=("StandUp", "BalanceStand"),
        sit_steps=("StandUp", "StandDown")),
    RobotType.B2: RobotProfile(
        RobotType.B2, Family.QUADRUPED, "sport", _sport_client("b2"),
        "unitree_go", _battery_go(),
        # The real b2 never publishes rt/utlidar/robot_odom (2026-10-15 rig
        # probe: 10 s of silence against a standing dog); its chassis odometry
        # is republished from rt/lf/sportmodestate like the AS2's.
        "sportmodestate",
        gait_assertion="classic_walk", needs_gait_worker=True,
        # The walk gait is not a row: the driver appends it to every stand (a
        # SwitchGait(1) here is what a b2 stand got wrong -- 2444767).
        stand_steps=("StandUp", "BalanceStand"),
        sit_steps=("StandDown",)),
    RobotType.B2W: RobotProfile(
        RobotType.B2W, Family.QUADRUPED_WHEELED, "sport",
        _sport_client("b2"), "unitree_go", _battery_go(), "robot_odom",
        gait_assertion="classic_walk", needs_gait_worker=True,
        stand_steps=("StandUp", "BalanceStand"),
        sit_steps=("StandDown",)),
    RobotType.G1: RobotProfile(
        RobotType.G1, Family.HUMANOID, "loco", _g1_client,
        "unitree_hg", None, None,
        gait_assertion=None, needs_gait_worker=False,
        stand_steps=("Damp", "Squat2StandUp"),
        sit_steps=("StandUp2Squat",)),
    RobotType.AS2: RobotProfile(
        RobotType.AS2, Family.QUADRUPED, "sport", _sport_client("as2"),
        "unitree_hg",
        BatterySpec("rt/lf/bmsstate", "unitree_hg", False,
                    CurrentSignPolicy(AS2_EXTERNAL_POWER_THRESHOLD_MA)),
        "sportmodestate",
        # NO per-motion assertion: BalanceStand is a TOGGLE (audit) and the
        # driver re-arms at every motion start -- an unconditional call would
        # walk NORMAL_WALK back to STAND_UP and the next Move gets -1. The
        # switch lives only in stand_steps, where the pre-state is known.
        gait_assertion=None, needs_gait_worker=False,
        stand_steps=("StandUp", "BalanceStand"),
        sit_steps=("StandDown",)),
}


def get_profile(robot_type: str) -> RobotProfile | None:
    """None for anything this table doesn't know (e.g. lekiwi) -- callers
    that only make sense on a unitree chassis refuse softly."""
    try:
        return PROFILES[RobotType(robot_type.strip().lower())]
    except ValueError:
        return None


def require_profile(robot_type: str) -> RobotProfile:
    p = get_profile(robot_type)
    if p is None:
        raise ValueError(f"Unsupported unitree robot type: {robot_type!r}, "
                         f"expected one of {[t.value for t in RobotType]}")
    return p
