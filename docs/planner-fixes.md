# Planner maintenance fixes

Based on main fc17408 on branch xiaole/planner-fixes. Two-stage candidates and their UI were removed from this branch; their previous work is archived on spark under /tmp/tinynav-two-stage-backup.

- Grid remains 0.05 m per cell, with height increased from 10 to 40 cells (2 m total).
- WebSim validates and passes the complete robot configuration to both planner and simulator controller. Changing any robot field restarts the relevant children and resets simulation state. Planner confirms its loaded configuration through /planning/robot_config; the UI displays confirmed safety, dilation and maximum speed. /api/effective-robot exposes requested and confirmed configurations separately.
- Forward/reverse trajectories now include the measured pose at t=0 and end at t=3.0 s, instead of starting at t=0.1 s and ending at t=3.1 s. The existing 106 candidates, 0.1 s spacing and every-tenth-pose publication remain. Individual published poses carry their future timestamps. Controllers keep their existing 1 s interval interpretation.
- Ground navigation goal distance uses world XY; target height no longer changes goal distance or heading fade.
- Unobserved occupancy decays exponentially by monotonic elapsed time, rather than multiplying by 0.99 once per frame. TINYNAV_OCCUPANCY_DECAY_TAU_S sets the physical time constant (default 10 s, approximately the former decay at 10 Hz; half-life 6.93 s). Measurement evidence still accumulates per observation; this does not claim complete mapping frame-rate invariance.
- Velocity estimation initializes its timestamp correctly, ignores duplicate/backward intervals and measures horizontal travel.

Validation: seven targeted tests pass in tests/test_planner_fixes.py. Python and JavaScript syntax and git whitespace checks pass. API integration confirms custom safety=0.35 m, dilation=3 and max speed=0.6 m/s are loaded by the actual planner; negative dilation returns HTTP 400 without replacing current config, then defaults are restored.

GO2 L WebSim, one 45 s run per configuration in the current session: baseline with grid40 ends 4.336 m from goal, final fixes 4.316 m; neither arrives or reports collision. An initial fixed S run ends 3.610 m from goal without collision. Reverting only trajectory origin also remained stuck (L distance 4.459 m). The earlier session's baseline L distance 0.327 m was not reproduced. These fixes do not solve corner deadlock or establish improved navigation success. No GO2 hardware or Nano performance test was performed.

An unobstructed 2 m target run reaches the 0.30 m arrival threshold in 6.35 s, ending 0.271 m away without collision.
