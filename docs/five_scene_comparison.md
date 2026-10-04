# Five-scene selectable comparison

The same-screen selector now exposes five existing, geometrically distinct presets. Existing geometry remains unchanged, preserving earlier results:

| Scene | Challenge |
|---|---|
| dead_end | U-shaped trap; retreat away from goal before routing around walls |
| l_turn | Occluded corner and continued turn navigation |
| s_bend | Repeated direction changes and recovery from repeated stalls |
| narrow_gate | Clearance and conservative footprint collision checks |
| back_target | Goal behind the initial camera direction with front blocked |

Both rule-on and rule-off starts use identical configuration and chosen time limit. The UI default limit is now 120 s to accommodate the previously observed 88.4 s L-turn run. The results table supports up to ten configuration/time groups, labels scenes in Chinese, and filters invalid/nonterminal historical statuses. Recording view and downloading remain available.

Validation: browser state confirms all five options, actual-run scene labels, challenge descriptions, and per-scene paired results. Scripts parse. Live tests launched both modes for each of the three added presets with a 60 s limit, and verified equal configuration hashes and no assistance events in disabled runs.

Results are not uniformly complete: S-bend off/on were manually stopped after 7.3/7.6 s; narrow-gate off was manually stopped after 3.0 s, while on arrived after 14.6 s; back-target off/on both arrived after about 10.7 s, zero collisions. Interrupted runs are explicitly labeled and are not counted as navigation failures or evidence of improvement. Prior dead-end pair remains off timeout at 60 s versus on arrival at 48.3 s. Previous L-turn arrival used a 120 s limit and is not directly paired with a different-limit run.

These five presets test different behaviors; they do not establish that assistance improves all scenes. No planner or recovery policy changes are part of this UI expansion.
