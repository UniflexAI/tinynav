import json
import os
from pathlib import Path
import shlex
import subprocess
import tempfile
import unittest


class AppEnvironmentTests(unittest.TestCase):
    def test_backend_uses_current_configuration_over_stale_tmux_environment(self):
        repo = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tmux = root / "tmux"
            tmux.write_text("""#!/usr/bin/env python3
import json, os, sys
if sys.argv[1:] == ['show-environment', '-g']:
    print('TINYNAV_RL_ENGINE=stale.plan')
    print('TINYNAV_PLANNING_DIAGNOSTIC_CLOUDS=1')
    print('TINYNAV_PLANNING_LATEST_PAIR_ONLY=0')
else:
    with open(os.environ['CAPTURE'], 'w') as out:
        json.dump(sys.argv[1:], out)
""")
            tmux.chmod(0o755)
            for engine in (None, "/models/test policy.plan"):
                with self.subTest(engine=engine):
                    env = dict(os.environ, PATH=str(root) + os.pathsep + os.environ["PATH"],
                               CAPTURE=str(root / "args.json"), TINYNAV_PLANNER="ppo",
                               TINYNAV_PPO_FRONT_M=".30")
                    for key in ("TINYNAV_RL_ENGINE", "TINYNAV_PLANNING_DIAGNOSTIC_CLOUDS",
                                "TINYNAV_PLANNING_LATEST_PAIR_ONLY"):
                        env.pop(key, None)
                    if engine:
                        env["TINYNAV_RL_ENGINE"] = engine
                    subprocess.run(["bash", str(repo / "scripts/start_app.sh")], env=env, check=True)
                    args = json.loads((root / "args.json").read_text())
                    command = shlex.split(args[args.index("send-keys") + 1])
                    prefix = command[command.index("env"):command.index("uvicorn")]
                    stale = dict(env, TINYNAV_RL_ENGINE="stale.plan",
                                 TINYNAV_PLANNING_DIAGNOSTIC_CLOUDS="1",
                                 TINYNAV_PLANNING_LATEST_PAIR_ONLY="0")
                    result = subprocess.run([*prefix, "env"], env=stale, check=True,
                                            capture_output=True, text=True)
                    actual = dict(line.split("=", 1) for line in result.stdout.splitlines() if "=" in line)
                    self.assertEqual(actual["TINYNAV_PLANNER"], "ppo")
                    self.assertEqual(actual["TINYNAV_PPO_FRONT_M"], ".30")
                    self.assertNotIn("TINYNAV_PLANNING_DIAGNOSTIC_CLOUDS", actual)
                    self.assertNotIn("TINYNAV_PLANNING_LATEST_PAIR_ONLY", actual)
                    self.assertEqual(actual.get("TINYNAV_RL_ENGINE"), engine)


if __name__ == "__main__":
    unittest.main()
