import unittest
from unittest import mock

from typer.testing import CliRunner

from tinynav import cli


class TestTinyNavCli(unittest.TestCase):
    def setUp(self) -> None:
        self.runner = CliRunner()

    def test_build_docker_run_args_navigation(self) -> None:
        with mock.patch("tinynav.cli.detect_runtime_args", return_value=["--gpus", "all"]):
            args = cli.build_docker_run_args("run_navigation.sh")
        self.assertEqual(args[:4], ["docker", "run", "--rm", "-it"])
        self.assertIn("uniflexai/tinynav:latest", args)
        self.assertEqual(args[-2:], ["bash", "/tinynav/scripts/run_navigation.sh"])

    def test_init_runs_env_check_then_pull(self) -> None:
        order = []

        def record(command, *, cwd=None):
            order.append((list(command), cwd))

        with mock.patch("tinynav.cli.run_checked", side_effect=record):
            result = self.runner.invoke(cli.app, ["init"])

        self.assertEqual(result.exit_code, 0)
        self.assertEqual(order[0][0][0:2], ["bash", str(cli.SCRIPTS_DIR / "check_env.sh")])
        self.assertEqual(order[1][0], ["docker", "pull", cli.DEFAULT_IMAGE])
        self.assertIn("TinyNav is ready", result.stdout)

    def test_run_mapping_command_dispatches_script(self) -> None:
        with mock.patch("tinynav.cli.run_container_script") as run_container_script:
            result = self.runner.invoke(cli.app, ["run", "mapping"])

        self.assertEqual(result.exit_code, 0)
        run_container_script.assert_called_once_with("run_rosbag_build_map.sh")


if __name__ == "__main__":
    unittest.main()
