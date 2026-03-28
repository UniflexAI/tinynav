from __future__ import annotations

import os
import platform
import shlex
import subprocess
from pathlib import Path
from typing import Sequence

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

DEFAULT_IMAGE = "uniflexai/tinynav:latest"
REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = REPO_ROOT / "scripts"
console = Console()
app = typer.Typer(
    name="tinynav",
    no_args_is_help=True,
    add_completion=False,
    help="TinyNav command line interface.",
    rich_markup_mode="rich",
)
run_app = typer.Typer(no_args_is_help=True, help="Run TinyNav workflows inside the default container.")
app.add_typer(run_app, name="run")


class CliError(RuntimeError):
    pass


def render_header(title: str, subtitle: str | None = None) -> None:
    console.print(
        Panel.fit(
            subtitle or "",
            title=f"[bold cyan]{title}[/bold cyan]",
            border_style="cyan",
        )
    )


def render_success(message: str) -> None:
    console.print(f"[bold green]✓[/bold green] {message}")


def render_warning(message: str) -> None:
    console.print(f"[bold yellow]![/bold yellow] {message}")


def render_error(message: str) -> None:
    console.print(f"[bold red]✗[/bold red] {message}")


def run_checked(command: Sequence[str], *, cwd: Path | None = None) -> None:
    console.print(f"[dim]$ {shlex.join(command)}[/dim]")
    subprocess.run(command, cwd=cwd, check=True)


def check_host_environment() -> None:
    script = SCRIPTS_DIR / "check_env.sh"
    if not script.exists():
        raise CliError(f"missing environment check script: {script}")
    run_checked(["bash", str(script)], cwd=REPO_ROOT)


def pull_default_image() -> None:
    run_checked(["docker", "pull", DEFAULT_IMAGE], cwd=REPO_ROOT)


def detect_runtime_args() -> list[str]:
    arch = platform.machine().lower()
    if arch in {"aarch64", "arm64"}:
        return ["--runtime", "nvidia"]
    return ["--gpus", "all"]


def build_docker_run_args(script_name: str) -> list[str]:
    script_path = Path("/tinynav/scripts") / script_name
    return [
        "docker",
        "run",
        "--rm",
        "-it",
        "--net=host",
        "--ipc=host",
        "--privileged",
        "-e",
        f"DISPLAY={os.environ.get('DISPLAY', ':0')}",
        "-e",
        "QT_X11_NO_MITSHM=1",
        "-v",
        f"{REPO_ROOT}:/tinynav",
        "-v",
        "/tmp/.X11-unix:/tmp/.X11-unix",
        "-v",
        "/etc/localtime:/etc/localtime:ro",
        *detect_runtime_args(),
        DEFAULT_IMAGE,
        "bash",
        str(script_path),
    ]


def run_container_script(script_name: str) -> None:
    script_on_host = SCRIPTS_DIR / script_name
    if not script_on_host.exists():
        raise CliError(f"missing script: {script_on_host}")
    run_checked(build_docker_run_args(script_name), cwd=REPO_ROOT)


def render_init_summary() -> None:
    table = Table(show_header=False, box=None, pad_edge=False)
    table.add_column(style="bold")
    table.add_column()
    table.add_row("Checks", "docker / docker daemon / GPU runtime")
    table.add_row("Image", DEFAULT_IMAGE)
    console.print(table)


@app.command()
def init() -> None:
    """Check host environment and pull the default TinyNav image."""
    render_header("tinynav init", "Check environment, then prepare the runtime image.")
    render_init_summary()

    try:
        console.print()
        console.rule("[bold]1. Checking environment[/bold]", style="cyan")
        check_host_environment()
        render_success("host environment check passed")

        console.print()
        console.rule("[bold]2. Pulling runtime image[/bold]", style="cyan")
        pull_default_image()
        render_success(f"image ready: {DEFAULT_IMAGE}")

        console.print()
        console.print(Panel.fit("[bold green]TinyNav is ready.[/bold green]", border_style="green"))
    except subprocess.CalledProcessError as exc:
        render_error(f"command failed with exit code {exc.returncode}")
        raise typer.Exit(exc.returncode) from exc
    except CliError as exc:
        render_error(str(exc))
        raise typer.Exit(1) from exc


@app.command()
def example() -> None:
    """Run the rosbag example in the default TinyNav container."""
    render_header("tinynav example", "Run the example workflow in Docker.")
    try:
        run_container_script("run_rosbag_examples.sh")
    except subprocess.CalledProcessError as exc:
        render_error(f"command failed with exit code {exc.returncode}")
        raise typer.Exit(exc.returncode) from exc
    except CliError as exc:
        render_error(str(exc))
        raise typer.Exit(1) from exc


@run_app.command("navigation")
def run_navigation() -> None:
    """Run online navigation."""
    render_header("tinynav run navigation")
    try:
        run_container_script("run_navigation.sh")
    except subprocess.CalledProcessError as exc:
        render_error(f"command failed with exit code {exc.returncode}")
        raise typer.Exit(exc.returncode) from exc
    except CliError as exc:
        render_error(str(exc))
        raise typer.Exit(1) from exc


@run_app.command("mapping")
def run_mapping() -> None:
    """Run map building from rosbag."""
    render_header("tinynav run mapping")
    try:
        run_container_script("run_rosbag_build_map.sh")
    except subprocess.CalledProcessError as exc:
        render_error(f"command failed with exit code {exc.returncode}")
        raise typer.Exit(exc.returncode) from exc
    except CliError as exc:
        render_error(str(exc))
        raise typer.Exit(1) from exc


def main(argv: Sequence[str] | None = None) -> int:
    app(args=list(argv) if argv is not None else None, standalone_mode=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
