#!/usr/bin/env python3
"""Run main.py from a TOML config."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError as exc:  # pragma: no cover - cluster guard
    raise SystemExit("Python 3.11+ is required to read TOML configs.") from exc


ENV_ALIASES = {
    "PY_TIME_LIMIT": "time_limit",
    "TIME_LIMIT": "time_limit",
}


def parse_bool(value: str) -> bool:
    lowered = value.strip().lower()
    if lowered in {"1", "true", "yes", "y", "on"}:
        return True
    if lowered in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"cannot parse boolean value {value!r}")


def coerce_override(raw: str, current: Any) -> Any:
    if isinstance(current, bool):
        return parse_bool(raw)
    if isinstance(current, int) and not isinstance(current, bool):
        return int(raw)
    if isinstance(current, float):
        return float(raw)
    if isinstance(current, list):
        parts = [part for part in raw.replace(",", " ").split() if part]
        if not current:
            return parts
        return [coerce_override(part, current[0]) for part in parts]
    return raw


def cli_value(value: Any) -> list[str]:
    if isinstance(value, bool):
        return ["true" if value else "false"]
    if isinstance(value, list):
        result: list[str] = []
        for item in value:
            result.extend(cli_value(item))
        return result
    return [str(value)]


def format_value(value: Any, context: dict[str, str]) -> str:
    return str(value).format(**context)


def first_slurm_host() -> str:
    nodelist = os.environ.get("SLURM_JOB_NODELIST")
    if not nodelist:
        return "127.0.0.1"
    try:
        output = subprocess.check_output(
            ["scontrol", "show", "hostnames", nodelist],
            text=True,
        )
        return output.splitlines()[0].strip()
    except Exception:
        return "127.0.0.1"


def build_args(config_args: dict[str, Any], extra_args: list[str]) -> list[str]:
    args: list[str] = []
    for key, value in config_args.items():
        if value == "":
            continue
        flag = "--" + key.replace("_", "-")
        args.append(flag)
        args.extend(cli_value(value))
    args.extend(extra_args)
    return args


def validate_args(config_args: dict[str, Any]) -> None:
    alg = str(config_args.get("alg", "")).upper()
    n_envs = int(config_args.get("n_envs", 1))
    if n_envs < 1:
        raise SystemExit("n_envs must be >= 1")

    if alg in {"MAPPO", "LAGRMAPPO"}:
        n_steps = int(config_args.get("n_steps", 0))
        eval_freq = int(config_args.get("eval_freq", 0))
        if n_steps <= 0 or eval_freq <= 0:
            raise SystemExit("Invalid config: n_steps and eval_freq must be > 0.")
        if n_steps % n_envs != 0:
            raise SystemExit(
                f"Invalid config: n_steps={n_steps} must be divisible by n_envs={n_envs}."
            )
        if eval_freq % n_envs != 0:
            raise SystemExit(
                f"Invalid config: eval_freq={eval_freq} must be divisible by n_envs={n_envs}."
            )

    if alg == "QPLEX":
        train_freq = int(config_args.get("train_freq", 0))
        eval_freq = int(config_args.get("eval_freq", 0))
        if train_freq <= 0 or eval_freq <= 0:
            raise SystemExit("Invalid config: train_freq and eval_freq must be > 0.")
        if train_freq % n_envs != 0:
            raise SystemExit(
                f"Invalid config: train_freq={train_freq} must be divisible by n_envs={n_envs}."
            )
        if eval_freq % n_envs != 0:
            raise SystemExit(
                f"Invalid config: eval_freq={eval_freq} must be divisible by n_envs={n_envs}."
            )


def print_python_summary() -> None:
    print("========== Python environment summary ==========")
    print("Python:", sys.version.replace("\n", " "))
    print("Executable:", sys.executable)
    for name in ["torch", "numpy", "grid2op", "lightsim2grid", "wandb"]:
        try:
            mod = __import__(name)
            print(f"{name}: {getattr(mod, '__version__', 'unknown')}")
        except Exception as exc:  # pragma: no cover - diagnostic only
            print(f"{name}: import failed: {exc}")
    try:
        import torch

        print("CUDA available:", torch.cuda.is_available())
        print("Torch threads:", torch.get_num_threads())
    except Exception:
        pass
    print("===============================================")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", help="TOML run config")
    parser.add_argument("overrides", nargs=argparse.REMAINDER, help="extra main.py args")
    parser.add_argument("--dry-run", action="store_true", help="print command without running")
    ns = parser.parse_args()

    config_path = Path(ns.config).expanduser()
    if not config_path.is_absolute():
        config_path = Path.cwd() / config_path
    with config_path.open("rb") as file:
        config = tomllib.load(file)

    run = config.get("run", {})
    launcher = config.get("launcher", {})
    config_args = dict(config.get("args", {}))

    if run.get("seed_from_slurm_array", False) and "SEED" not in os.environ:
        array_task_id = os.environ.get("SLURM_ARRAY_TASK_ID")
        if array_task_id:
            config_args["seed"] = int(array_task_id)

    env_to_arg = {key.upper(): key for key in config_args}
    env_to_arg.update(ENV_ALIASES)
    for env_key, arg_key in env_to_arg.items():
        if env_key in os.environ and arg_key in config_args:
            config_args[arg_key] = coerce_override(os.environ[env_key], config_args[arg_key])

    job_id = os.environ.get("SLURM_JOB_ID", "local")
    context = {
        "config_name": config_path.stem,
        "job_id": job_id,
        "array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID", "0"),
        "project_dir": str(Path.cwd()),
    }
    run_dir = Path(format_value(run.get("run_dir", "outputs/{config_name}-{job_id}"), context))
    if not run_dir.is_absolute():
        run_dir = Path.cwd() / run_dir
    context["run_dir"] = str(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    (Path.cwd() / "checkpoint").mkdir(exist_ok=True)

    for key, value in config.get("environment", {}).items():
        os.environ[key] = format_value(value, context)

    for key in ["MPLCONFIGDIR", "WANDB_DIR", "XDG_CACHE_HOME"]:
        if key in os.environ:
            Path(os.environ[key]).mkdir(parents=True, exist_ok=True)

    if os.environ.get("SLURM_JOB_ID") and Path(f"/tmp/{job_id}").is_dir():
        os.environ.setdefault("TMPDIR", f"/tmp/{job_id}")

    extra_args = list(run.get("extra_args", [])) + list(ns.overrides)
    validate_args(config_args)
    main_args = build_args(config_args, extra_args)
    command = [sys.executable, "-u", "main.py", *main_args]

    if launcher.get("use_srun", False) and os.environ.get("SLURM_JOB_ID") and shutil.which("srun"):
        srun_command = ["srun"]
        if launcher.get("distributed", False):
            os.environ.setdefault("MASTER_ADDR", first_slurm_host())
            os.environ.setdefault("MASTER_PORT", str(launcher.get("master_port", 29500)))
            srun_command.extend(
                [
                    f"--nodes={launcher.get('nodes', os.environ.get('SLURM_JOB_NUM_NODES', '1'))}",
                    f"--ntasks={launcher.get('ntasks', os.environ.get('SLURM_NTASKS', '1'))}",
                    f"--ntasks-per-node={launcher.get('ntasks_per_node', 1)}",
                    f"--cpus-per-task={launcher.get('cpus_per_task', os.environ.get('SLURM_CPUS_PER_TASK', '72'))}",
                ]
            )
        else:
            srun_command.extend(
                [
                    "--ntasks=1",
                    f"--cpus-per-task={os.environ.get('SLURM_CPUS_PER_TASK', '72')}",
                ]
            )
        srun_command.append(f"--cpu-bind={launcher.get('cpu_bind', 'cores')}")
        command = [*srun_command, *command]

    print("========== JED config run ==========")
    print(f"Config: {config_path}")
    print(f"Run name: {run.get('name', config_path.stem)}")
    print(f"Run dir: {run_dir}")
    print(f"Job id: {job_id}")
    print(f"Array task id: {os.environ.get('SLURM_ARRAY_TASK_ID', 'none')}")
    print(f"Partition: {os.environ.get('SLURM_JOB_PARTITION', 'academic')}")
    print(f"QOS: {os.environ.get('SLURM_JOB_QOS', 'academic')}")
    print(f"Nodes: {os.environ.get('SLURM_JOB_NUM_NODES', '1')}")
    print(f"CPUs per task: {os.environ.get('SLURM_CPUS_PER_TASK', '72')}")
    print(f"Command: {' '.join(command)}")
    print("====================================")

    if launcher.get("print_python_summary", True):
        print_python_summary()

    if ns.dry_run:
        return 0
    return subprocess.run(command, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
