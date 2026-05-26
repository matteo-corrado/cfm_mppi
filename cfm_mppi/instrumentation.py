"""Measurement-only instrumentation for CFM-MPPI evaluation.

Adds section timers (CFM/MPPI/SFM), per-scenario captures, and dump
artifacts without altering the numerical algorithm. CUDA timing uses
torch.cuda.Event (async, no sync inside the t-loop).
"""

from __future__ import annotations

import datetime
import os
import subprocess
import time
from typing import Optional

import numpy as np


class InstrumentationRecorder:
    SECTION_NAMES = ("cfm", "mppi", "sfm")

    def __init__(self, n_scenarios: int, horizon: int, use_cuda: bool = True):
        self.n_scenarios = n_scenarios
        self.horizon = horizon
        self.use_cuda = use_cuda
        self.section_times_ms = {
            name: np.full((n_scenarios, horizon), np.nan, dtype=np.float32)
            for name in self.SECTION_NAMES
        }
        self._cpu_starts: dict[tuple[str, int, int], float] = {}
        self._pending_cuda: list[tuple[str, int, int, object, Optional[object]]] = []
        self._current_idx: int = -1
        self._current_t: int = -1
        self._scenario_start: Optional[float] = None
        self.per_scenario_rows: list[dict] = []
        self._state_trajs: list[object] = []
        self._control_trajs: list[object] = []
        self._obs_state_trajs: list[object] = []
        self._obs_control_trajs: list[object] = []
        self._goals: list[object] = []
        self._start_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()

    def begin_scenario(self, idx: int) -> None:
        self._current_idx = idx
        self._scenario_start = time.perf_counter()

    def begin_step(self, t: int) -> None:
        self._current_t = t

    def start_section(self, name: str) -> None:
        if name not in self.SECTION_NAMES:
            raise ValueError(f"unknown section: {name}")
        if self._current_idx < 0:
            raise RuntimeError(
                f"start_section({name}) called before begin_scenario; "
                f"_current_idx={self._current_idx}"
            )
        key = (name, self._current_idx, self._current_t)
        if name == "sfm" or not self.use_cuda:
            self._cpu_starts[key] = time.perf_counter()
        else:
            import torch

            start_ev = torch.cuda.Event(enable_timing=True)
            start_ev.record()
            self._pending_cuda.append(
                (name, self._current_idx, self._current_t, start_ev, None)
            )

    def end_section(self, name: str) -> None:
        if name not in self.SECTION_NAMES:
            raise ValueError(f"unknown section: {name}")
        if self._current_idx < 0:
            raise RuntimeError(f"end_section({name}) called before begin_scenario")
        key = (name, self._current_idx, self._current_t)
        if name == "sfm" or not self.use_cuda:
            start = self._cpu_starts.pop(key, None)
            if start is None:
                raise RuntimeError(
                    f"end_section({name}) without start_section at idx={self._current_idx} t={self._current_t}"
                )
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            self.section_times_ms[name][self._current_idx, self._current_t] = elapsed_ms
        else:
            import torch

            for i, (n, idx, t, start_ev, end_ev) in enumerate(self._pending_cuda):
                if (
                    n == name
                    and idx == self._current_idx
                    and t == self._current_t
                    and end_ev is None
                ):
                    new_end = torch.cuda.Event(enable_timing=True)
                    new_end.record()
                    self._pending_cuda[i] = (n, idx, t, start_ev, new_end)
                    return
            raise RuntimeError(f"end_section({name}) without matching start_section")

    def end_scenario(self, idx: int, scenario_metrics: dict) -> None:
        if self._scenario_start is None:
            raise RuntimeError(
                f"end_scenario({idx}) called without preceding begin_scenario"
            )
        if self.use_cuda and any(e[1] == idx for e in self._pending_cuda):
            import torch

            torch.cuda.synchronize()
        for n, sc_idx, t, start_ev, end_ev in self._pending_cuda:
            if sc_idx == idx and end_ev is not None:
                self.section_times_ms[n][sc_idx, t] = start_ev.elapsed_time(end_ev)
        self._pending_cuda = [e for e in self._pending_cuda if e[1] != idx]
        wall_s = time.perf_counter() - self._scenario_start
        row = {"idx": idx, "scenario_wall_s": wall_s, **scenario_metrics}
        self.per_scenario_rows.append(row)

    def capture(self, state_hist, control_hist, pos_obs, vel_obs, goal) -> None:
        self._state_trajs.append(state_hist.detach().cpu().clone())
        self._control_trajs.append(control_hist.detach().cpu().clone())
        self._obs_state_trajs.append(pos_obs.detach().cpu().clone())
        self._obs_control_trajs.append(vel_obs.detach().cpu().clone())
        self._goals.append(goal.detach().cpu().clone())

    def dump_cell(self, out_dir) -> None:
        import json
        from pathlib import Path

        import torch

        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        dump = out_dir / "dump"
        dump.mkdir(exist_ok=True)

        # State + control + goal are uniform-shape per scenario — stackable.
        if self._state_trajs:
            torch.save(torch.stack(self._state_trajs), dump / "state_traj.pt")
            torch.save(torch.stack(self._control_trajs), dump / "control_traj.pt")
            torch.save(torch.stack(self._goals), dump / "goal.pt")
        # Obs may have ragged n_obs across scenarios — save as list.
        torch.save(self._obs_state_trajs, dump / "obs_state_traj.pt")
        torch.save(self._obs_control_trajs, dump / "obs_control_traj.pt")

        np.savez(
            out_dir / "section_times.npz",
            cfm_ms=self.section_times_ms["cfm"],
            mppi_ms=self.section_times_ms["mppi"],
            sfm_ms=self.section_times_ms["sfm"],
        )
        with open(out_dir / "per_scenario.jsonl", "w") as f:
            for row in self.per_scenario_rows:
                f.write(json.dumps(row, default=_jsonable) + "\n")

    def write_hyperparams(self, out_dir, hparams: dict) -> None:
        import json
        from pathlib import Path

        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / "hyperparams.json", "w") as f:
            json.dump(hparams, f, indent=2, default=_jsonable)

    def write_env(self, out_dir, requested_precision: str) -> None:
        import json
        from pathlib import Path

        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        env = _capture_env(requested_precision, self._start_iso)
        with open(out_dir / "env.json", "w") as f:
            json.dump(env, f, indent=2, default=_jsonable)


def _jsonable(o):
    import torch

    if isinstance(o, torch.Tensor):
        return o.tolist()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    return str(o)


def _read_file(path: str) -> str:
    try:
        with open(path) as fh:
            return fh.read().strip()
    except Exception:
        return "N/A"


def _run(cmd: list) -> str:
    try:
        return subprocess.check_output(
            cmd, stderr=subprocess.DEVNULL, text=True, timeout=5
        ).strip()
    except Exception:
        return "N/A"


def _read_cpu_model() -> str:
    try:
        with open("/proc/cpuinfo") as fh:
            for line in fh:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except Exception:
        pass
    return "N/A"


def _read_live_mhz(n: int) -> list:
    out = []
    try:
        with open("/proc/cpuinfo") as fh:
            for line in fh:
                if line.startswith("cpu MHz"):
                    out.append(float(line.split(":", 1)[1].strip()))
                    if len(out) >= n:
                        break
    except Exception:
        pass
    return out


def _capture_env(requested_precision: str, start_iso: str) -> dict:
    import torch as _t

    try:
        import jax

        jax_version = jax.__version__
        jax_devices = [str(d) for d in jax.devices()]
    except Exception:
        jax_version = "N/A"
        jax_devices = []

    return {
        "torch_version": _t.__version__,
        "jax_version": jax_version,
        "cuda_version": _t.version.cuda or "N/A",
        "cudnn_version": _t.backends.cudnn.version()
        if _t.cuda.is_available()
        else None,
        "jax_devices": jax_devices,
        "requested_precision": requested_precision,
        "active_matmul_precision": _t.get_float32_matmul_precision(),
        "allow_tf32_matmul": _t.backends.cuda.matmul.allow_tf32,
        "allow_tf32_cudnn": _t.backends.cudnn.allow_tf32,
        "git_rev": _run(["git", "rev-parse", "HEAD"]),
        "hostname": _run(["hostname"]),
        "cpu_model": _read_cpu_model(),
        "cpu_governor": _read_file(
            "/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor"
        ),
        "cpu_live_mhz_first4": _read_live_mhz(4),
        "cpu_scaling_max_freq": _read_file(
            "/sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq"
        ),
        "numa_topology_summary": _run(["numactl", "--hardware"]).split("\n")[:5],
        "nvidia_smi_l": _run(["nvidia-smi", "-L"]),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID", "N/A"),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID", "N/A"),
        "start_iso": start_iso,
        "end_iso": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
