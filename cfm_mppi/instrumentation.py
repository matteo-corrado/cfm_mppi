"""Measurement-only instrumentation for CFM-MPPI evaluation.

Adds section timers (CFM/MPPI/SFM), per-scenario captures, and dump
artifacts without altering the numerical algorithm. CUDA timing uses
torch.cuda.Event (async, no sync inside the t-loop).
"""

from __future__ import annotations

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

    def begin_scenario(self, idx: int) -> None:
        self._current_idx = idx
        self._scenario_start = time.perf_counter()

    def begin_step(self, t: int) -> None:
        self._current_t = t

    def start_section(self, name: str) -> None:
        if name not in self.SECTION_NAMES:
            raise ValueError(f"unknown section: {name}")
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
        if self.use_cuda and any(e[1] == idx for e in self._pending_cuda):
            import torch

            torch.cuda.synchronize()
        for n, sc_idx, t, start_ev, end_ev in self._pending_cuda:
            if sc_idx == idx and end_ev is not None:
                self.section_times_ms[n][sc_idx, t] = start_ev.elapsed_time(end_ev)
        self._pending_cuda = [e for e in self._pending_cuda if e[1] != idx]
        wall_s = time.perf_counter() - (self._scenario_start or time.perf_counter())
        row = {"idx": idx, "scenario_wall_s": wall_s, **scenario_metrics}
        self.per_scenario_rows.append(row)

    def capture(self, state_hist, control_hist, pos_obs, vel_obs, goal) -> None:
        self._state_trajs.append(state_hist.detach().cpu().clone())
        self._control_trajs.append(control_hist.detach().cpu().clone())
        self._obs_state_trajs.append(pos_obs.detach().cpu().clone())
        self._obs_control_trajs.append(vel_obs.detach().cpu().clone())
        self._goals.append(goal.detach().cpu().clone())
