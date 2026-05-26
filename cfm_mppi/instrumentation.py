"""Measurement-only instrumentation for CFM-MPPI evaluation.

Adds section timers (CFM/MPPI/SFM), per-scenario captures, and dump
artifacts without altering the numerical algorithm. CUDA timing uses
torch.cuda.Event (async, no sync inside the t-loop).
"""

from __future__ import annotations

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
