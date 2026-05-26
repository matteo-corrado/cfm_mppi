# vendor/cfm_mppi_discovery_eval/tests/test_instrumentation.py
import time

import numpy as np
import pytest
import torch

from cfm_mppi.instrumentation import InstrumentationRecorder


def test_recorder_init_allocates_section_arrays():
    rec = InstrumentationRecorder(n_scenarios=3, horizon=5, use_cuda=False)
    for name in ("cfm", "mppi", "sfm"):
        arr = rec.section_times_ms[name]
        assert arr.shape == (3, 5)
        assert arr.dtype == np.float32
        assert np.isnan(arr).all(), f"{name} should start NaN-filled"


def test_cpu_section_records_elapsed_ms():
    rec = InstrumentationRecorder(n_scenarios=2, horizon=3, use_cuda=False)
    rec.begin_scenario(0)
    rec.begin_step(1)
    rec.start_section("sfm")
    time.sleep(0.01)
    rec.end_section("sfm")
    rec.end_scenario(
        0, scenario_metrics={"coll": 0, "dist": 1.0, "mean_step_ms": 0.0, "n_obs": 0}
    )
    val = rec.section_times_ms["sfm"][0, 1]
    assert 5.0 <= val <= 50.0, f"expected ~10ms, got {val}"
    # other cells untouched
    assert np.isnan(rec.section_times_ms["sfm"][0, 0])
    assert np.isnan(rec.section_times_ms["sfm"][1, 1])


@pytest.mark.skipif(
    not __import__("torch").cuda.is_available(),
    reason="CUDA not available",
)
def test_cuda_section_records_elapsed_ms():
    import torch

    rec = InstrumentationRecorder(n_scenarios=1, horizon=2, use_cuda=True)
    rec.begin_scenario(0)
    rec.begin_step(0)
    rec.start_section("cfm")
    # synthetic GPU work
    x = torch.randn(2048, 2048, device="cuda")
    for _ in range(20):
        x = x @ x
    rec.end_section("cfm")
    rec.end_scenario(
        0, scenario_metrics={"coll": 0, "dist": 0.0, "mean_step_ms": 0.0, "n_obs": 0}
    )
    val = rec.section_times_ms["cfm"][0, 0]
    assert val > 0.0 and not np.isnan(val), f"cfm timing should be populated, got {val}"


def test_capture_appends_per_scenario_buffers():
    rec = InstrumentationRecorder(n_scenarios=2, horizon=3, use_cuda=False)
    state_hist = torch.zeros(3, 4)
    control_hist = torch.zeros(2, 3)
    pos_obs = torch.zeros(5, 2, 3)
    vel_obs = torch.zeros(5, 2, 3)
    goal = torch.tensor([1.0, 2.0])
    rec.capture(state_hist, control_hist, pos_obs, vel_obs, goal)
    rec.capture(state_hist + 1, control_hist + 1, pos_obs + 1, vel_obs + 1, goal + 1)
    assert len(rec._state_trajs) == 2
    assert rec._state_trajs[1][0, 0].item() == 1.0
    assert rec._goals[0].tolist() == [1.0, 2.0]
