# vendor/cfm_mppi_discovery_eval/tests/test_instrumentation.py
import time

import numpy as np

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
