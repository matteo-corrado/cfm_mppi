# vendor/cfm_mppi_discovery_eval/tests/test_instrumentation.py
import numpy as np

from cfm_mppi.instrumentation import InstrumentationRecorder


def test_recorder_init_allocates_section_arrays():
    rec = InstrumentationRecorder(n_scenarios=3, horizon=5, use_cuda=False)
    for name in ("cfm", "mppi", "sfm"):
        arr = rec.section_times_ms[name]
        assert arr.shape == (3, 5)
        assert arr.dtype == np.float32
        assert np.isnan(arr).all(), f"{name} should start NaN-filled"
