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


def test_dump_cell_writes_all_artifacts(tmp_path):
    import json

    rec = InstrumentationRecorder(n_scenarios=2, horizon=3, use_cuda=False)
    rec.begin_scenario(0)
    rec.begin_step(0)
    rec.start_section("sfm")
    rec.end_section("sfm")
    state_hist = torch.zeros(3, 4)
    control_hist = torch.zeros(2, 3)
    pos_obs = torch.zeros(5, 2, 3)
    vel_obs = torch.zeros(5, 2, 3)
    goal = torch.tensor([0.0, 0.0])
    rec.capture(state_hist, control_hist, pos_obs, vel_obs, goal)
    rec.end_scenario(
        0, scenario_metrics={"coll": 0, "dist": 1.0, "mean_step_ms": 5.0, "n_obs": 5}
    )

    rec.dump_cell(tmp_path)

    assert (tmp_path / "dump" / "state_traj.pt").exists()
    assert (tmp_path / "dump" / "control_traj.pt").exists()
    assert (tmp_path / "dump" / "obs_state_traj.pt").exists()
    assert (tmp_path / "dump" / "obs_control_traj.pt").exists()
    assert (tmp_path / "dump" / "goal.pt").exists()
    assert (tmp_path / "section_times.npz").exists()
    assert (tmp_path / "per_scenario.jsonl").exists()

    rows = [
        json.loads(line)
        for line in (tmp_path / "per_scenario.jsonl").read_text().splitlines()
    ]
    assert len(rows) == 1
    assert rows[0]["idx"] == 0
    assert rows[0]["coll"] == 0
    assert rows[0]["mean_step_ms"] == 5.0
    assert "scenario_wall_s" in rows[0]

    npz = np.load(tmp_path / "section_times.npz")
    assert set(npz.keys()) == {"cfm_ms", "mppi_ms", "sfm_ms"}
    assert npz["sfm_ms"].shape == (2, 3)


def test_write_hyperparams_and_env(tmp_path):
    import json

    rec = InstrumentationRecorder(n_scenarios=1, horizon=1, use_cuda=False)
    rec.write_hyperparams(
        tmp_path, {"SAFE_MARGIN": 0.5, "MPPI_LAMBDA": 0.1, "dataset": "ucy"}
    )
    rec.write_env(tmp_path, requested_precision="fp32")
    hp = json.loads((tmp_path / "hyperparams.json").read_text())
    assert hp["SAFE_MARGIN"] == 0.5
    assert hp["dataset"] == "ucy"
    env = json.loads((tmp_path / "env.json").read_text())
    assert env["requested_precision"] == "fp32"
    assert "torch_version" in env
    assert "active_matmul_precision" in env
    assert "allow_tf32_matmul" in env
    assert "git_rev" in env
    assert "hostname" in env
    assert "cpu_model" in env
    assert "cpu_governor" in env
    assert "cpu_live_mhz_first4" in env
    assert "start_iso" in env
    assert "end_iso" in env


def test_start_section_before_begin_scenario_raises():
    rec = InstrumentationRecorder(n_scenarios=2, horizon=3, use_cuda=False)
    rec.begin_step(0)  # without begin_scenario
    with pytest.raises(RuntimeError, match="begin_scenario"):
        rec.start_section("sfm")


def test_end_scenario_without_begin_raises():
    rec = InstrumentationRecorder(n_scenarios=2, horizon=3, use_cuda=False)
    with pytest.raises(RuntimeError, match="begin_scenario"):
        rec.end_scenario(
            0,
            scenario_metrics={"coll": 0, "dist": 0.0, "mean_step_ms": 0.0, "n_obs": 0},
        )


def test_iso_timestamps_include_utc_offset():
    import re

    rec = InstrumentationRecorder(n_scenarios=1, horizon=1, use_cuda=False)
    assert re.search(r"\+00:00$|Z$", rec._start_iso), (
        f"start_iso must be UTC-anchored; got {rec._start_iso!r}"
    )
