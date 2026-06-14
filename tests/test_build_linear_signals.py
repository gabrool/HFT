import json

import pytest

from mmrt.cli.build_linear_signals import BuildLinearSignalsConfig, _config_from_args, build_arg_parser, build_linear_signals_from_config
from mmrt.execution.decision_grid import save_decision_grid
from mmrt.execution.execution_tape import load_execution_tape, save_execution_tape
from mmrt.execution.env import ExecutionEnv
from mmrt.execution.linear_signal import LINEAR_SIGNALS_FILENAME, load_linear_signal_artifact_npz
from mmrt.linear import models as lm
from tests.test_execution_linear_signal_builder import _SCHED50, _tiny_tape, _train_result, _preprocess_state
from mmrt.linear import train as tr
from tests.grid_helpers import decision_grid_for_tape


def _write_result(path, result):
    path.write_text(json.dumps(result.as_dict(), sort_keys=True, allow_nan=False), encoding="utf-8")
    return path


def test_build_linear_signals_cli_end_to_end(tmp_path):
    tape_root = tmp_path / "tape"
    tape = _tiny_tape()
    grid = decision_grid_for_tape(tape, schedule_config=_SCHED50)
    save_execution_tape(tape, tape_root, overwrite=True)
    save_decision_grid(tape_root / "decision_grid", grid, overwrite=True)
    cols = ("x_mid_slope_bps_per_sec_1000000us", "x_time_since_mid_change_us", "x_bid_l1_notional_usd")
    result_path = _write_result(tmp_path / "linear_train_result.json", _train_result({head: cols for head in lm.MODEL_HEADS}, grid=grid))
    summary = build_linear_signals_from_config(
        BuildLinearSignalsConfig(
            tape_root=str(tape_root),
            decision_grid_path=str(tape_root / "decision_grid"),
            linear_train_result_json=str(result_path),
            chunk_rows=1,
        )
    )
    assert (tape_root / LINEAR_SIGNALS_FILENAME).exists()
    assert (tape_root / "linear_signals_summary.json").exists()
    artifact = load_linear_signal_artifact_npz(tape_root / LINEAR_SIGNALS_FILENAME)
    assert artifact.n_rows == summary["feature_dataset"]["num_decisions"]
    assert summary["resource_mode"]["chunked_features"] is True
    assert summary["resource_mode"]["disk_backed_signal_writers"] is True
    assert summary["resource_mode"]["single_pass_feature_replay"] is True
    env = ExecutionEnv(load_execution_tape(tape_root), decision_grid=grid, linear_signals=artifact)
    reset = env.reset()
    assert reset.info["event_index"] == int(artifact.decision_event_index[0])
    loaded_summary = json.loads((tape_root / "linear_signals_summary.json").read_text(encoding="utf-8"))
    assert loaded_summary["run_type"] == "build_linear_signals"


def test_build_linear_signals_overwrite_guard(tmp_path):
    tape_root = tmp_path / "tape"
    tape = _tiny_tape()
    grid = decision_grid_for_tape(tape, schedule_config=_SCHED50)
    save_execution_tape(tape, tape_root, overwrite=True)
    save_decision_grid(tape_root / "decision_grid", grid, overwrite=True)
    cols = ("x_mid_slope_bps_per_sec_1000000us",)
    result_path = _write_result(tmp_path / "linear_train_result.json", _train_result({head: cols for head in lm.MODEL_HEADS}, grid=grid))
    (tape_root / LINEAR_SIGNALS_FILENAME).write_bytes(b"exists")
    with pytest.raises(FileExistsError):
        build_linear_signals_from_config(BuildLinearSignalsConfig(str(tape_root), str(tape_root / "decision_grid"), str(result_path)))


def test_build_linear_signals_parser_no_mmap():
    args = build_arg_parser().parse_args([
        "--tape-root", "tape",
        "--decision-grid", "decision_grid",
        "--linear-train-result-json", "linear_train_result.json",
        "--no-mmap",
        "--chunk-rows", "7",
    ])
    cfg = _config_from_args(args)
    assert cfg.mmap_mode is None
    assert cfg.chunk_rows == 7


def test_build_linear_signals_rejects_feature_mismatch(tmp_path):
    tape_root = tmp_path / "tape"
    tape = _tiny_tape()
    grid = decision_grid_for_tape(tape, schedule_config=_SCHED50)
    save_execution_tape(tape, tape_root, overwrite=True)
    save_decision_grid(tape_root / "decision_grid", grid, overwrite=True)
    result_path = _write_result(tmp_path / "linear_train_result.json", _train_result({head: ("x_missing_feature",) for head in lm.MODEL_HEADS}, grid=grid))
    with pytest.raises(ValueError, match="x_missing_feature"):
        build_linear_signals_from_config(BuildLinearSignalsConfig(str(tape_root), str(tape_root / "decision_grid"), str(result_path)))
