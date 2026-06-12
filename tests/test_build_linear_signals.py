import json

import pytest

from mmrt.cli.build_linear_signals import BuildLinearSignalsConfig, _config_from_args, build_arg_parser, build_linear_signals_from_config
from mmrt.execution.execution_tape import load_execution_tape, save_execution_tape
from mmrt.execution.env import ExecutionEnv
from mmrt.execution.linear_signal import LINEAR_SIGNALS_FILENAME, load_linear_signal_artifact_npz
from mmrt.linear import models as lm
from tests.test_execution_linear_signal_builder import _tiny_tape, _train_result, _preprocess_state
from mmrt.linear import train as tr


def _write_result(path, result):
    path.write_text(json.dumps(result.as_dict(), sort_keys=True, allow_nan=False), encoding="utf-8")
    return path


def test_build_linear_signals_cli_end_to_end(tmp_path):
    tape_root = tmp_path / "tape"
    save_execution_tape(_tiny_tape(), tape_root, overwrite=True)
    cols = ("x_mid_slope_bps_per_sec_1000000us", "x_time_since_mid_change_us", "x_bid_l1_notional_usd")
    result_path = _write_result(tmp_path / "linear_train_result.json", _train_result({head: cols for head in lm.MODEL_HEADS}))
    summary = build_linear_signals_from_config(
        BuildLinearSignalsConfig(
            tape_root=str(tape_root),
            linear_train_result_json=str(result_path),
        )
    )
    assert (tape_root / LINEAR_SIGNALS_FILENAME).exists()
    assert (tape_root / "linear_signals_summary.json").exists()
    artifact = load_linear_signal_artifact_npz(tape_root / LINEAR_SIGNALS_FILENAME)
    assert artifact.n_rows == summary["feature_dataset"]["num_decisions"]
    env = ExecutionEnv(load_execution_tape(tape_root), linear_signals=artifact)
    reset = env.reset()
    assert reset.info["event_index"] == int(artifact.decision_event_index[0])
    loaded_summary = json.loads((tape_root / "linear_signals_summary.json").read_text(encoding="utf-8"))
    assert loaded_summary["run_type"] == "build_linear_signals"


def test_build_linear_signals_overwrite_guard(tmp_path):
    tape_root = tmp_path / "tape"
    save_execution_tape(_tiny_tape(), tape_root, overwrite=True)
    cols = ("x_mid_slope_bps_per_sec_1000000us",)
    result_path = _write_result(tmp_path / "linear_train_result.json", _train_result({head: cols for head in lm.MODEL_HEADS}))
    (tape_root / LINEAR_SIGNALS_FILENAME).write_bytes(b"exists")
    with pytest.raises(FileExistsError):
        build_linear_signals_from_config(BuildLinearSignalsConfig(str(tape_root), str(result_path)))


def test_build_linear_signals_parser_no_mmap():
    args = build_arg_parser().parse_args([
        "--tape-root", "tape",
        "--linear-train-result-json", "linear_train_result.json",
        "--no-mmap",
    ])
    cfg = _config_from_args(args)
    assert cfg.mmap_mode is None


def test_build_linear_signals_rejects_feature_mismatch(tmp_path):
    tape_root = tmp_path / "tape"
    save_execution_tape(_tiny_tape(), tape_root, overwrite=True)
    result_path = _write_result(tmp_path / "linear_train_result.json", _train_result({head: ("x_missing_feature",) for head in lm.MODEL_HEADS}))
    with pytest.raises(ValueError, match="x_missing_feature"):
        build_linear_signals_from_config(BuildLinearSignalsConfig(str(tape_root), str(result_path)))
