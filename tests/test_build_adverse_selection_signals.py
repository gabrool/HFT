import json
from pathlib import Path

import numpy as np
import pytest

from mmrt.cli.build_adverse_selection_signals import (
    BuildAdverseSelectionSignalsConfig,
    _config_from_args,
    build_adverse_selection_signals_from_config,
    build_arg_parser,
)
from mmrt.cli.train_adverse_selection import AdverseSelectionTrainCLIConfig, run_adverse_selection_training
from mmrt.execution.adverse_signal import (
    ADVERSE_SELECTION_MODEL_SCHEMA,
    ADVERSE_SELECTION_SIGNALS_SCHEMA,
    AdverseSelectionModelArtifact,
    load_adverse_selection_model,
    load_adverse_selection_signals,
    save_adverse_selection_model,
)
from tests.test_adverse_selection import _l2, _save_tape, _split_source_dataset_root, _tape, _trade
from mmrt.contracts import AggressorSide


def _training_root_and_model(tmp_path):
    tape = _tape(
        [
            _l2(seq=0, local_ts_us=100),
            _l2(seq=1, local_ts_us=200),
            _l2(seq=2, local_ts_us=300),
            _l2(seq=3, local_ts_us=400),
            _l2(seq=4, local_ts_us=500),
        ],
        [_trade(local_ts_us=150, side=AggressorSide.BUY, price_tick=1002, amount=1.0, source_row=0)],
    )
    root = _save_tape(tmp_path, tape)
    split_source = _split_source_dataset_root(tmp_path, root)
    model_npz = tmp_path / "model.npz"
    run_adverse_selection_training(
        AdverseSelectionTrainCLIConfig(
            tape_root=str(root),
            decision_grid_path=str(root / "decision_grid"),
            split_source_dataset_root=str(split_source),
            output_json=str(tmp_path / "train.json"),
            model_npz=str(model_npz),
            overwrite=True,
            flow_windows_us=(200,),
            kyle_sample_interval_us=50,
            kyle_response_horizon_us=100,
            kyle_windows_us=(200,),
            kyle_min_samples=1,
            quote_candidates="touch",
            order_qty=1.0,
            fill_horizon_us=100,
            adverse_horizon_us=100,
            drop_incomplete_horizon=False,
            min_train_samples=1,
        )
    )
    return root, model_npz


def test_build_adverse_selection_signals_end_to_end(tmp_path):
    root, model_npz = _training_root_and_model(tmp_path)
    output_npz = tmp_path / "signals.npz"
    output_json = tmp_path / "signals.json"
    summary = build_adverse_selection_signals_from_config(
        BuildAdverseSelectionSignalsConfig(
            tape_root=str(root),
            decision_grid_path=str(root / "decision_grid"),
            model_npz=str(model_npz),
            output_npz=str(output_npz),
            output_json=str(output_json),
            overwrite=True,
            mmap_mode=None,
        )
    )
    assert output_npz.exists()
    assert output_json.exists()
    loaded_summary = json.loads(output_json.read_text(encoding="utf-8"))
    model = load_adverse_selection_model(model_npz)
    signals = load_adverse_selection_signals(output_npz)
    assert signals.schema == ADVERSE_SELECTION_SIGNALS_SCHEMA
    assert signals.decision_event_seq.shape == signals.decision_local_ts_us.shape
    assert set(signals.target_names) == set(model.target_names)
    assert signals.adverse_label_config["queue_mode"] == "conservative"
    assert loaded_summary["adverse_label_config"] == signals.adverse_label_config
    assert loaded_summary["signals"]["adverse_label_config"] == signals.adverse_label_config
    assert loaded_summary["fill_simulator"] == signals.adverse_label_config
    assert summary["run_type"] == "build_adverse_selection_signals"
    assert loaded_summary["run_type"] == "build_adverse_selection_signals"


def test_build_adverse_selection_signals_rejects_symbol_mismatch(tmp_path):
    root, model_npz = _training_root_and_model(tmp_path)
    model = load_adverse_selection_model(model_npz)
    bad_model = tmp_path / "bad_model.npz"
    save_adverse_selection_model(
        bad_model,
        AdverseSelectionModelArtifact(
            schema=ADVERSE_SELECTION_MODEL_SCHEMA,
            feature_names=model.feature_names,
            target_names=model.target_names,
            feature_mean=model.feature_mean,
            feature_scale=model.feature_scale,
            coefficients=model.coefficients,
            intercepts=model.intercepts,
            config_json=model.config_json,
            exchange=model.exchange,
            symbol="OTHER",
            decision_grid_schema=model.decision_grid_schema,
            decision_grid_hash=model.decision_grid_hash,
            decision_grid_n_rows=model.decision_grid_n_rows,
            decision_schedule=model.decision_schedule,
            split_source_dataset_root=model.split_source_dataset_root,
            split_source_dataset_id=model.split_source_dataset_id,
            split_source_manifest_hash=model.split_source_manifest_hash,
            split_contract=model.split_contract,
        ),
    )
    with pytest.raises(ValueError, match="exchange/symbol"):
        build_adverse_selection_signals_from_config(
            BuildAdverseSelectionSignalsConfig(
                str(root),
                str(root / "decision_grid"),
                str(bad_model),
                output_npz=str(tmp_path / "x.npz"),
                output_json=str(tmp_path / "x.json"),
            )
        )


def test_build_adverse_selection_signals_rejects_feature_name_mismatch(tmp_path):
    root, model_npz = _training_root_and_model(tmp_path)
    model = load_adverse_selection_model(model_npz)
    bad_model = tmp_path / "bad_feature_model.npz"
    bad_feature_names = ("wrong",) + model.feature_names[1:]
    save_adverse_selection_model(
        bad_model,
        AdverseSelectionModelArtifact(
            schema=ADVERSE_SELECTION_MODEL_SCHEMA,
            feature_names=bad_feature_names,
            target_names=model.target_names,
            feature_mean=model.feature_mean,
            feature_scale=model.feature_scale,
            coefficients=model.coefficients,
            intercepts=model.intercepts,
            config_json=model.config_json,
            exchange=model.exchange,
            symbol=model.symbol,
            decision_grid_schema=model.decision_grid_schema,
            decision_grid_hash=model.decision_grid_hash,
            decision_grid_n_rows=model.decision_grid_n_rows,
            decision_schedule=model.decision_schedule,
            split_source_dataset_root=model.split_source_dataset_root,
            split_source_dataset_id=model.split_source_dataset_id,
            split_source_manifest_hash=model.split_source_manifest_hash,
            split_contract=model.split_contract,
        ),
    )
    with pytest.raises(ValueError, match="feature_names"):
        build_adverse_selection_signals_from_config(
            BuildAdverseSelectionSignalsConfig(
                str(root),
                str(root / "decision_grid"),
                str(bad_model),
                output_npz=str(tmp_path / "y.npz"),
                output_json=str(tmp_path / "y.json"),
            )
        )


def test_build_adverse_selection_signals_parser_no_mmap():
    args = build_arg_parser().parse_args([
        "--tape-root", "/tmp/tape",
        "--decision-grid", "/tmp/tape/decision_grid",
        "--model-npz", "/tmp/model.npz",
        "--no-mmap",
    ])
    cfg = _config_from_args(args)
    assert cfg.mmap_mode is None


def test_build_adverse_selection_signals_overwrite_guard(tmp_path):
    root, model_npz = _training_root_and_model(tmp_path)
    output_npz = tmp_path / "signals.npz"
    output_json = tmp_path / "signals.json"
    output_npz.write_bytes(b"exists")
    output_json.write_text("{}", encoding="utf-8")
    with pytest.raises(FileExistsError):
        build_adverse_selection_signals_from_config(
            BuildAdverseSelectionSignalsConfig(
                str(root),
                str(root / "decision_grid"),
                str(model_npz),
                output_npz=str(output_npz),
                output_json=str(output_json),
            )
        )


def test_build_adverse_selection_signals_does_not_import_rl():
    source = Path("mmrt/cli/build_adverse_selection_signals.py").read_text(encoding="utf-8")
    assert "mmrt.rl" not in source
    assert "gym" not in source
    assert "torch" not in source
