import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("pyarrow")
pytest.importorskip("pyarrow.parquet")

from mmrt import config as cfg
from mmrt.cli import ingest as cli
from mmrt.execution.execution_tape import save_execution_tape
from mmrt.execution.linear_signal_builder import build_execution_linear_feature_dataset
from mmrt.features.specs import FEATURE_NAMES_HASH, FEATURE_SCHEMA, FEATURE_SPECS_HASH
from mmrt.storage import manifest as mf
from mmrt.storage import reader as rd
from tests.test_execution_feature_replay import make_tape


def _saved_tape(tmp_path: Path, **kwargs):
    tape = make_tape(**kwargs)
    tape_root = tmp_path / "tape"
    save_execution_tape(tape, tape_root)
    return tape, tape_root


def _run_ingest(tmp_path: Path, tape_root: Path, *extra_args: str) -> Path:
    dataset_root = tmp_path / "dataset"
    rc = cli.main([
        "--dataset-root", str(dataset_root),
        "--dataset-id", "ds-tape",
        "--tape-root", str(tape_root),
        *extra_args,
    ])
    assert rc == 0
    return dataset_root


def test_ingest_builds_dataset_from_tape(tmp_path, capsys):
    tape, tape_root = _saved_tape(tmp_path)
    dataset_root = _run_ingest(tmp_path, tape_root)
    summary = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert summary["status"] == "ok"
    assert summary["exchange"] == tape.manifest.exchange
    assert summary["symbol"] == tape.manifest.symbol
    assert summary["book_data_type"] == "incremental_book_L2"
    assert summary["rows"] > 0
    assert summary["decisions_emitted"] >= summary["rows"]

    reader = rd.open_dataset(str(dataset_root), validate_on_open=True)
    manifest = reader.manifest
    assert manifest.exchange == tape.manifest.exchange
    assert manifest.symbol == tape.manifest.symbol
    assert manifest.feature_schema["schema"] == FEATURE_SCHEMA
    assert manifest.pipeline_config["source_data_types"] == ["incremental_book_L2", "trades"]
    assert manifest.transform_config["feature_names_hash"] == FEATURE_NAMES_HASH
    assert manifest.transform_config["feature_specs_hash"] == FEATURE_SPECS_HASH
    assert manifest.transform_diagnostics["rows_seen"] == summary["decisions_emitted"]
    assert manifest.notes["ingest_counters"]["rows_written"] == summary["rows"]

    table = reader.read_table()
    assert table.num_rows == summary["rows"]
    features = np.column_stack([np.asarray(table[c]) for c in manifest.feature_columns])
    assert np.isfinite(features).all()
    labels = np.column_stack([np.asarray(table[c]) for c in manifest.label_columns])
    assert np.isfinite(labels).all()


def test_ingest_rows_match_execution_feature_replay_exactly(tmp_path):
    """Training rows and execution-side signal features must be identical.

    This is the train/serve guard: both paths replay the same tape through
    the same decision feature pipeline, so the stored feature matrix must be
    a bitwise prefix of the execution feature dataset.
    """
    tape, tape_root = _saved_tape(tmp_path)
    dataset_root = _run_ingest(tmp_path, tape_root)

    reader = rd.open_dataset(str(dataset_root), validate_on_open=True)
    table = reader.read_table()
    manifest = reader.manifest
    dataset_features = np.column_stack(
        [np.asarray(table[c], dtype=np.float32) for c in manifest.feature_columns]
    )
    dataset_local_ts = np.asarray(table[mf.LOCAL_TS_US_COLUMN], dtype=np.int64)

    serving = build_execution_linear_feature_dataset(
        tape, decision_interval_us=cfg.DEFAULT_DECISION_STRIDE_US
    )
    assert serving.feature_names == manifest.feature_columns
    n = dataset_features.shape[0]
    assert 0 < n <= serving.num_decisions
    np.testing.assert_array_equal(dataset_local_ts, serving.decision_local_ts_us[:n])
    np.testing.assert_array_equal(dataset_features, serving.features[:n])


def test_ingest_applies_chronological_splits(tmp_path):
    tape, tape_root = _saved_tape(tmp_path)
    events_ts = np.asarray(tape.arrays.events["local_ts_us"], dtype=np.int64)
    start = int(events_ts[0])
    end = int(events_ts[-1]) + 1
    mid_point = start + (end - start) * 2 // 3
    dataset_root = _run_ingest(
        tmp_path,
        tape_root,
        "--split-train", f"{start}:{mid_point}",
        "--split-val", f"{mid_point}:{end}",
        "--purge-before-us", "0",
        "--purge-after-us", "0",
        "--embargo-before-us", "0",
        "--embargo-after-us", "0",
    )
    reader = rd.open_dataset(str(dataset_root), validate_on_open=True)
    roles = {s.role.value for s in reader.manifest.splits}
    assert roles == {"train", "val"}


def test_ingest_rejects_non_default_stride(tmp_path):
    _, tape_root = _saved_tape(tmp_path)
    with pytest.raises(ValueError, match="decision_stride_us"):
        cli.main([
            "--dataset-root", str(tmp_path / "ds"),
            "--dataset-id", "ds",
            "--tape-root", str(tape_root),
            "--decision-stride-us", "250000",
        ])


def test_ingest_rejects_existing_manifest(tmp_path):
    _, tape_root = _saved_tape(tmp_path)
    dataset_root = _run_ingest(tmp_path, tape_root)
    with pytest.raises(FileExistsError):
        cli.main([
            "--dataset-root", str(dataset_root),
            "--dataset-id", "ds-again",
            "--tape-root", str(tape_root),
        ])


def test_ingest_split_args_require_train_and_val(tmp_path):
    _, tape_root = _saved_tape(tmp_path)
    with pytest.raises(ValueError, match="split-train"):
        cli.main([
            "--dataset-root", str(tmp_path / "ds"),
            "--dataset-id", "ds",
            "--tape-root", str(tape_root),
            "--split-test", "1:2",
        ])


def test_ingest_max_events_limits_replay(tmp_path, capsys):
    _, tape_root = _saved_tape(tmp_path, n_l2=240)
    _run_ingest(tmp_path, tape_root, "--max-events", "200")
    summary = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    full_root = tmp_path / "dataset_full"
    rc = cli.main([
        "--dataset-root", str(full_root),
        "--dataset-id", "ds-full",
        "--tape-root", str(tape_root),
    ])
    assert rc == 0
    full_summary = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert summary["rows"] < full_summary["rows"]
