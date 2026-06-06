from pathlib import Path
import json

import numpy as np
import pytest
from mmrt.execution.execution_tape import execution_tape_manifest_from_dict
from mmrt.execution.linear_signal import load_linear_signal_artifact_npz
from mmrt.contracts import TimeRangeUS
from mmrt.storage import manifest as mf

ROOTS = (Path("mmrt"), Path("tests"))

FORBIDDEN_SUBSTRINGS = (
    "schema" + "_" + "version",
    "SCHEMA" + "_" + "VERSION",
    "_" + "V1",
    "_" + "V2",
    "_" + "V3",
    "_" + "v1",
    "_" + "v2",
    "_" + "v3",
    " " + "v1",
    " " + "v2",
    " " + "v3",
)

ALLOWED_SUBSTRINGS = (
    "parquet_version",
    "DEFAULT_PARQUET_VERSION",
)


def _iter_py_files():
    for root in ROOTS:
        for path in root.rglob("*.py"):
            yield path


def test_no_internal_legacy_schema_names():
    offenders = []
    for path in _iter_py_files():
        text = path.read_text(encoding="utf-8")
        for forbidden in FORBIDDEN_SUBSTRINGS:
            if forbidden not in text:
                continue
            for line_no, line in enumerate(text.splitlines(), 1):
                if forbidden in line and not any(allowed in line for allowed in ALLOWED_SUBSTRINGS):
                    offenders.append(f"{path}:{line_no}: {line.strip()}")
    assert offenders == []


def test_linear_signal_npz_rejects_legacy_schema_field(tmp_path):
    path = tmp_path / "linear_signals.npz"
    arr = np.ones(1, dtype=np.float32)
    payload = {
        name: arr
        for name in (
            "p_no_move",
            "p_move",
            "p_up_move",
            "p_down_move",
            "signed_move_prob",
            "expected_up_bps",
            "expected_down_bps",
            "expected_return_bps",
            "expected_abs_move_bps",
            "predicted_vol_bps",
            "confidence",
        )
    }
    payload.update(
        **{"schema" + "_" + "version": np.array("mmrt_execution_linear_signals" + "_" + "v3_aligned")},
        metadata_json=np.array(json.dumps({})),
        decision_event_index=np.array([0], dtype=np.int64),
        decision_local_ts_us=np.array([1], dtype=np.int64),
    )
    np.savez(path, **payload)
    with pytest.raises(ValueError, match="missing schema"):
        load_linear_signal_artifact_npz(path)


def test_checkpoint_rejects_legacy_schema_field(tmp_path):
    path = tmp_path / "checkpoint.pt"
    torch = pytest.importorskip("torch")
    from mmrt.cli.evaluate_execution_policy import _load_checkpoint

    torch.save({"schema" + "_" + "version": "mmrt_execution_ppo_checkpoint" + "_" + "v2_required_linear_signals"}, path)
    with pytest.raises(ValueError, match="checkpoint schema"):
        _load_checkpoint(path, device=torch.device("cpu"))


def test_storage_manifest_rejects_legacy_manifest_schema_field():
    payload = mf.make_manifest(
        dataset_id="ds",
        created_at_utc="2026-06-06T00:00:00Z",
        segments=(
            mf.StorageSegment(
                segment_key="seg",
                parquet_path="segments/seg.parquet",
                row_count=1,
                label_count=1,
                time_range=TimeRangeUS(1, 2),
                local_time_range=TimeRangeUS(1, 2),
                first_row_idx=0,
                last_row_idx=0,
            ),
        ),
    ).to_dict()
    payload["manifest_schema" + "_" + "version"] = payload.pop("schema")
    with pytest.raises(ValueError, match="missing required key"):
        mf.StorageManifest.from_dict(payload)


def test_execution_tape_manifest_rejects_legacy_schema_field():
    payload = {
        "schema" + "_" + "version": "mmrt_execution_tape" + "_" + "v2_book_depth",
        "tape_format": "l2_trades_arrays",
        "exchange": "binance-futures",
        "symbol": "BTCUSDT",
        "symbol_spec": {
            "exchange": "binance-futures",
            "symbol": "BTCUSDT",
            "tick_size": 0.1,
            "step_size": 0.001,
            "min_qty": 0.001,
            "max_qty": 100.0,
        },
    }
    with pytest.raises(ValueError, match="schema"):
        execution_tape_manifest_from_dict(payload)
