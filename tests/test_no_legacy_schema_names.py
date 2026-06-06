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

PRODUCTION_ROOT = Path("mmrt")

FORBIDDEN_HISTORY_WORDS = (
    "leg" + "acy",
    "compat" + "ibility",
    "back" + "ward " + "compat" + "ibility",
    "no longer " + "supported",
    "old " + "schema",
    "old " + "format",
)


def test_no_history_language_in_production_mmrt():
    offenders = []
    for path in PRODUCTION_ROOT.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        for forbidden in FORBIDDEN_HISTORY_WORDS:
            for line_no, line in enumerate(text.splitlines(), 1):
                if forbidden in line.lower():
                    offenders.append(f"{path}:{line_no}: {line.strip()}")
    assert offenders == []


def test_no_feature_retired_name_layer_symbols():
    text = Path("mmrt/features/specs.py").read_text(encoding="utf-8")
    forbidden = (
        "leg" + "acy_name",
        "leg" + "acy_feature_names",
        "canonical_name_for_leg" + "acy",
        "SOURCE_TO_" + "CANONICAL",
        "CANONICAL_TO_" + "SOURCE",
    )
    offenders = [s for s in forbidden if s in text]
    assert offenders == []


def _iter_py_files():
    for root in ROOTS:
        for path in root.rglob("*.py"):
            yield path


def test_no_internal_schema_release_names():
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


def test_linear_signal_npz_rejects_retired_schema_field(tmp_path):
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


def test_checkpoint_rejects_retired_schema_field(tmp_path):
    path = tmp_path / "checkpoint.pt"
    torch = pytest.importorskip("torch")
    from mmrt.cli.evaluate_execution_policy import _load_checkpoint

    torch.save({"schema" + "_" + "version": "mmrt_execution_ppo_checkpoint" + "_" + "v2_required_linear_signals"}, path)
    with pytest.raises(ValueError, match="checkpoint schema"):
        _load_checkpoint(path, device=torch.device("cpu"))


def test_storage_manifest_rejects_retired_manifest_schema_field():
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


def test_execution_tape_manifest_rejects_retired_schema_field():
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


def test_no_dead_generic_contract_layer_symbols():
    text = Path("mmrt/contracts.py").read_text(encoding="utf-8")
    forbidden = (
        "Event" + "Type",
        "Event" + "Meta",
        "Price" + "Level",
        "Book" + "SnapshotEvent",
        "Book" + "DeltaEvent",
        "Trade" + "Event",
        "Book" + "TickerEvent",
        "Derivative" + "TickerEvent",
        "Liquidation" + "Event",
        "Market" + "Event",
        "Feature" + "BuildResult",
        "Decision" + "RowRef",
        "Segment" + "Spec",
        "Split" + "Entry",
        "Split" + "Plan",
        "Dataset" + "Manifest",
    )
    offenders = [name for name in forbidden if name in text]
    assert offenders == []


def test_no_unused_tardis_schema_symbols():
    text = Path("mmrt/schemas.py").read_text(encoding="utf-8")
    forbidden = (
        "BOOK" + "_SNAPSHOT_5",
        "BOOK" + "_TICKER",
        "DERIVATIVE" + "_TICKER",
        "LIQ" + "UIDATIONS",
        "OPTIONS" + "_CHAIN",
        "QU" + "OTES",
        "Feature" + "Field",
        "Feature" + "Schema",
        "DECISION" + "_ROW_FIXED_COLUMNS",
        "LABEL" + "_ROW_FIXED_COLUMNS",
    )
    offenders = [name for name in forbidden if name in text]
    assert offenders == []


def test_ingest_has_no_numeric_zero_coercion_or_skip_counters():
    text = Path("mmrt/cli/ingest.py").read_text(encoding="utf-8")
    forbidden = (
        "_to_float" + "_or_zero",
        "skipped_bad" + "_trade_events",
        "skipped_empty" + "_book_events",
    )
    offenders = [name for name in forbidden if name in text]
    assert offenders == []
    assert "def _book_snapshot_input_from_row" in text
    assert "return None" not in text.split("def _book_snapshot_input_from_row", 1)[1].split("def _trade_input_from_row", 1)[0]
    assert "return None" not in text.split("def _trade_input_from_row", 1)[1].split("@dataclass", 1)[0]


def test_ingest_has_no_fake_data_type_or_validation_opt_out_flags():
    text = Path("mmrt/cli/ingest.py").read_text(encoding="utf-8")
    forbidden = (
        "--book" + "-data-type",
        "--no" + "-validate-output",
        "--validate" + "-output",
        "args.validate" + "_output",
    )
    offenders = [s for s in forbidden if s in text]
    assert offenders == []


def test_tardis_csv_has_no_side_otherwise_unknown_fallback():
    text = Path("mmrt/data/tardis_csv.py").read_text(encoding="utf-8")
    assert ".otherwise(pl.lit(SIDE" + "_UNKNOWN))" not in text
    assert ".otherwise(pl.lit(BOOK" + "_SIDE_UNKNOWN))" not in text
    assert "BOOK" + "_SIDE_UNKNOWN" not in text


def test_adapter_has_no_source_context_accepted_policy_helpers():
    text = Path("mmrt/data/binance_futures_adapter.py").read_text(encoding="utf-8")
    forbidden = (
        "SOURCE" + "_DATA_TYPES",
        "CONTEXT" + "_DATA_TYPES",
        "ACCEPTED" + "_DATA_TYPES",
        "is_binance_futures_source" + "_data_type",
        "is_binance_futures_context" + "_data_type",
        "is_binance_futures_accepted" + "_data_type",
        "default_binance_futures_source" + "_data_types",
        "default_binance_futures_context" + "_data_types",
        "default_binance_futures_accepted" + "_data_types",
        "normalize_binance_futures_data" + "_types",
        "require_binance_futures_data" + "_type",
    )
    offenders = [s for s in forbidden if s in text]
    assert offenders == []


def test_no_allow_nan_true_in_mmrt():
    offenders = []
    for path in Path("mmrt").rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        if "allow_nan" + "=True" in text:
            offenders.append(str(path))
    assert offenders == []
