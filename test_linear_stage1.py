from pathlib import Path

import pytest

try:
    from test_feature_event_result_contract import _install_optional_dependency_stubs
except Exception:
    _install_optional_dependency_stubs = None

if _install_optional_dependency_stubs is not None:
    _install_optional_dependency_stubs()


def test_summarize_linear_trim_stats_exports_jsonable_arrays():
    np = pytest.importorskip("numpy")
    pytest.importorskip("torch")
    from linear_offline import summarize_linear_trim_stats

    stats = {
        "pos_lo_raw_bps": np.array([0.1, 0.2], dtype=np.float32),
        "pos_hi_raw_bps": np.array([1.1, 1.2], dtype=np.float32),
        "neg_lo_abs_bps": np.array([0.3, 0.4], dtype=np.float32),
        "neg_hi_abs_bps": np.array([1.3, 1.4], dtype=np.float32),
        "kept_pos_q50_abs_raw_bps": np.array([0.5, 0.6], dtype=np.float32),
        "kept_neg_q50_abs_raw_bps": np.array([0.7, 0.8], dtype=np.float32),
        "ignored_extra_key": np.array([9.0], dtype=np.float32),
    }

    summary = summarize_linear_trim_stats(stats)

    assert set(summary) == {
        "pos_lo_raw_bps",
        "pos_hi_raw_bps",
        "neg_lo_abs_bps",
        "neg_hi_abs_bps",
        "kept_pos_q50_abs_raw_bps",
        "kept_neg_q50_abs_raw_bps",
    }
    assert summary["pos_lo_raw_bps"] == [float(np.float32(0.1)), float(np.float32(0.2))]
    assert isinstance(summary["kept_neg_q50_abs_raw_bps"], list)


def test_stage1_stats_only_payload_shape():
    payload = {
        "stage": "stage1",
        "status": "ok",
        "purpose": "linear_trim_stats_only",
        "trim_stats_cache_path": "/tmp/linear_signed_side_trim_stats_cache.npz",
        "train_label_rows": 3,
        "stats_summary": {"pos_lo_raw_bps": [0.0]},
    }

    assert payload["stage"] == "stage1"
    assert payload["purpose"] == "linear_trim_stats_only"
    assert payload["trim_stats_cache_path"]
    assert payload["train_label_rows"] > 0
    assert "stats_summary" in payload
    assert "prior" not in payload
    assert "val_fast_metrics" not in payload
    assert "val_full_metrics" not in payload
    assert "test_metrics" not in payload


def test_no_constant_prior_stage1_symbols():
    pytest.importorskip("numpy")
    pytest.importorskip("torch")
    import linear_offline

    forbidden = [
        "LinearConstantPriorModel",
        "build_constant_priors_from_train_labels",
        "linear_model_summary",
        "LINEAR_STAGE2_RUN_PRIOR_EVAL",
    ]
    for name in forbidden:
        assert not hasattr(linear_offline, name), name


def test_no_constant_prior_stage1_source_text():
    text = Path("linear_offline.py").read_text(encoding="utf-8")
    forbidden = [
        "linear_stage1_prior",
        "linear_val_fast",
        "linear_val_full",
        "linear-prior",
        "linear-prior-mag",
        "constant_prior",
        "BYBIT_LINEAR_STAGE2_RUN_PRIOR_EVAL",
    ]
    for needle in forbidden:
        assert needle not in text
