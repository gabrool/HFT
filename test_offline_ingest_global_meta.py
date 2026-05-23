import importlib
import sys
import types


def _install_optional_dependency_stubs() -> None:
    """Install lightweight import-time stubs for optional model dependencies."""
    stub_modules = {
        "tqdm": {"tqdm": lambda iterable=None, *args, **kwargs: iterable if iterable is not None else []},
        "einops": {"rearrange": lambda x, *args, **kwargs: x, "repeat": lambda x, *args, **kwargs: x},
        "huggingface_hub": {"PyTorchModelHubMixin": type("PyTorchModelHubMixin", (), {})},
        "mamba_ssm": {},
        "mamba_ssm.ops": {},
        "mamba_ssm.ops.triton": {},
        "mamba_ssm.ops.triton.selective_state_update": {"selective_state_update": None},
        "mamba_ssm.ops.triton.layernorm_gated": {"RMSNorm": type("RMSNorm", (), {})},
        "mamba_ssm.distributed": {},
        "mamba_ssm.distributed.tensor_parallel": {
            "ColumnParallelLinear": type("ColumnParallelLinear", (), {}),
            "RowParallelLinear": type("RowParallelLinear", (), {}),
        },
        "mamba_ssm.distributed.distributed_utils": {
            "all_reduce": lambda *args, **kwargs: None,
            "reduce_scatter": lambda *args, **kwargs: None,
        },
        "mamba_ssm.ops.triton.ssd_combined": {
            "mamba_chunk_scan_combined": lambda *args, **kwargs: None,
            "mamba_split_conv1d_scan_combined": lambda *args, **kwargs: None,
        },
    }
    for module_name, attrs in stub_modules.items():
        module = types.ModuleType(module_name)
        for attr, value in attrs.items():
            setattr(module, attr, value)
        sys.modules.setdefault(module_name, module)


_install_optional_dependency_stubs()
offline_ingest = importlib.import_module("offline_ingest")


def test_global_meta_pruned153_event10_has_required_linear_keys(tmp_path):
    weeks = [
        "22-02-2026-to-28-02-2026",
        "01-03-2026-to-07-03-2026",
        "08-03-2026-to-14-03-2026",
        "15-03-2026-to-21-03-2026",
        "22-03-2026-to-28-03-2026",
    ]

    base_ts = 1_700_000_000_000
    week_span_ms = 7 * 24 * 60 * 60 * 1000

    week_metas = {}
    for i, wk in enumerate(weeks):
        start = base_ts + i * week_span_ms
        end = start + week_span_ms - 1
        week_metas[wk] = {
            "week": wk,
            "feature_schema": offline_ingest.FEATURE_SCHEMA,
            "feature_transform": offline_ingest.FEATURE_TRANSFORM,
            "feature_transform_policy": offline_ingest.FEATURE_TRANSFORM_POLICY,
            "feature_transform_spec_hash": offline_ingest.RAW_FEATURE_TRANSFORM_SPEC_HASH,
            "feature_dim_core": offline_ingest.RAW_FEATURE_DIM_CORE,
            "feature_dim_total": offline_ingest.RAW_FEATURE_DIM_TOTAL,
            "feature_names_hash": offline_ingest.RAW_FEATURE_NAMES_HASH,
            "aux_schema": offline_ingest.AUX_SCHEMA,
            "aux_dim": offline_ingest.AUX_DIM,
            "aux_names": list(offline_ingest.AUX_FEATURE_NAMES),
            "aux_feature_names": list(offline_ingest.AUX_FEATURE_NAMES),
            "lookback": offline_ingest.LOOKBACK,
            "rows_total": 100,
            "labels_total": 90,
            "decision_ts_range": {"min": int(start), "max": int(end)},
        }

    meta = offline_ingest.build_global_meta_from_week_metas(
        pairs=[(wk, None, None) for wk in weeks],
        week_metas=week_metas,
        protocol=offline_ingest.FIVE_WEEK_PROTOCOL,
        ingest_stage="full",
        router=None,
        week_quality_records={},
        duplicate_decision_ts_count=0,
    )

    assert meta["feature_dim_core"] == 153
    assert meta["feature_dim_total"] == 159
    assert meta["lookback"] == 10
    assert len(meta["feature_names"]) == 153
    assert len(meta["aux_feature_names"]) == 6
    assert "splits" in meta
    splits = meta["splits"]
    assert splits["protocol"] == offline_ingest.FIVE_WEEK_PROTOCOL
    assert "cmssl" in splits
    assert "train" in splits["cmssl"]
    assert "val" in splits["cmssl"]
    assert "test" in splits["cmssl"]
    assert splits["cmssl"]["train"]["weeks"] == weeks[:2]
    assert splits["cmssl"]["val"]["week"] == weeks[2]
    assert splits["cmssl"]["test"]["week"] == weeks[3]
    assert splits["eval"]["full"]["week"] == weeks[4]
    assert "rl" in splits
    assert {"train", "val", "test"}.issubset(splits["rl"].keys())


def test_offline_ingest_pruned153_event10_constants():
    assert offline_ingest.RAW_FEATURE_DIM_CORE == 153
    assert offline_ingest.RAW_FEATURE_DIM_TOTAL == 159
    assert offline_ingest.LOOKBACK == 10
    assert len(offline_ingest.AUX_FEATURE_NAMES) == 6
    assert "pruned153_event10" in offline_ingest.FEATURE_SCHEMA


def test_schema_not_old_v10_pruned153_regression():
    old_v10 = "cmssl17_1s_maker_rtcore_v8_raw_no_pca_pruned153_lb10_xformv2"
    assert offline_ingest.FEATURE_SCHEMA != old_v10
    assert "pruned143" not in offline_ingest.FEATURE_SCHEMA
