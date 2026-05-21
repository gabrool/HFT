import offline_ingest


def test_global_meta_pruned159_lb10_has_required_linear_keys(tmp_path):
    weeks = [
        "22-02-2026-to-28-02-2026",
        "01-03-2026-to-07-03-2026",
        "08-03-2026-to-14-03-2026",
        "15-03-2026-to-21-03-2026",
        "22-03-2026-to-28-03-2026",
    ]

    week_metas = {}
    for wk in weeks:
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

    assert meta["feature_dim_core"] == 159
    assert meta["feature_dim_total"] == 165
    assert meta["lookback"] == 10
    assert len(meta["feature_names"]) == 159
    assert len(meta["aux_feature_names"]) == 6
    assert "splits" in meta
