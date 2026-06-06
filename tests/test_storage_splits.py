import inspect

import pytest

pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")

from mmrt.contracts import SplitRole, TimeRangeUS
from mmrt.features import specs
from mmrt.storage import manifest as mf
from mmrt.storage import reader as rd
from mmrt.storage import splits as sp
from mmrt.storage import writer as wr


def feature_values(scale=0.01):
    return tuple(float(i) * scale for i in range(specs.FEATURE_COUNT))


def label_values():
    return (1.0, -2.0, 0.5)


def local_ts_values(rows, start=1_000_000, step=500_000):
    return [start + i * step for i in range(rows)]


def assigned_rows(plan):
    out = []
    for entry in plan.entries:
        out.extend(range(entry.start_row, entry.end_row))
    return out


def replace_column(table, name, values, typ):
    idx = table.schema.get_field_index(name)
    return table.set_column(idx, pa.field(name, typ), pa.array(values, type=typ))


def row(i, *, local_ts_us=None, ts_us=None):
    t = 1_000_000 + i * 500_000
    lts = t if local_ts_us is None else local_ts_us
    return wr.DecisionRow(
        decision_index=i,
        ts_us=t if ts_us is None else ts_us,
        local_ts_us=lts,
        event_seq=i,
        raw_mid=100.0 + i,
        label_entry_ts_us=lts + 1_000,
        label_values=label_values(),
        feature_values=feature_values(),
    )


def make_dataset(tmp_path, *, rows=20, chunk_rows=5, row_fn=row):
    root = tmp_path / "d"
    w = wr.DecisionRowWriter(wr.WriterConfig(dataset_id="d", created_at_utc="2026", dataset_root=str(root), chunk_rows=chunk_rows))
    for i in range(rows):
        w.append(row_fn(i))
    manifest = w.finalize()
    return root, manifest




def split_time_range_for_rows(start_row: int, end_row: int, *, base_us: int = 1_000_000, step_us: int = 500_000) -> TimeRangeUS:
    if end_row <= start_row:
        raise ValueError("end_row must be > start_row")
    return TimeRangeUS(start_us=base_us + start_row * step_us, end_us=base_us + (end_row - 1) * step_us + 1)
def manifest_with_splits(manifest: mf.StorageManifest, splits: tuple[mf.SplitMetadata, ...]) -> mf.StorageManifest:
    return mf.StorageManifest(
        schema=manifest.schema,
        dataset_id=manifest.dataset_id,
        created_at_utc=manifest.created_at_utc,
        pipeline_config=manifest.pipeline_config,
        writer_metadata=manifest.writer_metadata,
        feature_schema=manifest.feature_schema,
        label_spec=manifest.label_spec,
        transform_config=manifest.transform_config,
        transform_diagnostics=manifest.transform_diagnostics,
        exchange=manifest.exchange,
        symbol=manifest.symbol,
        storage_format=manifest.storage_format,
        time_unit=manifest.time_unit,
        decision_stride_us=manifest.decision_stride_us,
        feature_columns=manifest.feature_columns,
        label_columns=manifest.label_columns,
        required_columns=manifest.required_columns,
        segments=manifest.segments,
        splits=tuple(splits),
        notes=manifest.notes,
    )


def test_public_api_boundary():
    assert sp.__all__ == [
        "DEFAULT_SPLIT_BATCH_SIZE",
        "SplitWindow",
        "SplitConfig",
        "SplitPlan",
        "chronological_windows",
        "build_split_plan",
        "apply_split_plan",
        "write_split_manifest",
        "build_and_write_splits",
    ]
    pub = " ".join(sp.__all__).lower()
    for t in ["by" + "bit", "cm" + "ssl", "aux", "pca", "target", "future"]:
        assert t not in pub


def test_no_forbidden_imports():
    src = inspect.getsource(sp)
    direct_bad = [
        "import pan" + "das", "from pan" + "das",
        "import po" + "lars", "from po" + "lars",
        "import to" + "rch", "from to" + "rch",
        "import sk" + "learn", "from sk" + "learn",
        "from mmrt.data", "import mmrt.data",
        "from mmrt.features.engine", "from mmrt.features.labels", "from mmrt.features.transforms",
        "CM" + "SSL", "offline_" + "ingest", "linear_" + "offline",
        "BY" + "BIT", "Mini" + "Rocket", "Multi" + "Rocket", "Hy" + "dra", "Ae" + "on",
        "P" + "CA", "Standard" + "Scaler",
        "stage" + "1", "stage" + "2", "stage" + "3", "stage" + "4", "stage" + "5",
    ]
    for token in direct_bad:
        assert token not in src
    assert "mmrt.storage.writer" not in src


def test_split_window_validation():
    sp.SplitWindow(role=SplitRole.TRAIN, start_local_ts_us=1, end_local_ts_us=2)
    sp.SplitWindow(role="train", start_local_ts_us=1, end_local_ts_us=2)
    with pytest.raises(ValueError):
        sp.SplitWindow(role="train", start_local_ts_us=0, end_local_ts_us=2)
    with pytest.raises(ValueError):
        sp.SplitWindow(role="train", start_local_ts_us=2, end_local_ts_us=2)
    with pytest.raises(ValueError):
        sp.SplitWindow(role="train", start_local_ts_us=True, end_local_ts_us=2)
    with pytest.raises(ValueError):
        sp.SplitWindow(role="train", start_local_ts_us=1, end_local_ts_us=False)
    with pytest.raises(ValueError):
        sp.SplitWindow(role="bad", start_local_ts_us=1, end_local_ts_us=2)


def test_split_config_validation():
    a = sp.SplitWindow(role="train", start_local_ts_us=1, end_local_ts_us=10)
    b = sp.SplitWindow(role="val", start_local_ts_us=10, end_local_ts_us=20)
    sp.SplitConfig(windows=(a, b))
    with pytest.raises(ValueError):
        sp.SplitConfig(windows=())
    with pytest.raises(ValueError):
        sp.SplitConfig(windows=(b, a))
    with pytest.raises(ValueError):
        sp.SplitConfig(windows=(a, sp.SplitWindow(role="val", start_local_ts_us=9, end_local_ts_us=15)))
    sp.SplitConfig(windows=(a, b, sp.SplitWindow(role="train", start_local_ts_us=20, end_local_ts_us=30)))
    for k in ("purge_before_us", "purge_after_us", "embargo_before_us", "embargo_after_us"):
        with pytest.raises(ValueError):
            sp.SplitConfig(windows=(a, b), **{k: -1})
    with pytest.raises(ValueError):
        sp.SplitConfig(windows=(a, b), min_rows_per_split=0)
    with pytest.raises(ValueError):
        sp.SplitConfig(windows=(a, b), allow_empty_roles=1)
    with pytest.raises(ValueError):
        sp.SplitConfig(windows=(a, b), validate_dataset_on_open=1)
    with pytest.raises(ValueError):
        sp.SplitConfig(windows=(a, b), batch_size=0)
    with pytest.raises(ValueError):
        sp.SplitConfig(windows=(a, b), batch_size=True)


def test_chronological_windows_helper():
    ws = sp.chronological_windows(train=(1, 2), val=(2, 3), test=(3, 4))
    assert tuple(w.role for w in ws) == (SplitRole.TRAIN, SplitRole.VAL, SplitRole.TEST)
    assert tuple((w.start_local_ts_us, w.end_local_ts_us) for w in ws) == ((1, 2), (2, 3), (3, 4))
    ws2 = sp.chronological_windows(train=(1, 2), val=(2, 3))
    assert tuple(w.role for w in ws2) == (SplitRole.TRAIN, SplitRole.VAL)
    with pytest.raises(ValueError):
        sp.chronological_windows(train=(2, 4), val=(3, 5))
    with pytest.raises(ValueError):
        sp.chronological_windows(train=(1, 2), val=(2, 3), test=(6, 5))
    with pytest.raises(ValueError):
        sp.chronological_windows(train=(1, 2), val=(2, 2))


def test_split_plan_validation_rejects_overlapping_entries():
    e1 = mf.SplitMetadata(role="train", segment_key="s0", start_row=0, end_row=5, local_time_range=TimeRangeUS(1, 6), embargo_before_us=0, embargo_after_us=0)
    e2 = mf.SplitMetadata(role="val", segment_key="s1", start_row=4, end_row=8, local_time_range=TimeRangeUS(6, 9), embargo_before_us=0, embargo_after_us=0)
    with pytest.raises(ValueError):
        sp.SplitPlan(dataset_id="d", entries=(e1, e2), purge_before_us=0, purge_after_us=0, embargo_before_us=0, embargo_after_us=0, source_windows=(sp.SplitWindow("train", 1, 10),))
    e3 = mf.SplitMetadata(role="val", segment_key="s1", start_row=5, end_row=8, local_time_range=TimeRangeUS(6, 9), embargo_before_us=0, embargo_after_us=0)
    sp.SplitPlan(dataset_id="d", entries=(e1, e3), purge_before_us=0, purge_after_us=0, embargo_before_us=0, embargo_after_us=0, source_windows=(sp.SplitWindow("train", 1, 10),))


def test_build_split_plan_basic_chronological(tmp_path):
    root, manifest = make_dataset(tmp_path, rows=20, chunk_rows=5)
    local_ts = local_ts_values(20)
    cfg = sp.SplitConfig(windows=sp.chronological_windows(train=(1_000_000, 4_000_000), val=(4_000_000, 7_000_000)), purge_before_us=0, purge_after_us=0, embargo_before_us=0, embargo_after_us=0)
    plan = sp.build_split_plan(str(root), cfg)
    exp_train = [i for i, t in enumerate(local_ts) if 1_000_000 <= t < 4_000_000]
    exp_val = [i for i, t in enumerate(local_ts) if 4_000_000 <= t < 7_000_000]
    assert plan.dataset_id == manifest.dataset_id
    assert plan.roles == (SplitRole.TRAIN, SplitRole.VAL)
    assert assigned_rows(plan) == exp_train + exp_val
    assert all(e.start_row < e.end_row for e in plan.entries)
    assert plan.total_rows == len(exp_train) + len(exp_val)


def test_build_split_plan_uses_local_ts_not_ts_us(tmp_path):
    def row_fn(i):
        return row(i, ts_us=10_000_000 + i, local_ts_us=1_000_000 + i * 500_000)

    root, _ = make_dataset(tmp_path, row_fn=row_fn)
    cfg = sp.SplitConfig(windows=(sp.SplitWindow("train", 1_000_000, 3_000_000),), purge_before_us=0, purge_after_us=0, embargo_before_us=0, embargo_after_us=0)
    plan = sp.build_split_plan(str(root), cfg)
    assert assigned_rows(plan) == [0, 1, 2, 3]


def test_build_split_plan_allows_equal_local_ts(tmp_path):
    vals = [1_000_000, 1_000_000, 2_000_000, 2_000_000, 3_000_000, 3_000_000]
    root, _ = make_dataset(tmp_path, rows=6, chunk_rows=3, row_fn=lambda i: row(i, local_ts_us=vals[i]))
    cfg = sp.SplitConfig(windows=(sp.SplitWindow("train", 1_000_000, 3_000_001),), purge_before_us=0, purge_after_us=0, embargo_before_us=0, embargo_after_us=0)
    plan = sp.build_split_plan(str(root), cfg)
    assert assigned_rows(plan) == [0, 1, 2, 3, 4, 5]
    assert plan.entries[0].local_time_range.start_us == 1_000_000
    assert plan.entries[-1].local_time_range.end_us == 3_000_001


def test_default_purge_embargo_uses_label_context(tmp_path):
    root, manifest = make_dataset(tmp_path, rows=20)
    cfg = sp.SplitConfig(windows=(sp.SplitWindow("train", 1_000_000, 9_000_000),))
    plan = sp.build_split_plan(str(root), cfg)
    ctx = manifest.label_spec.label_context_us
    assert plan.purge_before_us == ctx and plan.purge_after_us == ctx
    assert plan.embargo_before_us == ctx and plan.embargo_after_us == ctx
    assert all(e.embargo_before_us == ctx and e.embargo_after_us == ctx for e in plan.entries)


def test_explicit_purge_embargo_shrinks_windows(tmp_path):
    root, _ = make_dataset(tmp_path, rows=10, row_fn=lambda i: row(i, local_ts_us=1_000 + i * 1_000, ts_us=10_000 + i))
    cfg = sp.SplitConfig(windows=(sp.SplitWindow("train", 1_000, 11_000),), purge_before_us=1_000, purge_after_us=1_000, embargo_before_us=1_000, embargo_after_us=1_000)
    plan = sp.build_split_plan(str(root), cfg)
    assert assigned_rows(plan) == [2, 3, 4, 5, 6, 7]


def test_window_with_too_few_rows_rejected(tmp_path):
    root, _ = make_dataset(tmp_path, rows=5)
    cfg = sp.SplitConfig(windows=(sp.SplitWindow("train", 1_000_000, 3_000_000),), min_rows_per_split=10, allow_empty_roles=False)
    with pytest.raises(ValueError):
        sp.build_split_plan(str(root), cfg)


def test_window_with_too_few_rows_skipped_when_allowed(tmp_path):
    root, _ = make_dataset(tmp_path, rows=5)
    cfg = sp.SplitConfig(windows=(sp.SplitWindow("train", 1_000_000, 3_000_000),), min_rows_per_split=10, allow_empty_roles=True)
    plan = sp.build_split_plan(str(root), cfg)
    assert plan.entries == () or plan.entries_for("train") == ()


def test_apply_split_plan_replaces_existing_splits(tmp_path):
    _, manifest = make_dataset(tmp_path, rows=6)
    old = mf.SplitMetadata(role="train", segment_key=manifest.segments[0].segment_key, start_row=0, end_row=2, local_time_range=TimeRangeUS(1_000_000, 1_500_000), embargo_before_us=0, embargo_after_us=0)
    m2 = manifest_with_splits(manifest, (old,))
    new = mf.SplitMetadata(role="val", segment_key=manifest.segments[0].segment_key, start_row=2, end_row=4, local_time_range=TimeRangeUS(2_000_000, 2_500_000), embargo_before_us=0, embargo_after_us=0)
    plan = sp.SplitPlan(dataset_id=manifest.dataset_id, entries=(new,), purge_before_us=0, purge_after_us=0, embargo_before_us=0, embargo_after_us=0, source_windows=(sp.SplitWindow("val", 1, 10),))
    out = sp.apply_split_plan(m2, plan, replace_existing=True)
    assert out.splits == plan.entries
    assert m2.splits == (old,)


def test_apply_split_plan_append_rejects_overlap(tmp_path):
    _, manifest = make_dataset(tmp_path, rows=10, chunk_rows=10)
    old = mf.SplitMetadata(role="train", segment_key=manifest.segments[0].segment_key, start_row=0, end_row=5, local_time_range=split_time_range_for_rows(0, 5), embargo_before_us=0, embargo_after_us=0)
    m2 = manifest_with_splits(manifest, (old,))
    p = mf.SplitMetadata(role="val", segment_key=manifest.segments[0].segment_key, start_row=4, end_row=6, local_time_range=split_time_range_for_rows(4, 6), embargo_before_us=0, embargo_after_us=0)
    plan = sp.SplitPlan(dataset_id=manifest.dataset_id, entries=(p,), purge_before_us=0, purge_after_us=0, embargo_before_us=0, embargo_after_us=0, source_windows=(sp.SplitWindow("val", 1, 10),))
    with pytest.raises(ValueError):
        sp.apply_split_plan(m2, plan, replace_existing=False)


def test_apply_split_plan_append_nonoverlap(tmp_path):
    _, manifest = make_dataset(tmp_path, rows=10, chunk_rows=10)
    old = mf.SplitMetadata(role="train", segment_key=manifest.segments[0].segment_key, start_row=0, end_row=5, local_time_range=split_time_range_for_rows(0, 5), embargo_before_us=0, embargo_after_us=0)
    m2 = manifest_with_splits(manifest, (old,))
    p = mf.SplitMetadata(role="val", segment_key=manifest.segments[0].segment_key, start_row=5, end_row=10, local_time_range=split_time_range_for_rows(5, 10), embargo_before_us=0, embargo_after_us=0)
    plan = sp.SplitPlan(dataset_id=manifest.dataset_id, entries=(p,), purge_before_us=0, purge_after_us=0, embargo_before_us=0, embargo_after_us=0, source_windows=(sp.SplitWindow("val", 1, 10),))
    out = sp.apply_split_plan(m2, plan, replace_existing=False)
    assert out.splits == (old, p)


def test_write_split_manifest_atomic_update(tmp_path):
    root, manifest = make_dataset(tmp_path)
    seg_paths = [root / s.parquet_path for s in manifest.segments]
    cfg = sp.SplitConfig(windows=sp.chronological_windows(train=(1_000_000, 4_000_000), val=(4_000_000, 7_000_000)), purge_before_us=0, purge_after_us=0, embargo_before_us=0, embargo_after_us=0)
    plan = sp.build_split_plan(str(root), cfg)
    sp.write_split_manifest(str(root), plan)
    m2 = mf.read_manifest_json(root / mf.DEFAULT_MANIFEST_FILENAME)
    assert len(m2.splits) == len(plan.entries)
    assert all(p.exists() for p in seg_paths)
    r = rd.open_dataset(str(root))
    assert r.read_split_table("train", columns=(mf.ROW_IDX_COLUMN,)).num_rows > 0


def test_build_and_write_splits_convenience(tmp_path):
    root, _ = make_dataset(tmp_path)
    cfg = sp.SplitConfig(windows=sp.chronological_windows(train=(1_000_000, 4_000_000), val=(4_000_000, 7_000_000)), purge_before_us=0, purge_after_us=0, embargo_before_us=0, embargo_after_us=0)
    out = sp.build_and_write_splits(str(root), cfg)
    assert len(out.splits) > 0
    assert rd.open_dataset(str(root)).read_split_table("train", columns=(mf.ROW_IDX_COLUMN,)).num_rows > 0


def test_reader_consumes_generated_splits(tmp_path):
    root, _ = make_dataset(tmp_path)
    local_ts = local_ts_values(20)
    cfg = sp.SplitConfig(windows=sp.chronological_windows(train=(1_000_000, 4_000_000), val=(4_000_000, 7_000_000)), purge_before_us=0, purge_after_us=0, embargo_before_us=0, embargo_after_us=0)
    plan = sp.build_split_plan(str(root), cfg)
    sp.write_split_manifest(str(root), plan)
    exp_train = [i for i, t in enumerate(local_ts) if 1_000_000 <= t < 4_000_000]
    t = rd.open_dataset(str(root)).read_split_table("train", columns=(mf.ROW_IDX_COLUMN,))
    assert t[mf.ROW_IDX_COLUMN].to_pylist() == exp_train


def test_build_split_plan_rejects_nonmonotonic_reader_rows(tmp_path):
    root, manifest = make_dataset(tmp_path)
    seg = manifest.segments[0]
    p = root / seg.parquet_path
    table = pq.read_table(p)
    vals = table[mf.LOCAL_TS_US_COLUMN].to_pylist()
    vals[-1] = vals[0] - 1
    pq.write_table(replace_column(table, mf.LOCAL_TS_US_COLUMN, vals, pa.int64()), p)
    cfg = sp.SplitConfig(windows=(sp.SplitWindow("train", 1_000_000, 4_000_000),), validate_dataset_on_open=True)
    with pytest.raises(ValueError):
        sp.build_split_plan(str(root), cfg)


def test_repeated_role_windows_are_validated_against_correct_source_window(tmp_path):
    root, _ = make_dataset(tmp_path, rows=20)
    local_ts = local_ts_values(20)
    cfg = sp.SplitConfig(windows=(sp.SplitWindow("train", 1_000_000, 2_000_000), sp.SplitWindow("val", 2_000_000, 3_000_000), sp.SplitWindow("train", 3_000_000, 4_000_000)), purge_before_us=0, purge_after_us=0, embargo_before_us=0, embargo_after_us=0)
    plan = sp.build_split_plan(str(root), cfg)
    assert plan.roles == (SplitRole.TRAIN, SplitRole.VAL)
    train = plan.entries_for(SplitRole.TRAIN)
    assert len(train) > 0
    expected = [i for i, t in enumerate(local_ts) if 1_000_000 <= t < 2_000_000 or 3_000_000 <= t < 4_000_000]
    actual = [i for e in train for i in range(e.start_row, e.end_row)]
    assert actual == expected


def test_no_feature_or_label_columns_read_by_split_plan():
    src = inspect.getsource(sp.build_split_plan)
    assert "ROW_IDX_COLUMN" in src
    assert "LOCAL_TS_US_COLUMN" in src
    for b in ["x_columns", "y_columns", "feature_columns", "label_columns"]:
        assert b not in src


def test_no_future_leakage_or_repair_surface():
    src = inspect.getsource(sp)
    bad = [
        "future_" + "mid", "future_" + "ret", "fit_" + "transform", "Standard" + "Scaler", "P" + "CA", "GRACE_" + "MS",
        "drop_duplicate_" + "trades", "stride_" + "rows", "sort_" + "values", "re" + "pair", "de" + "dupe", "sh" + "uffle", "rand" + "om",
    ]
    assert all(b not in src for b in bad)
