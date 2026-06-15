import inspect

import pytest

pytest.importorskip("pyarrow")

import pyarrow as pa
import pyarrow.parquet as pq

from mmrt.contracts import SplitRole, TimeRangeUS
from mmrt.features import specs
from mmrt.storage import manifest as mf
from mmrt.storage import reader as rd
from mmrt.storage import writer as wr


def feature_values(scale=0.01):
    return tuple(float(i) * scale for i in range(specs.FEATURE_COUNT))


def label_values():
    return (1.0, -2.0, 0.5)


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


def make_dataset(tmp_path, *, rows=5, chunk_rows=2, splits=(), row_fn=row):
    root = tmp_path / "d"
    w = wr.DecisionRowWriter(wr.WriterConfig(dataset_id="d", created_at_utc="2026", dataset_root=str(root), chunk_rows=chunk_rows))
    for i in range(rows):
        w.append(row_fn(i))
    manifest = w.finalize()
    if splits:
        m0 = mf.read_manifest_json(root / mf.DEFAULT_MANIFEST_FILENAME)
        m1 = mf.StorageManifest(
            m0.schema,
            m0.dataset_id,
            m0.created_at_utc,
            m0.pipeline_config,
            m0.writer_metadata,
            m0.feature_schema,
            m0.label_spec,
            m0.transform_config,
            m0.transform_diagnostics,
            m0.exchange,
            m0.symbol,
            m0.storage_format,
            m0.time_unit,
            m0.decision_schedule,
            m0.feature_columns,
            m0.label_columns,
            m0.required_columns,
            m0.segments,
            tuple(splits),
            m0.notes,
        )
        mf.write_manifest_json(m1, root / mf.DEFAULT_MANIFEST_FILENAME)
        manifest = m1
    return root, manifest


def replace_column(table: pa.Table, name: str, values, typ=None) -> pa.Table:
    idx = table.schema.get_field_index(name)
    field = table.schema.field(name) if typ is None else pa.field(name, typ)
    arr = pa.array(values, type=field.type)
    return table.set_column(idx, field, arr)


def test_public_api_boundary():
    assert rd.__all__ == ["DEFAULT_BATCH_SIZE", "ReaderConfig", "SegmentReadPlan", "StorageDatasetReader", "open_dataset"]
    bad = ("bybit", "cmssl", "aux", "pca", "target", "future")
    assert all(token not in " ".join(rd.__all__).lower() for token in bad)


def test_no_forbidden_imports():
    src = inspect.getsource(rd)
    direct_bad = [
        "import pan" + "das",
        "from pan" + "das",
        "import po" + "lars",
        "from po" + "lars",
        "import to" + "rch",
        "from to" + "rch",
        "import sk" + "learn",
        "from sk" + "learn",
        "from mmrt.features.engine",
        "from mmrt.features.labels",
        "from mmrt.features.transforms",
        "CM" + "SSL",
        "offline_" + "ingest",
        "linear_" + "offline",
        "BY" + "BIT",
        "Mini" + "Rocket",
        "Multi" + "Rocket",
        "Hy" + "dra",
        "Ae" + "on",
        "P" + "CA",
        "Standard" + "Scaler",
        "stage" + "1",
        "stage" + "2",
        "stage" + "3",
        "stage" + "4",
        "stage" + "5",
    ]
    for token in direct_bad:
        assert token not in src


def test_reader_config_validation(tmp_path):
    rd.ReaderConfig(dataset_root=str(tmp_path))
    with pytest.raises(ValueError):
        rd.ReaderConfig(dataset_root="")
    with pytest.raises(ValueError):
        rd.ReaderConfig(dataset_root="x", validate_on_open=1)
    with pytest.raises(ValueError):
        rd.ReaderConfig(dataset_root="x", batch_size=0)
    with pytest.raises(ValueError):
        rd.ReaderConfig(dataset_root="x", batch_size=True)


def test_open_dataset_loads_manifest_and_properties(tmp_path):
    root, m = make_dataset(tmp_path)
    r = rd.open_dataset(str(root))
    assert r.total_rows == m.total_rows
    assert r.total_labels == m.total_labels
    assert r.feature_columns == m.x_columns
    assert r.label_columns == m.y_columns
    assert r.required_columns == m.required_columns
    assert len(r.segments) == len(m.segments)


def test_missing_manifest_rejected(tmp_path):
    with pytest.raises(FileNotFoundError):
        rd.open_dataset(str(tmp_path / "x"))


def test_missing_segment_file_rejected(tmp_path):
    root, m = make_dataset(tmp_path)
    (root / m.segments[0].parquet_path).unlink()
    with pytest.raises(FileNotFoundError):
        rd.open_dataset(str(root), validate_on_open=True)


def test_extra_unmanifested_parquet_rejected(tmp_path):
    root, _ = make_dataset(tmp_path)
    (root / "segments" / "extra.parquet").write_bytes(b"")
    with pytest.raises(ValueError):
        rd.open_dataset(str(root), validate_on_open=True)


def test_validate_schema_rejects_missing_column_or_wrong_type(tmp_path):
    root, m = make_dataset(tmp_path)
    path = root / m.segments[0].parquet_path
    table = pq.read_table(path)
    pq.write_table(table.drop([m.required_columns[-1]]), path)
    with pytest.raises(ValueError):
        rd.open_dataset(str(root), validate_on_open=True)

    root2, m2 = make_dataset(tmp_path / "b")
    path2 = root2 / m2.segments[0].parquet_path
    table2 = pq.read_table(path2)
    bad2 = replace_column(table2, mf.RAW_MID_COLUMN, table2[mf.RAW_MID_COLUMN].to_pylist(), pa.float32())
    pq.write_table(bad2, path2)
    with pytest.raises(ValueError):
        rd.open_dataset(str(root2), validate_on_open=True)


def test_validate_rejects_row_count_mismatch(tmp_path):
    root, m = make_dataset(tmp_path)
    path = root / m.segments[0].parquet_path
    table = pq.read_table(path).slice(0, 1)
    pq.write_table(table, path)
    with pytest.raises(ValueError):
        rd.open_dataset(str(root), validate_on_open=True)


def test_validate_rejects_row_idx_mismatch(tmp_path):
    root, m = make_dataset(tmp_path)
    path = root / m.segments[0].parquet_path
    table = pq.read_table(path)
    bad = replace_column(table, mf.ROW_IDX_COLUMN, [99] * table.num_rows, pa.int64())
    pq.write_table(bad, path)
    with pytest.raises(ValueError):
        rd.open_dataset(str(root), validate_on_open=True)


def test_validate_rejects_decision_index_not_increasing(tmp_path):
    root, m = make_dataset(tmp_path)
    path = root / m.segments[0].parquet_path
    table = pq.read_table(path)
    bad = replace_column(table, mf.DECISION_INDEX_COLUMN, [0] * table.num_rows, pa.int64())
    pq.write_table(bad, path)
    with pytest.raises(ValueError):
        rd.open_dataset(str(root), validate_on_open=True)


def test_validate_rejects_local_ts_decreasing_but_allows_equal(tmp_path):
    root, _ = make_dataset(tmp_path / "ok", row_fn=lambda i: row(i, local_ts_us=1_000_000 + (i // 2)))
    rd.open_dataset(str(root), validate_on_open=True)

    root2, m2 = make_dataset(tmp_path / "bad")
    path = root2 / m2.segments[0].parquet_path
    table = pq.read_table(path)
    vals = table[mf.LOCAL_TS_US_COLUMN].to_pylist()
    vals[-1] = vals[0] - 1
    pq.write_table(replace_column(table, mf.LOCAL_TS_US_COLUMN, vals, pa.int64()), path)
    with pytest.raises(ValueError):
        rd.open_dataset(str(root2), validate_on_open=True)


def test_validate_rejects_timestamp_outside_segment_range(tmp_path):
    root, m = make_dataset(tmp_path)
    seg = m.segments[0]
    path = root / seg.parquet_path
    table = pq.read_table(path)
    ts = table[mf.TS_US_COLUMN].to_pylist()
    ts[0] = seg.time_range.end_us
    pq.write_table(replace_column(table, mf.TS_US_COLUMN, ts, pa.int64()), path)
    with pytest.raises(ValueError):
        rd.open_dataset(str(root), validate_on_open=True)

    root2, m2 = make_dataset(tmp_path / "b")
    seg2 = m2.segments[0]
    path2 = root2 / seg2.parquet_path
    table2 = pq.read_table(path2)
    lts = table2[mf.LOCAL_TS_US_COLUMN].to_pylist()
    lts[0] = seg2.local_time_range.end_us
    pq.write_table(replace_column(table2, mf.LOCAL_TS_US_COLUMN, lts, pa.int64()), path2)
    with pytest.raises(ValueError):
        rd.open_dataset(str(root2), validate_on_open=True)


def test_select_columns(tmp_path):
    root, m = make_dataset(tmp_path)
    r = rd.open_dataset(str(root))
    assert r.select_columns(include_labels=False, include_features=False) == mf.BASE_ROW_COLUMNS
    assert r.select_columns(include_base=False, include_features=False) == m.y_columns
    assert r.select_columns(include_base=False, include_labels=False, feature_columns=(m.x_columns[0],)) == (m.x_columns[0],)
    cols = r.select_columns(include_labels=False, include_features=False, extra_columns=(mf.RAW_MID_COLUMN,))
    assert cols == mf.BASE_ROW_COLUMNS
    with pytest.raises(ValueError):
        r.select_columns(feature_columns=("x_not_real",))
    with pytest.raises(ValueError):
        r.select_columns(label_columns=("y_not_real",))
    with pytest.raises(ValueError):
        r.select_columns(extra_columns=("z_not_real",))


def test_read_table_column_projection(tmp_path):
    root, m = make_dataset(tmp_path)
    r = rd.open_dataset(str(root))
    cols = (mf.ROW_IDX_COLUMN, mf.LOCAL_TS_US_COLUMN, m.x_columns[0])
    t = r.read_table(columns=cols)
    assert t.column_names == list(cols)
    assert t.num_rows == m.total_rows
    assert t[mf.ROW_IDX_COLUMN].to_pylist() == list(range(m.total_rows))


def test_iter_batches_uses_projection_and_batch_size(tmp_path):
    root, _ = make_dataset(tmp_path, rows=5)
    r = rd.open_dataset(str(root))
    batches = list(r.iter_batches(columns=(mf.ROW_IDX_COLUMN,), batch_size=2))
    assert [b.num_rows for b in batches] == [2, 2, 1]
    assert all(b.schema.names == [mf.ROW_IDX_COLUMN] for b in batches)


def test_read_segment_table(tmp_path):
    root, _ = make_dataset(tmp_path, rows=5, chunk_rows=2)
    r = rd.open_dataset(str(root))
    t = r.read_segment_table("seg_000001", columns=(mf.ROW_IDX_COLUMN,))
    assert t[mf.ROW_IDX_COLUMN].to_pylist() == [2, 3]


def test_dataset_uses_manifest_files_only(tmp_path):
    root, _ = make_dataset(tmp_path, rows=5, chunk_rows=2)
    r = rd.open_dataset(str(root))
    d = r.dataset(segments=("seg_000000",))
    t = d.to_table(columns=[mf.ROW_IDX_COLUMN])
    assert t.num_rows == 2
    with pytest.raises(ValueError):
        r.dataset(segments=("missing",))


def test_dataset_reuses_cached_arrow_dataset_for_same_segments(tmp_path):
    root, _ = make_dataset(tmp_path, rows=5, chunk_rows=2)
    r = rd.open_dataset(str(root))

    all_segments = r.dataset()
    same_all_segments = r.dataset()
    first_segment = r.dataset(segments=("seg_000000",))
    same_first_segment = r.dataset(segments=("seg_000000",))

    assert same_all_segments is all_segments
    assert same_first_segment is first_segment
    assert first_segment is not all_segments


def test_split_entries_and_read_split_table(tmp_path):
    root, m = make_dataset(tmp_path, rows=6, chunk_rows=3)
    splits = (
        mf.SplitMetadata(SplitRole.TRAIN, "seg_000000", 0, 2, TimeRangeUS(m.segments[0].local_time_range.start_us, m.segments[0].local_time_range.end_us)),
        mf.SplitMetadata(SplitRole.VAL, "seg_000001", 3, 6, TimeRangeUS(m.segments[1].local_time_range.start_us, m.segments[1].local_time_range.end_us)),
    )
    m2 = mf.StorageManifest(
        m.schema, m.dataset_id, m.created_at_utc, m.pipeline_config, m.writer_metadata, m.feature_schema,
        m.label_spec, m.transform_config, m.transform_diagnostics, m.exchange, m.symbol, m.storage_format,
        m.time_unit, m.decision_schedule, m.feature_columns, m.label_columns, m.required_columns, m.segments, splits, m.notes
    )
    mf.write_manifest_json(m2, root / mf.DEFAULT_MANIFEST_FILENAME)
    r = rd.open_dataset(str(root))
    assert len(r.split_entries("train")) == 1
    tr = r.read_split_table(SplitRole.TRAIN, columns=(mf.ROW_IDX_COLUMN, mf.LOCAL_TS_US_COLUMN))
    assert tr[mf.ROW_IDX_COLUMN].to_pylist() == [0, 1]
    va = r.read_split_table("val", columns=(mf.ROW_IDX_COLUMN,))
    assert va[mf.ROW_IDX_COLUMN].to_pylist() == [3, 4, 5]
    with pytest.raises(ValueError):
        r.read_split_table("test")


def test_iter_split_batches(tmp_path):
    root, m = make_dataset(tmp_path, rows=6, chunk_rows=3)
    splits = (mf.SplitMetadata(SplitRole.VAL, "seg_000001", 3, 6, m.segments[1].local_time_range),)
    m2 = mf.StorageManifest(
        m.schema, m.dataset_id, m.created_at_utc, m.pipeline_config, m.writer_metadata, m.feature_schema,
        m.label_spec, m.transform_config, m.transform_diagnostics, m.exchange, m.symbol, m.storage_format,
        m.time_unit, m.decision_schedule, m.feature_columns, m.label_columns, m.required_columns, m.segments, splits, m.notes
    )
    mf.write_manifest_json(m2, root / mf.DEFAULT_MANIFEST_FILENAME)
    r = rd.open_dataset(str(root))
    vals = []
    for b in r.iter_split_batches("val", columns=(mf.ROW_IDX_COLUMN,), batch_size=2):
        vals.extend(b.column(0).to_pylist())
    assert vals == [3, 4, 5]




def test_read_split_table_filters_with_internal_row_idx_projection(tmp_path):
    root, m = make_dataset(tmp_path, rows=6, chunk_rows=3)
    splits = (
        mf.SplitMetadata(
            SplitRole.TRAIN,
            "seg_000000",
            1,
            3,
            m.segments[0].local_time_range,
        ),
    )
    m2 = mf.StorageManifest(
        m.schema,
        m.dataset_id,
        m.created_at_utc,
        m.pipeline_config,
        m.writer_metadata,
        m.feature_schema,
        m.label_spec,
        m.transform_config,
        m.transform_diagnostics,
        m.exchange,
        m.symbol,
        m.storage_format,
        m.time_unit,
        m.decision_schedule,
        m.feature_columns,
        m.label_columns,
        m.required_columns,
        m.segments,
        splits,
        m.notes,
    )
    mf.write_manifest_json(m2, root / mf.DEFAULT_MANIFEST_FILENAME)

    r = rd.open_dataset(str(root))
    table = r.read_split_table("train", columns=(mf.LOCAL_TS_US_COLUMN,))
    assert table.column_names == [mf.LOCAL_TS_US_COLUMN]
    assert table.num_rows == 2
    assert mf.ROW_IDX_COLUMN not in table.column_names


def test_iter_split_batches_without_row_idx_projection(tmp_path):
    root, m = make_dataset(tmp_path, rows=6, chunk_rows=3)
    splits = (
        mf.SplitMetadata(
            SplitRole.TRAIN,
            "seg_000000",
            1,
            3,
            m.segments[0].local_time_range,
        ),
    )
    m2 = mf.StorageManifest(
        m.schema,
        m.dataset_id,
        m.created_at_utc,
        m.pipeline_config,
        m.writer_metadata,
        m.feature_schema,
        m.label_spec,
        m.transform_config,
        m.transform_diagnostics,
        m.exchange,
        m.symbol,
        m.storage_format,
        m.time_unit,
        m.decision_schedule,
        m.feature_columns,
        m.label_columns,
        m.required_columns,
        m.segments,
        splits,
        m.notes,
    )
    mf.write_manifest_json(m2, root / mf.DEFAULT_MANIFEST_FILENAME)

    r = rd.open_dataset(str(root))
    batches = list(r.iter_split_batches("train", columns=(mf.LOCAL_TS_US_COLUMN,), batch_size=1))
    assert all(b.schema.names == [mf.LOCAL_TS_US_COLUMN] for b in batches)
    assert [b.num_rows for b in batches] == [1, 1]

def test_tardis_ts_and_local_ts_are_preserved_and_not_collapsed(tmp_path):
    root, _ = make_dataset(tmp_path, rows=3, row_fn=lambda i: row(i, local_ts_us=2_000_000 + i, ts_us=1_000_000 + i))
    r = rd.open_dataset(str(root))
    t = r.read_table(columns=(mf.TS_US_COLUMN, mf.LOCAL_TS_US_COLUMN))
    assert t.column_names == [mf.TS_US_COLUMN, mf.LOCAL_TS_US_COLUMN]
    assert any(a != b for a, b in zip(t[mf.TS_US_COLUMN].to_pylist(), t[mf.LOCAL_TS_US_COLUMN].to_pylist()))


def test_no_future_leakage_or_repair_surface():
    src = inspect.getsource(rd)
    forbidden = (
        "future_" + "mid",
        "future_" + "ret",
        "fit_" + "transform",
        "Standard" + "Scaler",
        "P" + "CA",
        "GRACE_" + "MS",
        "drop_duplicate_" + "trades",
        "stride_" + "rows",
        "sort_" + "values",
        "re" + "pair",
        "de" + "dupe",
        "sh" + "uffle",
        "BY" + "BIT",
        "CM" + "SSL",
        "offline_" + "ingest",
        "Mini" + "Rocket",
        "Multi" + "Rocket",
        "Hy" + "dra",
        "Ae" + "on",
        "sklearn",
        "torch",
        "pandas",
        "polars",
    )
    assert all(term not in src for term in forbidden)


def test_iter_split_batches_does_not_materialize_split_table(tmp_path, monkeypatch):
    root, m = make_dataset(tmp_path, rows=6, chunk_rows=3)
    splits = (
        mf.SplitMetadata(SplitRole.TRAIN, "seg_000000", 1, 3, m.segments[0].local_time_range),
        mf.SplitMetadata(SplitRole.TRAIN, "seg_000001", 3, 5, m.segments[1].local_time_range),
    )
    m2 = mf.StorageManifest(
        m.schema, m.dataset_id, m.created_at_utc, m.pipeline_config, m.writer_metadata, m.feature_schema,
        m.label_spec, m.transform_config, m.transform_diagnostics, m.exchange, m.symbol, m.storage_format,
        m.time_unit, m.decision_schedule, m.feature_columns, m.label_columns, m.required_columns, m.segments, splits, m.notes
    )
    mf.write_manifest_json(m2, root / mf.DEFAULT_MANIFEST_FILENAME)
    r = rd.open_dataset(str(root))
    monkeypatch.setattr(r, "read_split_table", lambda *a, **k: (_ for _ in ()).throw(AssertionError("should not be called")))
    batches = list(r.iter_split_batches("train", columns=(mf.ROW_IDX_COLUMN,), batch_size=2))
    vals = [x for b in batches for x in b.column(0).to_pylist()]
    assert vals == [1, 2, 3, 4]


def test_iter_split_batches_does_not_call_read_segment_table(tmp_path, monkeypatch):
    root, m = make_dataset(tmp_path, rows=6, chunk_rows=3)
    splits = (mf.SplitMetadata(SplitRole.TRAIN, "seg_000000", 0, 3, m.segments[0].local_time_range),)
    m2 = mf.StorageManifest(
        m.schema, m.dataset_id, m.created_at_utc, m.pipeline_config, m.writer_metadata, m.feature_schema,
        m.label_spec, m.transform_config, m.transform_diagnostics, m.exchange, m.symbol, m.storage_format,
        m.time_unit, m.decision_schedule, m.feature_columns, m.label_columns, m.required_columns, m.segments, splits, m.notes
    )
    mf.write_manifest_json(m2, root / mf.DEFAULT_MANIFEST_FILENAME)
    r = rd.open_dataset(str(root))
    monkeypatch.setattr(r, "read_segment_table", lambda *a, **k: (_ for _ in ()).throw(AssertionError("should not be called")))
    vals = [x for b in r.iter_split_batches("train", columns=(mf.ROW_IDX_COLUMN,), batch_size=2) for x in b.column(0).to_pylist()]
    assert vals == [0, 1, 2]


def test_iter_split_batches_filters_by_row_idx_without_returning_internal_row_idx(tmp_path):
    root, m = make_dataset(tmp_path, rows=6, chunk_rows=3)
    splits = (mf.SplitMetadata(SplitRole.TRAIN, "seg_000000", 1, 3, m.segments[0].local_time_range),)
    m2 = mf.StorageManifest(
        m.schema, m.dataset_id, m.created_at_utc, m.pipeline_config, m.writer_metadata, m.feature_schema,
        m.label_spec, m.transform_config, m.transform_diagnostics, m.exchange, m.symbol, m.storage_format,
        m.time_unit, m.decision_schedule, m.feature_columns, m.label_columns, m.required_columns, m.segments, splits, m.notes
    )
    mf.write_manifest_json(m2, root / mf.DEFAULT_MANIFEST_FILENAME)
    r = rd.open_dataset(str(root))
    col = m.x_columns[0]
    batches = list(r.iter_split_batches("train", columns=(col,), batch_size=1))
    assert all(b.schema.names == [col] for b in batches)
    vals = [x for b in batches for x in b.column(0).to_pylist()]
    assert vals == [feature_values()[0], feature_values()[0]]


def test_iter_split_batches_returns_row_idx_when_requested(tmp_path):
    root, m = make_dataset(tmp_path, rows=6, chunk_rows=3)
    splits = (
        mf.SplitMetadata(SplitRole.TRAIN, "seg_000000", 1, 3, m.segments[0].local_time_range),
    )
    m2 = mf.StorageManifest(
        m.schema, m.dataset_id, m.created_at_utc, m.pipeline_config, m.writer_metadata, m.feature_schema,
        m.label_spec, m.transform_config, m.transform_diagnostics, m.exchange, m.symbol, m.storage_format,
        m.time_unit, m.decision_schedule, m.feature_columns, m.label_columns, m.required_columns, m.segments, splits, m.notes
    )
    mf.write_manifest_json(m2, root / mf.DEFAULT_MANIFEST_FILENAME)
    r = rd.open_dataset(str(root))
    vals = [x for b in r.iter_split_batches("train", columns=(mf.ROW_IDX_COLUMN,), batch_size=2) for x in b.column(0).to_pylist()]
    assert vals == [1, 2]


def test_iter_split_batches_preserves_requested_column_order(tmp_path):
    root, m = make_dataset(tmp_path, rows=6, chunk_rows=3)
    splits = (mf.SplitMetadata(SplitRole.TRAIN, "seg_000001", 3, 6, m.segments[1].local_time_range),)
    m2 = mf.StorageManifest(
        m.schema, m.dataset_id, m.created_at_utc, m.pipeline_config, m.writer_metadata, m.feature_schema,
        m.label_spec, m.transform_config, m.transform_diagnostics, m.exchange, m.symbol, m.storage_format,
        m.time_unit, m.decision_schedule, m.feature_columns, m.label_columns, m.required_columns, m.segments, splits, m.notes
    )
    mf.write_manifest_json(m2, root / mf.DEFAULT_MANIFEST_FILENAME)
    r = rd.open_dataset(str(root))
    cols = (m.x_columns[1], mf.ROW_IDX_COLUMN, m.x_columns[0])
    batches = list(r.iter_split_batches("train", columns=cols, batch_size=2))
    assert batches
    assert all(b.schema.names == list(cols) for b in batches)


def test_iter_split_batches_preserves_multiple_split_entry_order(tmp_path):
    root, m = make_dataset(tmp_path, rows=10, chunk_rows=5)
    splits = (
        mf.SplitMetadata(SplitRole.TRAIN, "seg_000000", 1, 3, m.segments[0].local_time_range),
        mf.SplitMetadata(SplitRole.TRAIN, "seg_000001", 7, 9, m.segments[1].local_time_range),
    )
    m2 = mf.StorageManifest(
        m.schema, m.dataset_id, m.created_at_utc, m.pipeline_config, m.writer_metadata, m.feature_schema,
        m.label_spec, m.transform_config, m.transform_diagnostics, m.exchange, m.symbol, m.storage_format,
        m.time_unit, m.decision_schedule, m.feature_columns, m.label_columns, m.required_columns, m.segments, splits, m.notes
    )
    mf.write_manifest_json(m2, root / mf.DEFAULT_MANIFEST_FILENAME)
    r = rd.open_dataset(str(root))
    vals = [x for b in r.iter_split_batches("train", columns=(mf.ROW_IDX_COLUMN,), batch_size=2) for x in b.column(0).to_pylist()]
    assert vals == [1, 2, 7, 8]


def test_iter_split_batches_honors_batch_size(tmp_path):
    root, m = make_dataset(tmp_path, rows=8, chunk_rows=4)
    splits = (mf.SplitMetadata(SplitRole.TRAIN, "seg_000001", 4, 8, m.segments[1].local_time_range),)
    m2 = mf.StorageManifest(
        m.schema, m.dataset_id, m.created_at_utc, m.pipeline_config, m.writer_metadata, m.feature_schema,
        m.label_spec, m.transform_config, m.transform_diagnostics, m.exchange, m.symbol, m.storage_format,
        m.time_unit, m.decision_schedule, m.feature_columns, m.label_columns, m.required_columns, m.segments, splits, m.notes
    )
    mf.write_manifest_json(m2, root / mf.DEFAULT_MANIFEST_FILENAME)
    r = rd.open_dataset(str(root))
    batches = list(r.iter_split_batches("train", columns=(mf.ROW_IDX_COLUMN,), batch_size=2))
    assert all(b.num_rows <= 2 for b in batches)
    vals = [x for b in batches for x in b.column(0).to_pylist()]
    expect = r.read_split_table("train", columns=(mf.ROW_IDX_COLUMN,))[mf.ROW_IDX_COLUMN].to_pylist()
    assert vals == expect


def test_iter_split_batches_source_is_streaming():
    src = inspect.getsource(rd.StorageDatasetReader.iter_split_batches)
    assert "read_split_table" not in src
    assert "read_segment_table" not in src
    assert "concat_tables" not in src
    assert "to_batches(max_chunksize" not in src
    assert ".scanner(" in src
    assert ".to_batches()" in src
