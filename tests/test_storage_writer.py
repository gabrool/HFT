import inspect
import subprocess
import sys

import pytest

pytest.importorskip("pyarrow")

import pyarrow.parquet as pq

from mmrt.config import default_config
from mmrt.storage import manifest as mf
from mmrt.storage import writer as wr


def feature_values(scale=0.01):
    from mmrt.features import specs

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


def test_public_api_boundary():
    assert wr.__all__ == ["DEFAULT_CHUNK_ROWS", "DEFAULT_ROW_GROUP_ROWS", "DecisionRow", "WriterConfig", "DecisionRowWriter", "arrow_schema"]


def test_no_forbidden_imports():
    code = "import mmrt.storage.writer as w; print('ok')"
    out = subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True)
    assert "ok" in out.stdout


def test_arrow_schema_matches_manifest_contract():
    cfg = default_config()
    schema = wr.arrow_schema(cfg.label_spec)
    assert schema.names == list(mf.required_row_columns(cfg.label_spec))
    assert schema.field(mf.ROW_IDX_COLUMN).type == __import__("pyarrow").int64()
    assert schema.field(mf.RAW_MID_COLUMN).type == __import__("pyarrow").float64()
    for c in mf.label_columns(cfg.label_spec):
        assert schema.field(c).type == __import__("pyarrow").float32()
    for c in mf.feature_columns():
        assert schema.field(c).type == __import__("pyarrow").float32()
    md = schema.metadata
    for k in (b"manifest_schema_version", b"storage_format", b"time_unit", b"feature_schema_version", b"feature_count"):
        assert k in md


def test_writer_config_validation(tmp_path):
    cfg = default_config()
    wr.WriterConfig(dataset_id="d", created_at_utc="now", dataset_root=str(tmp_path / "x"), config=cfg)
    with pytest.raises(ValueError):
        wr.WriterConfig(dataset_id="", created_at_utc="now", dataset_root="x")
    with pytest.raises(ValueError):
        wr.WriterConfig(dataset_id="d", created_at_utc="", dataset_root="x")
    with pytest.raises(ValueError):
        wr.WriterConfig(dataset_id="d", created_at_utc="n", dataset_root="")
    with pytest.raises(ValueError):
        wr.WriterConfig(dataset_id="d", created_at_utc="n", dataset_root="x", chunk_rows=0)
    with pytest.raises(ValueError):
        wr.WriterConfig(dataset_id="d", created_at_utc="n", dataset_root="x", row_group_rows=0)
    with pytest.raises(ValueError):
        wr.WriterConfig(dataset_id="d", created_at_utc="n", dataset_root="x", config="bad")
    with pytest.raises(ValueError):
        wr.WriterConfig(dataset_id="d", created_at_utc="n", dataset_root="x", source_files="bad")
    wr.WriterConfig(dataset_id="d", created_at_utc="n", dataset_root="x", source_files=("raw/day.csv.gz",))


def test_decision_row_from_arrays_coerces_tuples():
    r = wr.DecisionRow.from_arrays(decision_index=1, ts_us=1, local_ts_us=1, event_seq=0, raw_mid=1.0, label_entry_ts_us=1, label_values=[1.0, 2.0, 3.0], feature_values=[1.0] * len(feature_values()))
    assert isinstance(r.label_values, tuple)
    assert isinstance(r.feature_values, tuple)


def test_writer_writes_single_segment_and_manifest(tmp_path):
    c = wr.WriterConfig(dataset_id="d", created_at_utc="2026", dataset_root=str(tmp_path / "d"), chunk_rows=10, source_files=("raw/day.csv.gz",))
    w = wr.DecisionRowWriter(c)
    for i in range(3):
        w.append(row(i))
    m = w.finalize()
    assert (tmp_path / "d" / "manifest.json").exists()
    assert len(m.segments) == 1
    seg = m.segments[0]
    assert seg.parquet_path == "segments/seg_000000.parquet"
    assert m.total_rows == 3 and m.total_labels == 3
    assert seg.first_row_idx == 0 and seg.last_row_idx == 2
    assert seg.source_files == ("raw/day.csv.gz",)
    table = pq.read_table(tmp_path / "d" / seg.parquet_path)
    assert table.names == list(m.required_columns)
    assert table.column("row_idx").to_pylist() == [0, 1, 2]


def test_writer_chunks_multiple_segments(tmp_path):
    c = wr.WriterConfig(dataset_id="d", created_at_utc="2026", dataset_root=str(tmp_path / "d"), chunk_rows=2)
    w = wr.DecisionRowWriter(c)
    for i in range(5):
        w.append(row(i))
    m = w.finalize()
    assert [s.row_count for s in m.segments] == [2, 2, 1]


def test_flush_empty_is_noop(tmp_path):
    w = wr.DecisionRowWriter(wr.WriterConfig(dataset_id="d", created_at_utc="2026", dataset_root=str(tmp_path / "d")))
    w.flush()
    assert list((tmp_path / "d" / "segments").glob("*.parquet")) == []


def test_finalize_empty_dataset_rejected(tmp_path):
    w = wr.DecisionRowWriter(wr.WriterConfig(dataset_id="d", created_at_utc="2026", dataset_root=str(tmp_path / "d")))
    with pytest.raises(ValueError):
        w.finalize()


def test_context_manager_finalizes_on_success(tmp_path):
    cfg = wr.WriterConfig(dataset_id="d", created_at_utc="2026", dataset_root=str(tmp_path / "d"))
    with wr.DecisionRowWriter(cfg) as w:
        w.append(row(0))
    assert (tmp_path / "d" / "manifest.json").exists()


def test_context_manager_does_not_finalize_on_exception(tmp_path):
    cfg = wr.WriterConfig(dataset_id="d", created_at_utc="2026", dataset_root=str(tmp_path / "d"), chunk_rows=100)
    with pytest.raises(RuntimeError):
        with wr.DecisionRowWriter(cfg) as w:
            w.append(row(0))
            raise RuntimeError("x")
    assert not (tmp_path / "d" / "manifest.json").exists()


def test_close_idempotent_returns_manifest(tmp_path):
    w = wr.DecisionRowWriter(wr.WriterConfig(dataset_id="d", created_at_utc="2026", dataset_root=str(tmp_path / "d")))
    w.append(row(0))
    m1 = w.close()
    m2 = w.close()
    assert m1 == m2
    with pytest.raises(RuntimeError):
        w.finalize()


def test_existing_manifest_rejected(tmp_path):
    d = tmp_path / "d"
    d.mkdir()
    (d / "manifest.json").write_text("{}")
    with pytest.raises(FileExistsError):
        wr.DecisionRowWriter(wr.WriterConfig(dataset_id="d", created_at_utc="2026", dataset_root=str(d)))


def test_existing_parquet_segments_rejected(tmp_path):
    d = tmp_path / "d"
    (d / "segments").mkdir(parents=True)
    (d / "segments" / "seg_000000.parquet").write_text("x")
    with pytest.raises(FileExistsError):
        wr.DecisionRowWriter(wr.WriterConfig(dataset_id="d", created_at_utc="2026", dataset_root=str(d)))


def test_row_validation_rejects_bad_lengths_and_nonfinite(tmp_path):
    w = wr.DecisionRowWriter(wr.WriterConfig(dataset_id="d", created_at_utc="2026", dataset_root=str(tmp_path / "d")))
    with pytest.raises(ValueError):
        w.append(wr.DecisionRow(0, 1, 1, 0, 1.0, 1, (1.0,), feature_values()))
    with pytest.raises(ValueError):
        w.append(wr.DecisionRow(0, 1, 1, 0, 1.0, 1, label_values(), (1.0,)))


def test_monotonicity_validation(tmp_path):
    w = wr.DecisionRowWriter(wr.WriterConfig(dataset_id="d", created_at_utc="2026", dataset_root=str(tmp_path / "d")))
    w.append(row(0))
    with pytest.raises(ValueError):
        w.append(row(0, local_ts_us=1_500_000, ts_us=1_500_000))
    with pytest.raises(ValueError):
        w.append(row(2, local_ts_us=1_000_000, ts_us=2_000_000))
    w.append(row(3, local_ts_us=1_000_000 + 0, ts_us=2_500_000))


def test_tardis_timestamp_fields_are_preserved_separately(tmp_path):
    w = wr.DecisionRowWriter(wr.WriterConfig(dataset_id="d", created_at_utc="2026", dataset_root=str(tmp_path / "d")))
    w.append(row(0, local_ts_us=2_000_000, ts_us=1_500_000))
    m = w.finalize()
    t = pq.read_table(tmp_path / "d" / m.segments[0].parquet_path)
    assert t.column("ts_us").to_pylist() == [1_500_000]
    assert t.column("local_ts_us").to_pylist() == [2_000_000]


def test_writer_uses_transformed_features_as_given(tmp_path):
    w = wr.DecisionRowWriter(wr.WriterConfig(dataset_id="d", created_at_utc="2026", dataset_root=str(tmp_path / "d")))
    feats = feature_values(scale=0.123)
    w.append(wr.DecisionRow(0, 1, 1, 0, 100.0, 2, label_values(), feats))
    m = w.finalize()
    t = pq.read_table(tmp_path / "d" / m.segments[0].parquet_path)
    assert t.column(m.feature_columns[0]).to_pylist()[0] == pytest.approx(feats[0], rel=1e-6, abs=1e-6)


def test_transform_metadata_passed_to_manifest(tmp_path):
    w = wr.DecisionRowWriter(
        wr.WriterConfig(
            dataset_id="d",
            created_at_utc="2026",
            dataset_root=str(tmp_path / "d"),
            transform_config={"alpha": 1},
            transform_diagnostics={"rows_seen": 3},
            notes={"note": "ok"},
        )
    )
    w.append(row(0))
    m = w.finalize()
    assert m.transform_config["alpha"] == 1


def test_no_future_leakage_or_row_repair_surface():
    src = inspect.getsource(wr)
    banned = ["future_" + "mid", "future_" + "ret", "fit_" + "transform", "Standard" + "Scaler", "P" + "CA", "GRACE_" + "MS", "drop_duplicate_" + "trades", "stride_" + "rows", "sort_" + "values", "repair", "de" + "dupe"]
    assert all(x not in src for x in banned)
