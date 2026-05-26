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
    code = r'''
import sys
before = set(sys.modules)
import mmrt.storage.writer  # noqa: F401
after = set(sys.modules) - before
forbidden = (
    "pan" + "das",
    "po" + "lars",
    "to" + "rch",
    "sk" + "learn",
    "mmrt.data.tardis_csv",
    "mmrt.data.event_merge",
    "mmrt.data.quality",
    "mmrt.features.engine",
    "mmrt.features.la" + "bels",
    "mmrt.features.trans" + "forms",
    "mmrt.storage.reader",
    "mmrt.storage.splits",
    "mmrt.linear",
    "CM" + "SSL17",
    "offline_" + "ingest",
)
bad = sorted(name for name in forbidden if name in after)
if bad:
    raise SystemExit(repr(bad))
print("ok")
'''
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
    assert [(s.first_row_idx, s.last_row_idx) for s in m.segments] == [(0, 1), (2, 3), (4, 4)]
    last = m.segments[-1]
    assert last.time_range.end_us == last.time_range.start_us + 1
    assert last.local_time_range.end_us == last.local_time_range.start_us + 1


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


def test_row_validation_rejects_invalid_values(tmp_path):
    def expect_bad_row(case_name, bad_row):
        w = wr.DecisionRowWriter(
            wr.WriterConfig(dataset_id="d", created_at_utc="2026", dataset_root=str(tmp_path / f"d_{case_name}"))
        )
        with pytest.raises(ValueError):
            w.append(bad_row)

    base = wr.DecisionRow(0, 1, 1, 0, 100.0, 2, label_values(), feature_values())
    expect_bad_row("label_len", wr.DecisionRow(0, 1, 1, 0, 100.0, 2, (1.0,), feature_values()))
    expect_bad_row("feature_len", wr.DecisionRow(0, 1, 1, 0, 100.0, 2, label_values(), (1.0,)))
    expect_bad_row("label_nan", wr.DecisionRow(base.decision_index, base.ts_us, base.local_ts_us, base.event_seq, base.raw_mid, base.label_entry_ts_us, (float("nan"), -2.0, 0.5), base.feature_values))
    expect_bad_row("label_inf", wr.DecisionRow(base.decision_index, base.ts_us, base.local_ts_us, base.event_seq, base.raw_mid, base.label_entry_ts_us, (float("inf"), -2.0, 0.5), base.feature_values))
    bad_feats_nan = list(base.feature_values)
    bad_feats_nan[0] = float("nan")
    expect_bad_row("feat_nan", wr.DecisionRow(base.decision_index, base.ts_us, base.local_ts_us, base.event_seq, base.raw_mid, base.label_entry_ts_us, base.label_values, tuple(bad_feats_nan)))
    bad_feats_inf = list(base.feature_values)
    bad_feats_inf[0] = float("inf")
    expect_bad_row("feat_inf", wr.DecisionRow(base.decision_index, base.ts_us, base.local_ts_us, base.event_seq, base.raw_mid, base.label_entry_ts_us, base.label_values, tuple(bad_feats_inf)))
    expect_bad_row("raw_mid_zero", wr.DecisionRow(0, 1, 1, 0, 0.0, 2, label_values(), feature_values()))
    expect_bad_row("raw_mid_neg", wr.DecisionRow(0, 1, 1, 0, -1.0, 2, label_values(), feature_values()))
    expect_bad_row("raw_mid_nan", wr.DecisionRow(0, 1, 1, 0, float("nan"), 2, label_values(), feature_values()))
    expect_bad_row("entry_before_local", wr.DecisionRow(0, 1, 2, 0, 100.0, 1, label_values(), feature_values()))
    expect_bad_row("ts_nonpos", wr.DecisionRow(0, 0, 1, 0, 100.0, 2, label_values(), feature_values()))
    expect_bad_row("local_nonpos", wr.DecisionRow(0, 1, 0, 0, 100.0, 2, label_values(), feature_values()))
    expect_bad_row("event_seq_lt_neg1", wr.DecisionRow(0, 1, 1, -2, 100.0, 2, label_values(), feature_values()))
    expect_bad_row("decision_index_neg", wr.DecisionRow(-1, 1, 1, 0, 100.0, 2, label_values(), feature_values()))
    expect_bad_row("decision_index_bool", wr.DecisionRow(True, 1, 1, 0, 100.0, 2, label_values(), feature_values()))
    expect_bad_row("ts_bool", wr.DecisionRow(0, True, 1, 0, 100.0, 2, label_values(), feature_values()))
    expect_bad_row("local_bool", wr.DecisionRow(0, 1, True, 0, 100.0, 2, label_values(), feature_values()))
    expect_bad_row("event_seq_bool", wr.DecisionRow(0, 1, 1, False, 100.0, 2, label_values(), feature_values()))
    expect_bad_row("entry_bool", wr.DecisionRow(0, 1, 1, 0, 100.0, True, label_values(), feature_values()))
    expect_bad_row("raw_mid_bool", wr.DecisionRow(0, 1, 1, 0, True, 2, label_values(), feature_values()))
    expect_bad_row("label_bool", wr.DecisionRow(0, 1, 1, 0, 100.0, 2, (True, -2.0, 0.5), feature_values()))
    bad_feats_bool = list(feature_values())
    bad_feats_bool[0] = True
    expect_bad_row("feature_bool", wr.DecisionRow(0, 1, 1, 0, 100.0, 2, label_values(), tuple(bad_feats_bool)))


def test_monotonicity_validation(tmp_path):
    w = wr.DecisionRowWriter(wr.WriterConfig(dataset_id="d", created_at_utc="2026", dataset_root=str(tmp_path / "d"), chunk_rows=10))
    w.append(row(0, local_ts_us=1_000_000, ts_us=1_000_000))
    with pytest.raises(ValueError):
        w.append(row(0, local_ts_us=1_500_000, ts_us=1_500_000))
    with pytest.raises(ValueError):
        w.append(row(2, local_ts_us=999_999, ts_us=2_000_000))
    w.append(row(3, local_ts_us=1_000_000, ts_us=2_500_000))
    m = w.finalize()
    table = pq.read_table(tmp_path / "d" / m.segments[-1].parquet_path)
    assert table.column("local_ts_us").to_pylist() == [1_000_000, 1_000_000]
    assert table.column("decision_index").to_pylist() == [0, 3]


def test_writer_single_row_segment_uses_half_open_time_ranges(tmp_path):
    cfg = wr.WriterConfig(dataset_id="d", created_at_utc="2026", dataset_root=str(tmp_path / "d"), chunk_rows=1)
    w = wr.DecisionRowWriter(cfg)
    w.append(row(0, ts_us=1_500_000, local_ts_us=2_000_000))
    m = w.finalize()
    assert len(m.segments) == 1
    seg = m.segments[0]
    assert seg.time_range.start_us == 1_500_000
    assert seg.time_range.end_us == 1_500_001
    assert seg.local_time_range.start_us == 2_000_000
    assert seg.local_time_range.end_us == 2_000_001
    table = pq.read_table(tmp_path / "d" / seg.parquet_path)
    assert table.column("ts_us").to_pylist() == [1_500_000]
    assert table.column("local_ts_us").to_pylist() == [2_000_000]


def test_tardis_timestamp_fields_are_preserved_separately(tmp_path):
    w = wr.DecisionRowWriter(wr.WriterConfig(dataset_id="d", created_at_utc="2026", dataset_root=str(tmp_path / "d")))
    w.append(row(0, local_ts_us=2_000_000, ts_us=1_500_000))
    m = w.finalize()
    seg = m.segments[0]
    assert seg.time_range.start_us == 1_500_000
    assert seg.time_range.end_us == 1_500_001
    assert seg.local_time_range.start_us == 2_000_000
    assert seg.local_time_range.end_us == 2_000_001
    assert seg.time_range != seg.local_time_range
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


def test_transform_metadata_non_json_safe_rejected_on_finalize(tmp_path):
    w = wr.DecisionRowWriter(
        wr.WriterConfig(
            dataset_id="d",
            created_at_utc="2026",
            dataset_root=str(tmp_path / "d"),
            transform_config={"bad": float("nan")},
        )
    )
    w.append(row(0))
    with pytest.raises(ValueError):
        w.finalize()
    assert not (tmp_path / "d" / "manifest.json").exists()

    w2 = wr.DecisionRowWriter(
        wr.WriterConfig(
            dataset_id="d2",
            created_at_utc="2026",
            dataset_root=str(tmp_path / "d2"),
            transform_diagnostics={1: "bad"},
        )
    )
    w2.append(row(0))
    with pytest.raises(ValueError):
        w2.finalize()
    assert not (tmp_path / "d2" / "manifest.json").exists()


def test_append_does_not_mutate_decision_row(tmp_path):
    from mmrt.features import specs

    r = wr.DecisionRow(
        decision_index=0,
        ts_us=1,
        local_ts_us=1,
        event_seq=0,
        raw_mid=100,
        label_entry_ts_us=2,
        label_values=(1, 2, 3),
        feature_values=tuple(range(specs.FEATURE_COUNT)),
    )
    before = r
    w = wr.DecisionRowWriter(wr.WriterConfig(dataset_id="d", created_at_utc="2026", dataset_root=str(tmp_path / "d")))
    w.append(r)
    assert r == before
    assert isinstance(r.raw_mid, int)


def test_no_future_leakage_or_row_repair_surface():
    src = inspect.getsource(wr)
    banned = ["future_" + "mid", "future_" + "ret", "fit_" + "transform", "Standard" + "Scaler", "P" + "CA", "GRACE_" + "MS", "drop_duplicate_" + "trades", "stride_" + "rows", "sort_" + "values", "repair", "de" + "dupe"]
    assert all(x not in src for x in banned)
