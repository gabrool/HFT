import json
import subprocess
import sys

import pytest

import mmrt.storage.manifest as mf
from mmrt.config import default_config
from mmrt.contracts import LabelSpec, SplitRole, StorageFormat, TimeRangeUS, TimeUnit
from mmrt.features import specs


def segment(key="seg_000", start=0, rows=10, local_start=1_000_000):
    return mf.StorageSegment(
        segment_key=key,
        parquet_path=f"segments/{key}.parquet",
        row_count=rows,
        label_count=rows,
        time_range=TimeRangeUS(local_start, local_start + rows * 500_000),
        local_time_range=TimeRangeUS(local_start, local_start + rows * 500_000),
        first_row_idx=start,
        last_row_idx=start + rows - 1,
        source_files=(f"raw/{key}.csv.gz",),
    )


def manifest_one_segment():
    return mf.make_manifest(dataset_id="ds1", created_at_utc="2026-05-26T00:00:00Z", segments=(segment(),))


def manifest_from_base_with_columns(base, *, feature_columns=None, label_columns=None, required_columns=None):
    return mf.StorageManifest(
        schema=base["schema"],
        dataset_id=base["dataset_id"],
        created_at_utc=base["created_at_utc"],
        pipeline_config=base["pipeline_config"],
        writer_metadata=base["writer_metadata"],
        feature_schema=base["feature_schema"],
        label_spec=mf.label_spec_from_dict(base["label_spec"]),
        transform_config=base["transform_config"],
        transform_diagnostics=base["transform_diagnostics"],
        exchange=base["exchange"],
        symbol=base["symbol"],
        storage_format=base["storage_format"],
        time_unit=base["time_unit"],
        decision_stride_us=base["decision_stride_us"],
        feature_columns=list(base["feature_columns"]) if feature_columns is None else feature_columns,
        label_columns=list(base["label_columns"]) if label_columns is None else label_columns,
        required_columns=list(base["required_columns"]) if required_columns is None else required_columns,
        segments=tuple(mf.StorageSegment.from_dict(s) for s in base["segments"]),
        splits=tuple(mf.SplitMetadata.from_dict(sp) for sp in base.get("splits", [])),
        notes=base.get("notes"),
    )


def test_public_api_boundary():
    expected = {
        "STORAGE_MANIFEST_SCHEMA", "DEFAULT_MANIFEST_FILENAME", "ROW_IDX_COLUMN", "DECISION_INDEX_COLUMN", "TS_US_COLUMN",
        "LOCAL_TS_US_COLUMN", "EVENT_SEQ_COLUMN", "RAW_MID_COLUMN", "LABEL_ENTRY_TS_US_COLUMN", "FEATURE_COLUMN_PREFIX",
        "LABEL_COLUMN_PREFIX", "BASE_ROW_COLUMNS", "DEFAULT_COMPRESSION", "DEFAULT_PARQUET_VERSION", "StorageSegment",
        "SplitMetadata", "StorageManifest", "feature_columns", "label_columns", "required_row_columns", "feature_schema_record",
        "default_writer_metadata", "label_spec_to_dict", "label_spec_from_dict", "time_range_to_dict", "time_range_from_dict",
        "pipeline_config_to_manifest_dict", "make_manifest", "manifest_sha256", "manifest_to_json_bytes", "manifest_from_json_bytes",
        "write_manifest_json", "read_manifest_json",
    }
    assert set(mf.__all__) == expected
    forbidden = ["by" + "bit", "cm" + "ssl", "aux", "p" + "ca", "target", "future", "to" + "rch", "pan" + "das", "po" + "lars", "py" + "arrow"]
    for name in mf.__all__:
        lower = name.lower()
        for token in forbidden:
            assert token not in lower


def test_no_forbidden_imports():
    code = "import sys; before=set(sys.modules); import mmrt.storage.manifest; after=set(sys.modules); print('\\n'.join(sorted(after-before)))"
    out = subprocess.check_output([sys.executable, "-c", code], text=True)
    delta = set(out.splitlines())
    forbidden = {
        "pan" + "das", "po" + "lars", "to" + "rch", "py" + "arrow", "mmrt.features.engine", "mmrt.features.la" + "bels",
        "mmrt.data.tardis_csv", "mmrt.data.event_merge", "mmrt.data.quality", "mmrt.storage.writer", "mmrt.storage.reader",
        "mmrt.storage.splits", "mmrt.linear", "CM" + "SSL17", "offline_" + "ingest",
    }
    for mod in delta:
        for token in forbidden:
            assert token not in mod


def test_feature_and_label_columns_match_specs():
    cols = mf.feature_columns()
    assert len(cols) == specs.FEATURE_COUNT
    assert cols[0] == "x_" + specs.FEATURE_NAMES[0]
    assert cols[-1] == "x_" + specs.FEATURE_NAMES[-1]
    assert len(set(cols)) == len(cols)
    ls = LabelSpec((200_000, 500_000, 1_000_000), 1_000)
    y = mf.label_columns(ls)
    assert y == ("y_ret_bps_200000us", "y_ret_bps_500000us", "y_ret_bps_1000000us")
    req = mf.required_row_columns(ls)
    assert req[: len(mf.BASE_ROW_COLUMNS)] == mf.BASE_ROW_COLUMNS
    assert req == mf.BASE_ROW_COLUMNS + y + cols
    assert mf.TS_US_COLUMN in req and mf.LOCAL_TS_US_COLUMN in req


def test_feature_schema_record_matches_specs():
    got = mf.feature_schema_record()
    want = specs.schema_record()
    for k in ("schema", "feature_count", "feature_names_hash", "feature_specs_hash", "feature_dtype", "time_unit"):
        assert got[k] == want[k]


def test_storage_segment_validation_and_roundtrip():
    s = segment()
    assert mf.StorageSegment.from_dict(s.to_dict()) == s
    assert s.start_local_us == s.local_time_range.start_us
    assert s.end_local_us == s.local_time_range.end_us
    with pytest.raises(ValueError): mf.StorageSegment("", "a.parquet", 1, 0, TimeRangeUS(1, 2), TimeRangeUS(1, 2), 0, 0)
    with pytest.raises(ValueError): mf.StorageSegment("a", "/a.parquet", 1, 0, TimeRangeUS(1, 2), TimeRangeUS(1, 2), 0, 0)
    with pytest.raises(ValueError): mf.StorageSegment("a", "a/../b.parquet", 1, 0, TimeRangeUS(1, 2), TimeRangeUS(1, 2), 0, 0)
    with pytest.raises(ValueError): mf.StorageSegment("a", "a\\b.parquet", 1, 0, TimeRangeUS(1, 2), TimeRangeUS(1, 2), 0, 0)
    with pytest.raises(ValueError): mf.StorageSegment("a", "a//b.parquet", 1, 0, TimeRangeUS(1, 2), TimeRangeUS(1, 2), 0, 0)
    with pytest.raises(ValueError): mf.StorageSegment("a", "a.parquet", 0, 0, TimeRangeUS(1, 2), TimeRangeUS(1, 2), 0, 0)
    with pytest.raises(ValueError): mf.StorageSegment("a", "a.parquet", 1, 2, TimeRangeUS(1, 2), TimeRangeUS(1, 2), 0, 0)
    with pytest.raises(ValueError): mf.StorageSegment("a", "a.parquet", 1, 1, "bad", TimeRangeUS(1, 2), 0, 0)
    with pytest.raises(ValueError): mf.StorageSegment("a", "a.parquet", 1, 1, TimeRangeUS(1, 2), "bad", 0, 0)
    with pytest.raises(ValueError): mf.StorageSegment("a", "a.parquet", 3, 1, TimeRangeUS(1, 2), TimeRangeUS(1, 2), 0, 1)
    with pytest.raises(ValueError): mf.StorageSegment("a", "a.parquet", 1, 1, TimeRangeUS(1, 2), TimeRangeUS(1, 2), 0, 0, ("../bad",))
    with pytest.raises(ValueError): mf.StorageSegment("a", "a.parquet", 1, 1, TimeRangeUS(1, 2), TimeRangeUS(1, 2), 0, 0, "raw/a.csv.gz")
    with pytest.raises(ValueError): mf.StorageSegment("a", "a.parquet", 1, 1, TimeRangeUS(1, 2), TimeRangeUS(1, 2), 0, 0, b"raw/a.csv.gz")


def test_split_metadata_validation_and_roundtrip():
    sp = mf.SplitMetadata(SplitRole.TRAIN, "seg_000", 0, 5, TimeRangeUS(1, 2))
    assert mf.SplitMetadata.from_dict(sp.to_dict()) == sp
    assert mf.SplitMetadata("train", "seg_000", 0, 5, TimeRangeUS(1, 2)).role == SplitRole.TRAIN
    with pytest.raises(ValueError): mf.SplitMetadata(SplitRole.TRAIN, "", 0, 5, TimeRangeUS(1, 2))
    with pytest.raises(ValueError): mf.SplitMetadata(SplitRole.TRAIN, "seg", 5, 5, TimeRangeUS(1, 2))
    with pytest.raises(ValueError): mf.SplitMetadata(SplitRole.TRAIN, "seg", 5, 4, TimeRangeUS(1, 2))
    with pytest.raises(ValueError): mf.SplitMetadata(SplitRole.TRAIN, "seg", 0, 1, TimeRangeUS(1, 2), -1, 0)
    with pytest.raises(ValueError): mf.SplitMetadata(SplitRole.TRAIN, "seg", 0, 1, TimeRangeUS(1, 2), 0, -1)
    with pytest.raises(ValueError): mf.SplitMetadata(SplitRole.TRAIN, "seg", 0, 1, "bad")


def test_label_spec_serialization_roundtrip_and_missing_keys():
    ls = LabelSpec((200_000, 500_000), 1_000)
    d = mf.label_spec_to_dict(ls)
    assert "label_context_us" in d
    assert mf.label_spec_from_dict(d) == ls
    for key in ("horizons_us", "entry_delay_us", "price_reference", "asof_policy"):
        dd = dict(d)
        del dd[key]
        with pytest.raises(ValueError):
            mf.label_spec_from_dict(dd)
    with pytest.raises(ValueError): mf.label_spec_from_dict({"horizons_us": [], "entry_delay_us": 1_000, "price_reference": "mid", "asof_policy": "strict_prev"})
    with pytest.raises(ValueError): mf.label_spec_from_dict({"horizons_us": [1000], "entry_delay_us": 1_000, "price_reference": "bad", "asof_policy": "strict_prev"})


def test_time_range_serialization_roundtrip_and_missing_keys():
    tr = TimeRangeUS(1, 2)
    d = mf.time_range_to_dict(tr)
    assert mf.time_range_from_dict(d) == tr
    with pytest.raises(ValueError): mf.time_range_from_dict({"end_us": 2})
    with pytest.raises(ValueError): mf.time_range_from_dict({"start_us": 1})
    with pytest.raises(ValueError): mf.time_range_to_dict("bad")


def test_make_manifest_defaults_and_validation():
    cfg = default_config()
    m = manifest_one_segment()
    assert m.schema == mf.STORAGE_MANIFEST_SCHEMA
    assert m.storage_format == StorageFormat.FLAT_DECISION_ROWS_US
    assert m.time_unit == TimeUnit.MICROSECOND
    assert m.decision_stride_us == 500_000
    assert m.exchange == cfg.market.exchange and m.symbol == cfg.market.symbol
    assert m.feature_schema == mf.feature_schema_record()
    assert m.feature_columns == mf.feature_columns()
    assert m.label_columns == mf.label_columns(cfg.label_spec)
    assert m.required_columns == mf.required_row_columns(cfg.label_spec)
    assert m.total_rows == 10 and m.total_labels == 10
    assert m.segment_keys == ("seg_000",)
    assert m.parquet_paths == ("segments/seg_000.parquet",)
    assert m.x_columns == m.feature_columns and m.y_columns == m.label_columns
    h = m.content_hash()
    assert len(h) == 64 and all(c in "0123456789abcdef" for c in h)


def test_manifest_constructor_coerces_column_lists_to_tuples():
    base = manifest_one_segment().to_dict()
    m = manifest_from_base_with_columns(base)

    assert isinstance(m.feature_columns, tuple)
    assert isinstance(m.label_columns, tuple)
    assert isinstance(m.required_columns, tuple)
    assert isinstance(m.x_columns, tuple)
    assert isinstance(m.y_columns, tuple)
    assert m.feature_columns == mf.feature_columns()
    assert m.label_columns == mf.label_columns(m.label_spec)
    assert m.required_columns == mf.required_row_columns(m.label_spec)


def test_manifest_constructor_rejects_wrong_column_lists():
    base = manifest_one_segment().to_dict()
    bad = list(base["feature_columns"])
    bad = bad[:-1]

    with pytest.raises(ValueError):
        manifest_from_base_with_columns(base, feature_columns=bad)


def test_manifest_to_dict_from_dict_and_canonical_json():
    m = manifest_one_segment()
    d = m.to_dict()
    assert mf.StorageManifest.from_dict(d) == m
    b1 = mf.manifest_to_json_bytes(m)
    b2 = mf.manifest_to_json_bytes(m)
    assert b1.endswith(b"\n")
    assert b1 == b2
    assert isinstance(json.loads(b1), dict)
    assert mf.manifest_from_json_bytes(b1) == m
    assert mf.manifest_from_json_bytes(b1.decode("utf-8")) == m
    assert mf.manifest_sha256(d) == mf.manifest_sha256(d)


def test_manifest_json_file_io_atomic(tmp_path):
    m = manifest_one_segment()
    p = tmp_path / "nested" / "manifest.json"
    mf.write_manifest_json(m, p)
    assert p.exists()
    assert not (tmp_path / "nested" / "manifest.json.tmp").exists()
    assert mf.read_manifest_json(p) == m


def test_manifest_rejects_feature_schema_drift():
    base = manifest_one_segment().to_dict()
    for k, v in {
        "feature_count": specs.FEATURE_COUNT + 1,
        "feature_names_hash": "bad",
        "feature_specs_hash": "bad",
        "schema": "bad",
        "feature_dtype": "float64",
        "time_unit": "ms",
    }.items():
        d = json.loads(json.dumps(base))
        d["feature_schema"][k] = v
        with pytest.raises(ValueError): mf.StorageManifest.from_dict(d)
    d = json.loads(json.dumps(base)); d["feature_columns"] = d["feature_columns"][:-1]
    with pytest.raises(ValueError): mf.StorageManifest.from_dict(d)
    d = json.loads(json.dumps(base)); d["label_columns"] = list(reversed(d["label_columns"]))
    with pytest.raises(ValueError): mf.StorageManifest.from_dict(d)
    d = json.loads(json.dumps(base)); d["required_columns"] = d["required_columns"][:-1]
    with pytest.raises(ValueError): mf.StorageManifest.from_dict(d)


def test_validate_against_current_code_catches_post_construction_mutation():
    m = manifest_one_segment()
    m.validate_against_current_code()
    for k, v in {
        "feature_count": specs.FEATURE_COUNT + 1,
        "feature_names_hash": "bad",
        "feature_specs_hash": "bad",
        "schema": "bad",
        "feature_dtype": "float64",
        "time_unit": "ms",
    }.items():
        mm = manifest_one_segment()
        mm.feature_schema[k] = v
        with pytest.raises(ValueError): mm.validate_against_current_code()


def test_manifest_rejects_pipeline_config_inconsistency():
    base = manifest_one_segment().to_dict()
    for k, v in {
        "feature_schema": "bad",
        "time_unit": "ms",
        "storage_format": "bad",
        "decision_stride_us": 123,
        "exchange": "wrong",
        "symbol": "WRONG",
    }.items():
        d = json.loads(json.dumps(base))
        d["pipeline_config"][k] = v
        with pytest.raises(ValueError): mf.StorageManifest.from_dict(d)
    d = json.loads(json.dumps(base))
    for k in ("feature_schema", "time_unit", "storage_format", "decision_stride_us", "exchange", "symbol"):
        del d["pipeline_config"][k]
    mf.StorageManifest.from_dict(d)


def test_manifest_rejects_segment_inconsistencies():
    s0 = segment("seg_000", 0, 10, 1_000_000)
    s1 = segment("seg_001", 10, 10, 2_000_000)
    m = manifest_one_segment().to_dict()
    d = dict(m); d["segments"] = [s0.to_dict(), s0.to_dict()]
    with pytest.raises(ValueError): mf.StorageManifest.from_dict(d)
    d = dict(m); a = s0.to_dict(); b = s1.to_dict(); b["parquet_path"] = a["parquet_path"]; d["segments"] = [a, b]
    with pytest.raises(ValueError): mf.StorageManifest.from_dict(d)
    d = dict(m); a = s0.to_dict(); a["first_row_idx"] = 1; d["segments"] = [a]
    with pytest.raises(ValueError): mf.StorageManifest.from_dict(d)
    d = dict(m); a = s0.to_dict(); b = s1.to_dict(); b["first_row_idx"] = 11; d["segments"] = [a, b]
    with pytest.raises(ValueError): mf.StorageManifest.from_dict(d)
    d = dict(m); a = s0.to_dict(); b = s1.to_dict(); b["local_time_range"]["start_us"] = 999_999; d["segments"] = [a, b]
    with pytest.raises(ValueError): mf.StorageManifest.from_dict(d)
    d = dict(m); a = s0.to_dict(); b = s1.to_dict(); b["local_time_range"]["end_us"] = 999_999; d["segments"] = [a, b]
    with pytest.raises(ValueError): mf.StorageManifest.from_dict(d)


def test_manifest_split_validation():
    s = segment()
    valid = mf.make_manifest(
        dataset_id="ds2",
        created_at_utc="2026-05-26T00:00:00Z",
        segments=(s,),
        splits=(mf.SplitMetadata(SplitRole.TRAIN, s.segment_key, 0, 5, TimeRangeUS(s.local_time_range.start_us, s.local_time_range.start_us + 1)),),
    )
    assert valid.splits
    d = manifest_one_segment().to_dict()
    d["splits"] = [mf.SplitMetadata(SplitRole.TRAIN, "missing", 0, 5, TimeRangeUS(1_000_000, 1_000_001)).to_dict()]
    with pytest.raises(ValueError): mf.StorageManifest.from_dict(d)
    d = manifest_one_segment().to_dict(); bad = mf.SplitMetadata(SplitRole.TRAIN, "seg_000", 0, 5, TimeRangeUS(1_000_000, 1_000_001)).to_dict(); bad["start_row"] = -1; d["splits"] = [bad]
    with pytest.raises(ValueError): mf.StorageManifest.from_dict(d)
    d = manifest_one_segment().to_dict(); d["splits"] = [mf.SplitMetadata(SplitRole.TRAIN, "seg_000", 0, 20, TimeRangeUS(1_000_000, 1_000_001)).to_dict()]
    with pytest.raises(ValueError): mf.StorageManifest.from_dict(d)
    d = manifest_one_segment().to_dict(); d["splits"] = [mf.SplitMetadata(SplitRole.TRAIN, "seg_000", 0, 5, TimeRangeUS(999_999, 1_000_001)).to_dict()]
    with pytest.raises(ValueError): mf.StorageManifest.from_dict(d)
    d = manifest_one_segment().to_dict(); d["splits"] = [mf.SplitMetadata(SplitRole.TRAIN, "seg_000", 0, 5, TimeRangeUS(1_000_000, 99_000_000)).to_dict()]
    with pytest.raises(ValueError): mf.StorageManifest.from_dict(d)
    sp = mf.SplitMetadata(SplitRole.TRAIN, "seg_000", 0, 5, TimeRangeUS(1_000_000, 1_000_001)).to_dict()
    d = manifest_one_segment().to_dict(); d["splits"] = [sp, sp]
    with pytest.raises(ValueError): mf.StorageManifest.from_dict(d)
    d = manifest_one_segment().to_dict(); d["splits"] = []
    mf.StorageManifest.from_dict(d)


def test_pipeline_config_to_manifest_dict():
    d = mf.pipeline_config_to_manifest_dict(default_config())
    for k in ("exchange", "symbol", "source_data_types", "disabled_context_data_types", "decision_policy", "decision_reason", "decision_stride_us", "horizons_us", "entry_delay_us", "price_reference", "asof_policy", "lookback_rows", "feature_dtype", "label_dtype", "timestamp_dtype", "storage_format", "time_unit", "pipeline_schema", "feature_schema"):
        assert k in d
    assert d["decision_stride_us"] == 500_000
    assert d["storage_format"] == "flat_decision_rows_us"
    assert d["time_unit"] == "us"
    assert d["feature_schema"] == specs.FEATURE_SCHEMA


def test_transform_metadata_json_safe():
    transforms = pytest.importorskip("mmrt.features.transforms")
    m = mf.make_manifest(
        dataset_id="ds3",
        created_at_utc="2026-05-26T00:00:00Z",
        segments=(segment(),),
        transform_config=transforms.TransformConfig().as_dict(),
        transform_diagnostics={"rows_seen": 10, "nonfinite_raw_count": 0},
    )
    assert m.transform_config == transforms.TransformConfig().as_dict()
    assert m.transform_diagnostics == {"rows_seen": 10, "nonfinite_raw_count": 0}
    with pytest.raises(ValueError): mf.make_manifest(dataset_id="ds3", created_at_utc="2026-05-26T00:00:00Z", segments=(segment(),), transform_config={"bad": float("nan")})
    with pytest.raises(ValueError): mf.make_manifest(dataset_id="ds3", created_at_utc="2026-05-26T00:00:00Z", segments=(segment(),), transform_diagnostics={1: "bad"})


def test_manifest_from_dict_missing_required_keys_raise_value_error():
    base = manifest_one_segment().to_dict()
    required = ["schema", "dataset_id", "created_at_utc", "pipeline_config", "writer_metadata", "feature_schema", "label_spec", "transform_config", "transform_diagnostics", "exchange", "symbol", "storage_format", "time_unit", "decision_stride_us", "feature_columns", "label_columns", "required_columns", "segments"]
    for key in required:
        d = json.loads(json.dumps(base))
        del d[key]
        with pytest.raises(ValueError): mf.StorageManifest.from_dict(d)
    with pytest.raises(ValueError): mf.StorageSegment.from_dict({"segment_key": "s", "parquet_path": "p.parquet", "label_count": 0, "time_range": {"start_us": 1, "end_us": 2}, "local_time_range": {"start_us": 1, "end_us": 2}, "first_row_idx": 0, "last_row_idx": 0})
    with pytest.raises(ValueError): mf.SplitMetadata.from_dict({"segment_key": "s", "start_row": 0, "end_row": 1, "local_time_range": {"start_us": 1, "end_us": 2}})


def test_no_future_leakage_terms_or_row_repair_surface():
    payload = json.dumps(manifest_one_segment().to_dict()).lower()
    api = " ".join(mf.__all__).lower()
    text = payload + " " + api
    forbidden = ["future_" + "mid", "future_" + "ret", "fit_" + "transform", "standard" + "scaler", "p" + "ca", "grace_" + "ms", "drop_duplicate_" + "trades", "stride_" + "rows"]
    for term in forbidden:
        assert term.lower() not in text


def test_tardis_microsecond_local_timestamp_metadata():
    assert mf.TS_US_COLUMN == "ts_us"
    assert mf.LOCAL_TS_US_COLUMN == "local_ts_us"
    assert mf.TS_US_COLUMN in mf.BASE_ROW_COLUMNS and mf.LOCAL_TS_US_COLUMN in mf.BASE_ROW_COLUMNS
    ls = default_config().label_spec
    req = mf.required_row_columns(ls)
    assert mf.TS_US_COLUMN in req and mf.LOCAL_TS_US_COLUMN in req
    s = mf.StorageSegment("segx", "segments/segx.parquet", 10, 10, TimeRangeUS(900_000, 5_900_000), TimeRangeUS(1_000_000, 6_000_000), 0, 9, ("raw/segx.csv.gz",))
    m = mf.make_manifest(dataset_id="ds4", created_at_utc="2026-05-26T00:00:00Z", segments=(s,))
    assert m.time_unit == TimeUnit.MICROSECOND
    d = m.to_dict()
    assert "time_range" in d["segments"][0] and "local_time_range" in d["segments"][0]
    assert d["segments"][0]["time_range"]["start_us"] == 900_000
    assert d["segments"][0]["local_time_range"]["start_us"] == 1_000_000
    assert mf.StorageManifest.from_dict(d) == m
