import importlib
import json
import subprocess
import sys

import pytest

import mmrt.storage.manifest as mf
from mmrt.config import default_config
from mmrt.contracts import LabelSpec, SplitRole, StorageFormat, TimeRangeUS, TimeUnit
from mmrt.features import specs


def segment(key="seg_000", start=0, rows=10, local_start=1_000_000):
    return mf.StorageSegment(key, f"segments/{key}.parquet", rows, rows, TimeRangeUS(local_start, local_start + rows * 500_000), TimeRangeUS(local_start, local_start + rows * 500_000), start, start + rows - 1, (f"raw/{key}.csv.gz",))


def test_public_api_boundary():
    expected = ["MANIFEST_SCHEMA_VERSION","DEFAULT_MANIFEST_FILENAME","ROW_IDX_COLUMN","DECISION_INDEX_COLUMN","TS_US_COLUMN","LOCAL_TS_US_COLUMN","EVENT_SEQ_COLUMN","RAW_MID_COLUMN","LABEL_ENTRY_TS_US_COLUMN","FEATURE_COLUMN_PREFIX","LABEL_COLUMN_PREFIX","BASE_ROW_COLUMNS","DEFAULT_COMPRESSION","DEFAULT_PARQUET_VERSION","StorageSegment","SplitMetadata","StorageManifest","feature_columns","label_columns","required_row_columns","feature_schema_record","default_writer_metadata","label_spec_to_dict","label_spec_from_dict","time_range_to_dict","time_range_from_dict","pipeline_config_to_manifest_dict","make_manifest","manifest_sha256","manifest_to_json_bytes","manifest_from_json_bytes","write_manifest_json","read_manifest_json"]
    assert set(mf.__all__) == set(expected)

def test_feature_and_label_columns_match_specs():
    cols = mf.feature_columns(); assert len(cols) == specs.FEATURE_COUNT; assert cols[0] == "x_" + specs.FEATURE_NAMES[0]; assert cols[-1] == "x_" + specs.FEATURE_NAMES[-1]

def test_storage_segment_validation_and_roundtrip():
    s = segment(); assert mf.StorageSegment.from_dict(s.to_dict()) == s
    with pytest.raises(ValueError): mf.StorageSegment("", "a.parquet", 1, 0, TimeRangeUS(1,2), TimeRangeUS(1,2), 0, 0)

def test_split_metadata_validation_and_roundtrip():
    sp = mf.SplitMetadata(SplitRole.TRAIN, "seg_000", 0, 5, TimeRangeUS(1, 2)); assert mf.SplitMetadata.from_dict(sp.to_dict()) == sp

def test_make_manifest_defaults_and_validation(tmp_path):
    m = mf.make_manifest(dataset_id="ds1", created_at_utc="2026-05-26T00:00:00Z", segments=(segment(),))
    assert m.manifest_schema_version == mf.MANIFEST_SCHEMA_VERSION and m.storage_format == StorageFormat.FLAT_DECISION_ROWS_US_V1 and m.time_unit == TimeUnit.MICROSECOND
    b = mf.manifest_to_json_bytes(m); assert b.endswith(b"\n")
    p = tmp_path / "manifest.json"; mf.write_manifest_json(m, p); assert mf.read_manifest_json(p) == m


def test_pipeline_config_to_manifest_dict():
    d = mf.pipeline_config_to_manifest_dict(default_config())
    assert d["decision_stride_us"] == 500_000 and d["storage_format"] == "flat_decision_rows_us_v1" and d["time_unit"] == "us"
