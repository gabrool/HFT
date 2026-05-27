import inspect
import json
from pathlib import Path

import pytest

pytest.importorskip("pyarrow")
pytest.importorskip("pyarrow.parquet")

import mmrt.cli.audit_dataset as cli
from mmrt.contracts import SplitRole, TimeRangeUS
from mmrt.features import specs
from mmrt.storage import manifest as mf
from mmrt.storage import reader as rd
from mmrt.storage import writer as wr


def feature_values(scale: float = 0.01) -> tuple[float, ...]:
    return tuple(float(i) * scale for i in range(specs.FEATURE_COUNT))


def label_values() -> tuple[float, float, float]:
    return (1.0, -2.0, 0.5)


def row(i: int, *, local_ts_us: int | None = None, ts_us: int | None = None) -> wr.DecisionRow:
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


def make_dataset(tmp_path: Path, *, rows: int = 5, chunk_rows: int = 2, splits=()):
    root = tmp_path / "d"
    w = wr.DecisionRowWriter(wr.WriterConfig(dataset_id="d", created_at_utc="2026", dataset_root=str(root), chunk_rows=chunk_rows))
    for i in range(rows):
        w.append(row(i))
    manifest = w.finalize()
    if splits:
        m0 = mf.read_manifest_json(root / mf.DEFAULT_MANIFEST_FILENAME)
        m1 = mf.StorageManifest(
            m0.manifest_schema_version,
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
            m0.decision_stride_us,
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


def test_public_api_boundary() -> None:
    assert cli.__all__ == ["build_arg_parser", "audit_dataset", "main"]


def test_build_arg_parser_defaults() -> None:
    parser = cli.build_arg_parser()
    args = parser.parse_args(["--dataset-root", "ds"])
    assert args.output_json is None
    assert args.batch_size == rd.DEFAULT_BATCH_SIZE
    assert args.max_scan_rows == 200_000
    assert args.no_validate_on_open is False
    assert args.no_scan_splits is False


def test_parser_rejects_bad_values() -> None:
    parser = cli.build_arg_parser()
    for bad in (["--batch-size", "0"], ["--batch-size", "-1"], ["--max-scan-rows", "0"], ["--max-scan-rows", "-1"]):
        with pytest.raises(SystemExit):
            parser.parse_args(["--dataset-root", "ds", *bad])


def test_audit_dataset_manifest_only_without_splits(tmp_path: Path) -> None:
    root, manifest = make_dataset(tmp_path, rows=6, chunk_rows=2, splits=())
    report = cli.audit_dataset(str(root), batch_size=2, max_scan_rows=10)
    assert report["status"] == "ok"
    assert report["manifest"]["dataset_id"] == manifest.dataset_id
    assert report["manifest"]["manifest_hash"] == manifest.content_hash()
    assert report["segments"]["count"] == len(manifest.segments)
    assert report["splits"]["train"]["row_count"] == 0
    assert report["readiness"]["train_ready"] is False
    assert "no_split_entries" in report["warnings"]
    assert "missing_train_split" in report["warnings"]
    assert "missing_val_split" in report["warnings"]


def test_audit_dataset_with_train_val_test_splits(tmp_path: Path) -> None:
    root, manifest = make_dataset(tmp_path, rows=12, chunk_rows=4)
    seg0, seg1, seg2 = manifest.segments
    splits = (
        mf.SplitMetadata(
            role=SplitRole.TRAIN,
            segment_key=seg0.segment_key,
            start_row=0,
            end_row=4,
            local_time_range=TimeRangeUS(seg0.local_time_range.start_us, seg0.local_time_range.end_us),
        ),
        mf.SplitMetadata(
            role=SplitRole.VAL,
            segment_key=seg1.segment_key,
            start_row=4,
            end_row=8,
            local_time_range=TimeRangeUS(seg1.local_time_range.start_us, seg1.local_time_range.end_us),
        ),
        mf.SplitMetadata(
            role=SplitRole.TEST,
            segment_key=seg2.segment_key,
            start_row=8,
            end_row=12,
            local_time_range=TimeRangeUS(seg2.local_time_range.start_us, seg2.local_time_range.end_us),
        ),
    )
    m2 = mf.StorageManifest(
        manifest.manifest_schema_version, manifest.dataset_id, manifest.created_at_utc, manifest.pipeline_config, manifest.writer_metadata,
        manifest.feature_schema, manifest.label_spec, manifest.transform_config, manifest.transform_diagnostics,
        manifest.exchange, manifest.symbol, manifest.storage_format, manifest.time_unit, manifest.decision_stride_us,
        manifest.feature_columns, manifest.label_columns, manifest.required_columns, manifest.segments, splits, manifest.notes,
    )
    mf.write_manifest_json(m2, root / mf.DEFAULT_MANIFEST_FILENAME)

    report = cli.audit_dataset(str(root), batch_size=2, max_scan_rows=100)
    assert report["readiness"] == {
        "has_train_split": True,
        "has_val_split": True,
        "has_test_split": True,
        "train_ready": True,
    }
    for role, first, last in (("train", 0, 3), ("val", 4, 7), ("test", 8, 11)):
        assert report["splits"][role]["row_count"] == 4
        scan = report["splits"][role]["scan"]
        assert scan["scanned_rows"] == 4
        assert scan["scan_limit_hit"] is False
        assert scan["first_row_idx"] == first
        assert scan["last_row_idx"] == last
        assert scan["strictly_increasing_row_idx"] is True
    assert report["warnings"] == []


def test_audit_dataset_scan_limit(tmp_path: Path) -> None:
    root, manifest = make_dataset(tmp_path, rows=4, chunk_rows=4)
    seg0 = manifest.segments[0]
    split = (mf.SplitMetadata(
            role=SplitRole.TRAIN,
            segment_key=seg0.segment_key,
            start_row=0,
            end_row=4,
            local_time_range=TimeRangeUS(seg0.local_time_range.start_us, seg0.local_time_range.end_us),
        ),)
    m0 = mf.read_manifest_json(root / mf.DEFAULT_MANIFEST_FILENAME)
    m1 = mf.StorageManifest(
        m0.manifest_schema_version, m0.dataset_id, m0.created_at_utc, m0.pipeline_config, m0.writer_metadata,
        m0.feature_schema, m0.label_spec, m0.transform_config, m0.transform_diagnostics,
        m0.exchange, m0.symbol, m0.storage_format, m0.time_unit, m0.decision_stride_us,
        m0.feature_columns, m0.label_columns, m0.required_columns, m0.segments, split, m0.notes,
    )
    mf.write_manifest_json(m1, root / mf.DEFAULT_MANIFEST_FILENAME)
    report = cli.audit_dataset(str(root), batch_size=2, max_scan_rows=2)
    scan = report["splits"]["train"]["scan"]
    assert scan["scanned_rows"] == 2
    assert scan["manifest_row_count"] == 4
    assert scan["scan_limit_hit"] is True
    assert "split_scan_limit_hit:train" in report["warnings"]


def test_audit_dataset_no_scan_splits_does_not_iterate(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root, manifest = make_dataset(tmp_path, rows=8, chunk_rows=4)
    seg0, seg1 = manifest.segments[:2]
    splits = (
        mf.SplitMetadata(
            role=SplitRole.TRAIN,
            segment_key=seg0.segment_key,
            start_row=0,
            end_row=4,
            local_time_range=TimeRangeUS(seg0.local_time_range.start_us, seg0.local_time_range.end_us),
        ),
        mf.SplitMetadata(
            role=SplitRole.VAL,
            segment_key=seg1.segment_key,
            start_row=4,
            end_row=8,
            local_time_range=TimeRangeUS(seg1.local_time_range.start_us, seg1.local_time_range.end_us),
        ),
    )
    m0 = mf.read_manifest_json(root / mf.DEFAULT_MANIFEST_FILENAME)
    m1 = mf.StorageManifest(
        m0.manifest_schema_version, m0.dataset_id, m0.created_at_utc, m0.pipeline_config, m0.writer_metadata,
        m0.feature_schema, m0.label_spec, m0.transform_config, m0.transform_diagnostics,
        m0.exchange, m0.symbol, m0.storage_format, m0.time_unit, m0.decision_stride_us,
        m0.feature_columns, m0.label_columns, m0.required_columns, m0.segments, splits, m0.notes,
    )
    mf.write_manifest_json(m1, root / mf.DEFAULT_MANIFEST_FILENAME)
    reader = rd.open_dataset(str(root))
    monkeypatch.setattr(rd, "open_dataset", lambda *args, **kwargs: reader)
    monkeypatch.setattr(reader, "iter_split_batches", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("should not iterate")))
    report = cli.audit_dataset(str(root), scan_splits=False)
    assert report["validation"]["split_scan_enabled"] is False
    assert report["splits"]["train"]["scan"] is None


def test_audit_dataset_uses_streaming_split_batches(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root, manifest = make_dataset(tmp_path, rows=8, chunk_rows=4)
    seg0, seg1 = manifest.segments[:2]
    splits = (
        mf.SplitMetadata(
            role=SplitRole.TRAIN,
            segment_key=seg0.segment_key,
            start_row=0,
            end_row=4,
            local_time_range=TimeRangeUS(seg0.local_time_range.start_us, seg0.local_time_range.end_us),
        ),
        mf.SplitMetadata(
            role=SplitRole.VAL,
            segment_key=seg1.segment_key,
            start_row=4,
            end_row=8,
            local_time_range=TimeRangeUS(seg1.local_time_range.start_us, seg1.local_time_range.end_us),
        ),
    )
    m0 = mf.read_manifest_json(root / mf.DEFAULT_MANIFEST_FILENAME)
    m1 = mf.StorageManifest(
        m0.manifest_schema_version, m0.dataset_id, m0.created_at_utc, m0.pipeline_config, m0.writer_metadata,
        m0.feature_schema, m0.label_spec, m0.transform_config, m0.transform_diagnostics,
        m0.exchange, m0.symbol, m0.storage_format, m0.time_unit, m0.decision_stride_us,
        m0.feature_columns, m0.label_columns, m0.required_columns, m0.segments, splits, m0.notes,
    )
    mf.write_manifest_json(m1, root / mf.DEFAULT_MANIFEST_FILENAME)
    reader = rd.open_dataset(str(root))
    monkeypatch.setattr(rd, "open_dataset", lambda *args, **kwargs: reader)
    monkeypatch.setattr(reader, "read_split_table", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("bad")))
    monkeypatch.setattr(reader, "read_table", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("bad")))
    monkeypatch.setattr(reader, "read_segment_table", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("bad")))
    report = cli.audit_dataset(str(root), scan_splits=True)
    assert report["splits"]["train"]["scan"]["scanned_rows"] > 0


def test_main_writes_output_json_and_prints_report(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setattr(cli, "audit_dataset", lambda *args, **kwargs: {"status": "ok", "dataset_root": "ds", "warnings": []})
    out = tmp_path / "audit.json"
    rc = cli.main(["--dataset-root", "ds", "--output-json", str(out)])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out.strip())
    assert payload["output_json"] == str(out)
    assert out.exists()
    assert json.loads(out.read_text(encoding="utf-8"))["status"] == "ok"
    assert not (tmp_path / "audit.json.tmp").exists()


def test_main_passes_flags_to_audit(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {}

    def fake_audit(dataset_root: str, **kwargs):
        calls["dataset_root"] = dataset_root
        calls.update(kwargs)
        return {"status": "ok", "warnings": []}

    monkeypatch.setattr(cli, "audit_dataset", fake_audit)
    cli.main([
        "--dataset-root", "ds", "--batch-size", "7", "--max-scan-rows", "11", "--no-validate-on-open", "--no-scan-splits",
    ])
    assert calls["dataset_root"] == "ds"
    assert calls["validate_on_open"] is False
    assert calls["batch_size"] == 7
    assert calls["max_scan_rows"] == 11
    assert calls["scan_splits"] is False


def test_output_json_requires_json_suffix() -> None:
    with pytest.raises(ValueError):
        cli._write_json_atomic({"status": "ok"}, "x.txt")


def test_no_bad_imports() -> None:
    src = inspect.getsource(cli)
    forbidden = [
        "from mmrt.data", "import mmrt.data", "from mmrt.features.engine", "from mmrt.features.labels", "from mmrt.features.transforms",
        "import pan" + "das", "from pan" + "das", "import to" + "rch", "from to" + "rch", "import sk" + "learn", "from sk" + "learn",
        "CM" + "SSL", "offline_" + "ingest", "linear_" + "offline",
    ]
    for token in forbidden:
        assert token not in src


def test_no_old_pipeline_residue() -> None:
    src = inspect.getsource(cli)
    forbidden = [
        "BY" + "BIT", "CM" + "SSL", "offline_" + "ingest", "stage" + "1", "stage" + "2", "stage" + "3", "stage" + "4", "stage" + "5",
        "Mini" + "Rocket", "Multi" + "Rocket", "Hy" + "dra", "Ae" + "on", "P" + "CA", "Standard" + "Scaler", "sk" + "learn",
        "to" + "rch", "pan" + "das", "po" + "lars", "GRACE_" + "MS", "global_" + "meta", "week", "tar." + "zst",
    ]
    for token in forbidden:
        assert token not in src


def test_no_raw_ingest_feature_train_or_split_building_surface() -> None:
    src = inspect.getsource(cli)
    forbidden = [
        "tardis_" + "csv", "event_" + "merge", "book_" + "reconstructor", "Feature" + "Engine", "Label" + "Builder",
        "CausalFeature" + "Transformer", "DecisionRow" + "Writer", "build_" + "split_plan", "write_" + "split_manifest",
        "build_" + "and_write_splits", "Split" + "Metadata", "train_linear_model", "write_linear_train_artifacts", "LinearTrainConfig",
    ]
    for token in forbidden:
        assert token not in src


def test_no_future_leakage_or_mutation_surface() -> None:
    src = inspect.getsource(cli)
    forbidden = [
        "future_" + "mid", "future_" + "ret", "shu" + "ffle", "sort_" + "values", "rand" + "om", "threshold_" + "search",
        "optimize_" + "threshold", "fit_" + "transform", "append_values", "finalize(", "write_manifest_json", "LOCAL_TS_" + "US_COLUMN",
        "TS_US_" + "COLUMN", "EVENT_SEQ_" + "COLUMN", "RAW_MID_" + "COLUMN",
    ]
    for token in forbidden:
        assert token not in src


def test_scan_source_is_streaming_and_narrow() -> None:
    src = inspect.getsource(cli._scan_split)
    assert "iter_split_batches" in src
    for bad in ["read_split_table", "read_table", "read_segment_table", "feature_columns", "label_columns", "to_pandas", ".iterrows"]:
        assert bad not in src
