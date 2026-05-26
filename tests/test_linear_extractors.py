import inspect
import subprocess
import sys

import numpy as np
import pytest

pa = pytest.importorskip("pyarrow")

import mmrt.linear.extractors as ex
import mmrt.storage.manifest as mf


def _segment():
    return mf.StorageSegment(
        segment_key="seg_000",
        parquet_path="segments/seg_000.parquet",
        row_count=3,
        label_count=3,
        time_range=mf.TimeRangeUS(1, 4),
        local_time_range=mf.TimeRangeUS(1, 4),
        first_row_idx=0,
        last_row_idx=2,
        source_files=("raw/seg_000.csv.gz",),
    )


def make_manifest() -> mf.StorageManifest:
    return mf.make_manifest(dataset_id="ds", created_at_utc="2026-05-26T00:00:00Z", segments=(_segment(),))


def test_public_api_boundary():
    assert ex.__all__ == [
        "DEFAULT_EXTRACTOR_DTYPE",
        "ALLOWED_EXTRACTOR_DTYPES",
        "LinearFeatureExtractorConfig",
        "LinearFeatureBatch",
        "IdentityFeatureExtractor",
        "resolve_feature_columns",
        "table_to_feature_matrix",
        "make_identity_extractor",
    ]
    forbidden = ["mi"+"ni", "rock"+"et", "hy"+"dra", "ae"+"on", "cm"+"ssl", "by"+"bit", "sta"+"ge", "win"+"dow", "la"+"g", "to"+"rch", "sk"+"learn", "tar"+"get"]
    for name in ex.__all__:
        lower = name.lower()
        for token in forbidden:
            assert token not in lower


def test_no_forbidden_imports():
    code = "import sys; before=set(sys.modules); import mmrt.linear.extractors; after=set(sys.modules)-before; print('\\n'.join(sorted(after)))"
    out = subprocess.check_output([sys.executable, "-c", code], text=True)
    delta = set(out.splitlines())
    forbidden = [
        "pan" + "das",
        "po" + "lars",
        "to" + "rch",
        "sk" + "learn",
        "aeon",
        "ski"+"time",
        "num" + "ba",
        "mmrt.features.engine",
        "mmrt.features.labels",
        "mmrt.features.transforms",
        "mmrt.data.tardis_csv",
        "mmrt.data.event_merge",
        "mmrt.storage.writer",
        "CM" + "SSL17",
        "offline_" + "ingest",
    ]
    for mod in delta:
        for token in forbidden:
            assert token not in mod


def test_config_validation():
    cfg = ex.LinearFeatureExtractorConfig()
    assert cfg.output_dtype == "float32"
    assert cfg.dtype == np.dtype("float32")
    assert ex.LinearFeatureExtractorConfig(output_dtype="float64").dtype == np.dtype("float64")
    with pytest.raises(ValueError):
        ex.LinearFeatureExtractorConfig(output_dtype="float16")
    assert ex.LinearFeatureExtractorConfig(feature_columns=None).feature_columns is None
    assert ex.LinearFeatureExtractorConfig(feature_columns=["a", "b"]).feature_columns == ("a", "b")
    with pytest.raises(ValueError):
        ex.LinearFeatureExtractorConfig(feature_columns=["a", "a"])
    with pytest.raises(ValueError):
        ex.LinearFeatureExtractorConfig(feature_columns=[])
    with pytest.raises(ValueError):
        ex.LinearFeatureExtractorConfig(feature_columns=[""])
    with pytest.raises(ValueError):
        ex.LinearFeatureExtractorConfig(feature_columns=[1])
    with pytest.raises(ValueError):
        ex.LinearFeatureExtractorConfig(require_all_finite=1)
    with pytest.raises(ValueError):
        ex.LinearFeatureExtractorConfig(copy=1)
    assert not hasattr(cfg, "extractor_type")
    assert not hasattr(cfg, "rand"+"om_state")
    assert not hasattr(cfg, "window_size")


def test_linear_feature_batch_validation_and_copy():
    x = np.ascontiguousarray(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32))
    b = ex.LinearFeatureBatch(X=x, feature_columns=("a", "b"))
    assert b.n_rows == 2 and b.n_features == 2
    with pytest.raises(ValueError):
        ex.LinearFeatureBatch(X=x, feature_columns=("a",))
    with pytest.raises(ValueError):
        ex.LinearFeatureBatch(X=np.array([1.0, 2.0], dtype=np.float32), feature_columns=("a", "b"))
    with pytest.raises(ValueError):
        ex.LinearFeatureBatch(X=np.array([[1.0, np.nan]], dtype=np.float32), feature_columns=("a", "b"))
    with pytest.raises(ValueError):
        ex.LinearFeatureBatch(X=np.array([[1, 2]], dtype=np.int64), feature_columns=("a", "b"))
    x[0, 0] = 999.0
    assert b.X[0, 0] == 1.0


def test_resolve_feature_columns_all_and_subset():
    m = make_manifest()
    all_cols = ex.resolve_feature_columns(m)
    assert all_cols == tuple(m.feature_columns)
    subset = (m.feature_columns[2], m.feature_columns[0])
    got = ex.resolve_feature_columns(m, subset)
    assert got == subset
    with pytest.raises(ValueError):
        ex.resolve_feature_columns(m, ("x_missing",))
    with pytest.raises(ValueError):
        ex.resolve_feature_columns(m, ("a", "a"))
    with pytest.raises(ValueError):
        ex.resolve_feature_columns(m, ())
    assert mf.ROW_IDX_COLUMN not in all_cols


def test_table_to_feature_matrix_preserves_order_and_dtype():
    t = pa.table({"x_a": [1, 2, 3], "x_b": [10, 20, 30], "x_c": [100, 200, 300]})
    x = ex.table_to_feature_matrix(t, ("x_b", "x_a"))
    assert x.shape == (3, 2)
    assert np.array_equal(x, np.array([[10, 1], [20, 2], [30, 3]], dtype=np.float32))
    assert x.dtype == np.float32
    assert x.flags.c_contiguous
    x64 = ex.table_to_feature_matrix(t, ("x_b", "x_a"), output_dtype="float64")
    assert x64.dtype == np.float64


def test_table_to_feature_matrix_empty_rows():
    t = pa.table({"x_a": pa.array([], type=pa.float64()), "x_b": pa.array([], type=pa.float64())})
    x = ex.table_to_feature_matrix(t, ("x_a", "x_b"))
    assert x.shape == (0, 2)
    assert x.dtype == np.float32


def test_table_to_feature_matrix_missing_or_nonfinite_rejected():
    t = pa.table({"x_a": [1.0, 2.0], "x_b": [3.0, 4.0]})
    with pytest.raises(ValueError):
        ex.table_to_feature_matrix(t, ("x_a", "x_c"))
    bad = pa.table({"x_a": [1.0, np.nan], "x_b": [3.0, np.inf]})
    with pytest.raises(ValueError):
        ex.table_to_feature_matrix(bad, ("x_a", "x_b"))
    ok = ex.table_to_feature_matrix(bad, ("x_a", "x_b"), require_all_finite=False)
    assert ok.shape == (2, 2)


def test_identity_extractor_resolves_manifest_and_transforms_table():
    m = make_manifest()
    cols = tuple(m.feature_columns[:3])
    t = pa.table({cols[0]: [1, 2], cols[1]: [3, 4], cols[2]: [5, 6]})
    cfg = ex.LinearFeatureExtractorConfig(feature_columns=cols)
    ext = ex.IdentityFeatureExtractor(config=cfg, manifest=m)
    assert ext.feature_columns == cols
    out = ext.transform_table(t)
    assert out.feature_columns == cols
    assert out.X.dtype == np.float32
    d = ext.as_dict()
    assert d["extractor"] == "identity"
    assert d["output_dtype"] == "float32"
    assert d["feature_columns"] == list(cols)
    assert d["feature_schema_hash"] == m.feature_schema.get("feature_specs_hash")


def test_identity_extractor_subset_projection():
    m = make_manifest()
    subset = (m.feature_columns[3], m.feature_columns[1])
    cfg = ex.LinearFeatureExtractorConfig(feature_columns=subset)
    ext = ex.IdentityFeatureExtractor(config=cfg)
    assert ext.column_projection(m) == subset
    t = pa.table({subset[0]: [7, 8], subset[1]: [1, 2]})
    out = ext.transform_table(t)
    assert np.array_equal(out.X, np.array([[7, 1], [8, 2]], dtype=np.float32))


def test_identity_extractor_transform_numpy():
    cfg = ex.LinearFeatureExtractorConfig(output_dtype="float64", copy=True)
    ext = ex.IdentityFeatureExtractor(config=cfg)
    out = ext.transform_numpy(np.array([[1, 2], [3, 4]], dtype=np.int64), feature_columns=("a", "b"))
    assert out.X.dtype == np.float64
    with pytest.raises(ValueError):
        ext.transform_numpy(np.array([[1.0, 2.0]], dtype=np.float64), feature_columns=("a", "c"))
    with pytest.raises(ValueError):
        ext.transform_numpy(np.array([[1.0]], dtype=np.float64))
    ext2 = ex.IdentityFeatureExtractor(config=ex.LinearFeatureExtractorConfig(require_all_finite=True), manifest=make_manifest())
    with pytest.raises(ValueError):
        ext2.transform_numpy(np.array([[np.nan] * len(ext2.feature_columns)], dtype=np.float32), feature_columns=ext2.feature_columns)


def test_identity_extractor_has_no_fit_or_stage_api():
    ext = ex.IdentityFeatureExtractor()
    assert not hasattr(ext, "fit")
    assert not hasattr(ext, "fit_transform")
    assert not hasattr(ext, "partial_fit")
    assert not hasattr(ext, "extract_windows")
    assert not hasattr(ext, "stage")
    assert not hasattr(ex, "build_extractor")
    assert not hasattr(ex, "Mini" + "RocketExtractor")
    assert not hasattr(ex, "Hy" + "draExtractor")
    assert not hasattr(ex, "Multi" + "RocketExtractor")


def test_no_future_leakage_or_timestamp_surface():
    src = inspect.getsource(ex)
    forbidden = [
        "future_" + "mid",
        "future_" + "ret",
        "label_" + "now",
        "on_" + "decision",
        "transform_" + "one",
        "local_" + "ts_us",
        "ts_" + "us",
        "timestamp",
        "event_seq",
        "sort_"+"values",
        "shuf"+"fle",
        "rand" + "om",
    ]
    lowered = src.lower()
    for token in forbidden:
        assert token not in lowered


def test_no_nonlinear_old_extractor_residue():
    src = inspect.getsource(ex)
    forbidden = [
        "Mini" + "Rocket",
        "Multi" + "Rocket",
        "Hy" + "dra",
        "Ae" + "on",
        "rock" + "et",
        "ker" + "nel",
        "convol"+"ution",
        "stage" + "1",
        "stage" + "2",
        "stage" + "3",
        "stage" + "4",
        "stage" + "5",
        "BY" + "BIT",
        "CM" + "SSL",
        "offline_" + "ingest",
    ]
    for token in forbidden:
        assert token not in src
