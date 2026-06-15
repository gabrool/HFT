import inspect
import subprocess
import sys

import numpy as np
import pytest

import mmrt.linear.preprocess as pp


def test_public_api_boundary():
    expected = [
        "DEFAULT_PREPROCESS_DTYPE",
        "ALLOWED_PREPROCESS_DTYPES",
        "DEFAULT_VARIANCE_FLOOR",
        "DEFAULT_CLIP_Z",
        "LinearPreprocessConfig",
        "RunningFeatureStats",
        "LinearPreprocessState",
        "LinearPreprocessor",
        "fit_preprocessor",
        "transform_with_state",
    ]
    assert pp.__all__ == expected
    forbidden = (
        "bybit",
        "cmssl",
        "stage",
        "pca",
        "sklearn",
        "torch",
        "pandas",
        "polars",
        "target",
        "model",
        "evaluate",
        "reader",
        "writer",
    )
    for name in pp.__all__:
        lowered = name.lower()
        assert all(token not in lowered for token in forbidden)


def test_no_forbidden_imports():
    code = (
        "import sys\n"
        "before=set(sys.modules)\n"
        "import mmrt.linear.preprocess\n"
        "after=set(sys.modules)-before\n"
        "print('\\n'.join(sorted(after)))\n"
    )
    proc = subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True)
    loaded = set(proc.stdout.splitlines())
    forbidden = {
        "pandas",
        "polars",
        "torch",
        "sklearn",
        "scipy",
        "numba",
        "pyarrow",
        "mmrt.storage.reader",
        "mmrt.storage.writer",
        "mmrt.storage.splits",
        "mmrt.linear.extractors",
        "mmrt.linear.targets",
        "mmrt.features.engine",
        "mmrt.features.labels",
        "mmrt.features.transforms",
        "CMSSL17",
        "offline_ingest",
    }
    for mod in forbidden:
        assert mod not in loaded


def test_config_validation():
    cfg = pp.LinearPreprocessConfig()
    assert cfg.variance_floor == 1e-12
    assert cfg.clip_z == 8.0
    assert cfg.output_dtype == "float32"
    assert pp.LinearPreprocessConfig(output_dtype="float64").dtype == np.dtype("float64")
    with pytest.raises(ValueError):
        pp.LinearPreprocessConfig(output_dtype="int32")
    for bad in (0.0, -1.0, np.nan, np.inf, True):
        with pytest.raises(ValueError):
            pp.LinearPreprocessConfig(variance_floor=bad)
    for bad in (0.0, -1.0, np.nan, np.inf, True):
        with pytest.raises(ValueError):
            pp.LinearPreprocessConfig(clip_z=bad)
    for name in ("pca_components", "stage", "random_state", "fit_on", "split", "sample_weight", "drop_constant_features"):
        assert not hasattr(cfg, name)


def test_running_stats_matches_numpy_single_batch():
    X = np.array([[1.0, 3.0], [2.0, 5.0], [7.0, 11.0]], dtype=np.float64)
    stats = pp.RunningFeatureStats.empty(2)
    stats.update(X)
    np.testing.assert_allclose(stats.mean, X.mean(axis=0))
    np.testing.assert_allclose(stats.variance(), X.var(axis=0, ddof=1))
    assert stats.n_rows == 3


def test_running_stats_matches_numpy_multiple_batches():
    X = np.array([[1.0, 2.0], [2.0, 4.0], [3.0, 8.0], [6.0, 16.0]], dtype=np.float64)
    stats = pp.RunningFeatureStats.empty(2)
    stats.update(X[:2])
    mean_before = stats.mean.copy()
    var_before = stats.variance().copy()
    stats.update(np.empty((0, 2), dtype=np.float64))
    np.testing.assert_allclose(stats.mean, mean_before)
    np.testing.assert_allclose(stats.variance(), var_before)
    stats.update(X[2:])
    np.testing.assert_allclose(stats.mean, X.mean(axis=0))
    np.testing.assert_allclose(stats.variance(), X.var(axis=0, ddof=1))


def test_running_stats_reuses_centered_scratch_for_matching_batches():
    first = np.array([[1.0, 2.0], [2.0, 4.0]], dtype=np.float64)
    second = np.array([[3.0, 8.0], [6.0, 16.0]], dtype=np.float64)
    stats = pp.RunningFeatureStats.empty(2)

    stats.update(first)
    scratch = stats._centered_scratch
    assert scratch is not None

    stats.update(second)
    assert stats._centered_scratch is scratch
    expected = np.vstack([first, second])
    np.testing.assert_allclose(stats.mean, expected.mean(axis=0))
    np.testing.assert_allclose(stats.variance(), expected.var(axis=0, ddof=1))


def test_running_stats_validation():
    with pytest.raises(ValueError):
        pp.RunningFeatureStats.empty(0)
    stats = pp.RunningFeatureStats.empty(2)
    with pytest.raises(ValueError):
        stats.update(np.ones((2, 3), dtype=np.float64))
    with pytest.raises(ValueError):
        stats.update(np.array([[1.0, np.nan]], dtype=np.float64))
    with pytest.raises(ValueError):
        stats.update(np.array([1.0, 2.0], dtype=np.float64))
    roundtrip = pp.RunningFeatureStats.from_dict(stats.as_dict())
    assert roundtrip.n_rows == stats.n_rows
    np.testing.assert_allclose(roundtrip.mean, stats.mean)


def test_preprocessor_fit_and_transform_formula():
    X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float64)
    Xin = X.copy()
    pre = pp.LinearPreprocessor()
    state = pre.fit(X, feature_columns=("a", "b"))
    np.testing.assert_allclose(state.mean, X.mean(axis=0))
    var = X.var(axis=0, ddof=1)
    np.testing.assert_allclose(state.variance, var)
    np.testing.assert_allclose(state.scale, np.sqrt(np.maximum(var, pre.config.variance_floor)))
    Z = pre.transform(X, feature_columns=("a", "b"))
    manual = np.clip((X - state.mean) / state.scale, -pre.config.clip_z, pre.config.clip_z).astype(np.float32)
    np.testing.assert_allclose(Z, manual)
    assert Z.dtype == np.float32
    assert Z.flags.c_contiguous
    np.testing.assert_allclose(X, Xin)


def test_partial_fit_finalize_matches_fit():
    X = np.array([[1.0, 1.0], [2.0, 4.0], [4.0, 16.0], [8.0, 32.0]], dtype=np.float64)
    pre1 = pp.LinearPreprocessor()
    pre1.partial_fit(X[:2], feature_columns=("a", "b"))
    s1 = pre1.finalize()

    pre2 = pp.LinearPreprocessor()
    s2 = pre2.fit(X[:2], feature_columns=("a", "b"))
    np.testing.assert_allclose(s1.mean, s2.mean)
    np.testing.assert_allclose(s1.variance, s2.variance)

    val = X[2:]
    np.testing.assert_allclose(pre1.transform(val), pre2.transform(val))


def test_transform_does_not_update_state():
    X = np.array([[0.0, 1.0], [1.0, 2.0], [2.0, 3.0]], dtype=np.float64)
    pre = pp.LinearPreprocessor()
    state = pre.fit(X, feature_columns=("a", "b"))
    before = (state.n_rows_fit, state.mean.copy(), state.variance.copy(), state.scale.copy(), state.active_mask.copy())
    pre.transform(np.array([[1000.0, -1000.0]], dtype=np.float64))
    pre.transform(np.array([[-900.0, 900.0]], dtype=np.float64))
    after = pre.state
    assert after.n_rows_fit == before[0]
    np.testing.assert_array_equal(after.mean, before[1])
    np.testing.assert_array_equal(after.variance, before[2])
    np.testing.assert_array_equal(after.scale, before[3])
    np.testing.assert_array_equal(after.active_mask, before[4])


def test_feature_column_order_enforced():
    pre = pp.LinearPreprocessor()
    pre.partial_fit(np.ones((3, 2), dtype=np.float64), feature_columns=("a", "b"))
    with pytest.raises(ValueError):
        pre.partial_fit(np.ones((1, 2), dtype=np.float64), feature_columns=("b", "a"))
    pre.finalize()
    with pytest.raises(ValueError):
        pre.transform(np.ones((1, 2), dtype=np.float64), feature_columns=("b", "a"))
    pre.transform(np.ones((1, 2), dtype=np.float64))
    with pytest.raises(ValueError):
        pre2 = pp.LinearPreprocessor()
        pre2.partial_fit(np.ones((1, 2), dtype=np.float64), feature_columns=("a", "a"))
    with pytest.raises(ValueError):
        pp.LinearPreprocessor().partial_fit(np.ones((1, 1), dtype=np.float64), feature_columns=("",))
    with pytest.raises(ValueError):
        pp.LinearPreprocessor().partial_fit(np.ones((1, 1), dtype=np.float64), feature_columns=(1,))


def test_constant_features_shape_preserved_zeroed():
    X = np.array([[2.0, 1.0], [2.0, 2.0], [2.0, 4.0]], dtype=np.float64)
    pre = pp.LinearPreprocessor()
    state = pre.fit(X, feature_columns=("c", "v"))
    assert state.active_mask.tolist() == [False, True]
    Z = pre.transform(X)
    assert Z.shape[1] == 2
    np.testing.assert_allclose(Z[:, 0], 0.0)
    manual_v = np.clip((X[:, 1] - state.mean[1]) / state.scale[1], -pre.config.clip_z, pre.config.clip_z)
    np.testing.assert_allclose(Z[:, 1], manual_v.astype(np.float32))


def test_clip_z_applied():
    X = np.array([[0.0], [1.0], [2.0]], dtype=np.float64)
    pre = pp.LinearPreprocessor(pp.LinearPreprocessConfig(clip_z=1.5))
    pre.fit(X, feature_columns=("x",))
    Z = pre.transform(np.array([[1e6], [-1e6]], dtype=np.float64))
    np.testing.assert_allclose(Z[:, 0], np.array([1.5, -1.5], dtype=np.float32))


def test_state_validation_and_roundtrip():
    X = np.array([[1.0, 2.0], [3.0, 5.0], [8.0, 13.0]], dtype=np.float64)
    pre = pp.LinearPreprocessor()
    state = pre.fit(X, feature_columns=("a", "b"))
    state2 = pp.LinearPreprocessState.from_dict(state.as_dict())
    np.testing.assert_allclose(pre.transform(X), pp.LinearPreprocessor.from_state(state2).transform(X))

    bad = state.as_dict()
    bad["active_mask"] = [False, True]
    with pytest.raises(ValueError):
        pp.LinearPreprocessState.from_dict(bad)

    bad = state.as_dict()
    bad["scale"] = [1.0, 1.0]
    with pytest.raises(ValueError):
        pp.LinearPreprocessState.from_dict(bad)

    bad = state.as_dict()
    bad["n_rows_fit"] = 0
    with pytest.raises(ValueError):
        pp.LinearPreprocessState.from_dict(bad)

    bad = state.as_dict()
    bad["mean"] = [np.nan, 0.0]
    with pytest.raises(ValueError):
        pp.LinearPreprocessState.from_dict(bad)

    bad = state.as_dict()
    bad["variance"] = [-1.0, 1.0]
    with pytest.raises(ValueError):
        pp.LinearPreprocessState.from_dict(bad)


def test_fit_preprocessor_helper_streams_batches():
    batches = [
        np.array([[1.0, 2.0], [2.0, 3.0]], dtype=np.float64),
        np.empty((0, 2), dtype=np.float64),
        np.array([[3.0, 4.0]], dtype=np.float64),
    ]
    state = pp.fit_preprocessor(batches, feature_columns=("a", "b"))
    full = np.vstack([b for b in batches if b.shape[0] > 0])
    state2 = pp.LinearPreprocessor().fit(full, feature_columns=("a", "b"))
    np.testing.assert_allclose(state.mean, state2.mean)
    np.testing.assert_allclose(state.variance, state2.variance)


def test_transform_with_state_helper():
    X = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    pre = pp.LinearPreprocessor()
    state = pre.fit(X, feature_columns=("a", "b"))
    np.testing.assert_allclose(pre.transform(X), pp.transform_with_state(X, state))


def test_no_fit_transform_or_reader_api():
    p = pp.LinearPreprocessor()
    assert not hasattr(p, "fit_transform")
    assert not hasattr(p, "read_split")
    assert not hasattr(p, "transform_table")
    assert not hasattr(p, "partial_fit_reader")
    assert not hasattr(pp, "StandardScaler")
    assert not hasattr(pp, "PCA")


def test_no_future_leakage_or_timestamp_surface():
    src = inspect.getsource(pp)
    forbidden = [
        "local_" + "ts_us",
        "ts_" + "us",
        "event_seq",
        "raw_mid",
        "row_idx",
        "target",
        "label",
        "future_" + "mid",
        "future_" + "ret",
        "shuffle",
        "sort_values",
        "random",
        "fit_transform",
    ]
    lowered = src.lower()
    for token in forbidden:
        assert token.lower() not in lowered


def test_no_old_pipeline_residue():
    src = inspect.getsource(pp)
    forbidden = [
        "BY" + "BIT",
        "CM" + "SSL",
        "offline_" + "ingest",
        "stage" + "1",
        "stage" + "2",
        "stage" + "3",
        "stage" + "4",
        "stage" + "5",
        "PCA",
        "StandardScaler",
        "sklearn",
        "torch",
        "pandas",
        "polars",
        "pyarrow",
    ]
    lowered = src.lower()
    for token in forbidden:
        assert token.lower() not in lowered


def test_vectorized_no_row_loop_smoke():
    src = inspect.getsource(pp)
    for token in (".iterrows", "to_pandas", "for i in range(len(", "for row in"):
        assert token not in src
