import inspect
import subprocess
import sys

import numpy as np
import pytest

import mmrt.linear.models as lm


def test_public_api_boundary():
    expected = [
        "DEFAULT_MODEL_DTYPE",
        "ALLOWED_MODEL_DTYPES",
        "DEFAULT_LEARNING_RATE",
        "DEFAULT_L2",
        "DEFAULT_MAX_GRAD_NORM",
        "DEFAULT_INIT_SCALE",
        "NO_MOVE_HEAD",
        "DIRECTION_HEAD",
        "MAGNITUDE_UP_HEAD",
        "MAGNITUDE_DOWN_HEAD",
        "MODEL_HEADS",
        "LinearModelConfig",
        "LinearHeadState",
        "BaseLinearHead",
        "NoMoveLinearHead",
        "DirectionLinearHead",
        "MagnitudeLinearHead",
        "LinearModelBundle",
        "make_linear_model_bundle",
        "load_linear_head_state",
        "load_linear_model_bundle",
    ]
    assert lm.__all__ == expected
    assert lm.MODEL_HEADS == ("no_move", "direction", "magnitude_up", "magnitude_down")
    forbidden = ["bybit", "cmssl", "stage", "pca", "sklearn", "torch", "pandas", "polars", "reader", "writer", "storage", "extract", "preprocess", "evaluate"]
    lowered = [name.lower() for name in lm.__all__]
    for needle in forbidden:
        assert all(needle not in name for name in lowered)


def test_no_forbidden_imports():
    code = (
        "import importlib, sys;\n"
        "before=set(sys.modules.keys());\n"
        "importlib.import_module('mmrt.linear.models');\n"
        "after=set(sys.modules.keys())-before;\n"
        "print('\\n'.join(sorted(after)))"
    )
    result = subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True)
    loaded = set(result.stdout.splitlines())
    forbidden = {
        "pandas", "polars", "torch", "sklearn", "scipy", "numba", "pyarrow",
        "mmrt.storage.reader", "mmrt.storage.writer", "mmrt.storage.splits",
        "mmrt.linear.extractors", "mmrt.linear.targets", "mmrt.linear.preprocess",
        "mmrt.features.engine", "mmrt.features.labels", "mmrt.features.transforms",
        "mmrt.data.tardis_csv", "mmrt.data.event_merge", "CMSSL17", "offline_ingest",
    }
    assert loaded.isdisjoint(forbidden)


def test_config_validation():
    cfg = lm.LinearModelConfig()
    assert cfg.learning_rate == 0.05
    assert cfg.l2 == 1e-4
    assert cfg.max_grad_norm == 10.0
    assert cfg.output_dtype == "float32"
    assert lm.LinearModelConfig(output_dtype="float64").dtype == np.dtype("float64")
    for bad in [0.0, -1.0, np.nan, np.inf, True]:
        with pytest.raises(ValueError):
            lm.LinearModelConfig(learning_rate=bad)
    for bad in [-1.0, np.nan, np.inf, True]:
        with pytest.raises(ValueError):
            lm.LinearModelConfig(l2=bad)
    lm.LinearModelConfig(l2=0.0)
    for bad in [0.0, -1.0, np.nan, np.inf, True]:
        with pytest.raises(ValueError):
            lm.LinearModelConfig(max_grad_norm=bad)
    with pytest.raises(ValueError):
        lm.LinearModelConfig(output_dtype="bad")
    for attr in ["solver", "random_state", "class_weight", "epochs", "batch_size", "stage", "no_move"]:
        assert not hasattr(cfg, attr)


def test_head_initialization_and_state():
    head = lm.DirectionLinearHead(("a", "b"))
    assert head.weights.shape == (2,)
    assert np.allclose(head.weights, 0.0)
    assert head.intercept == 0.0
    assert head.n_updates == 0
    assert head.n_rows_seen == 0
    assert not head.is_fitted()
    state = head.state()
    roundtrip = lm.LinearHeadState.from_dict(state.as_dict())
    assert roundtrip.head_name == state.head_name
    assert roundtrip.feature_columns == state.feature_columns
    assert np.allclose(roundtrip.weights, state.weights)
    with pytest.raises(ValueError):
        lm.DirectionLinearHead(("a", "a"))
    with pytest.raises(ValueError):
        lm.DirectionLinearHead(("",))
    with pytest.raises(ValueError):
        lm.DirectionLinearHead((1,))


def test_direction_partial_fit_updates_and_reduces_loss():
    X = np.array([[-2.0], [-1.0], [1.0], [2.0]])
    y = np.array([0, 0, 1, 1])
    head = lm.DirectionLinearHead(("x",), config=lm.LinearModelConfig(learning_rate=0.2, l2=0.0))
    before = head.loss(X, y)
    for _ in range(50):
        head.partial_fit(X, y)
    after = head.loss(X, y)
    assert after < before
    proba = head.predict_proba(X)
    assert proba.shape == (4, 2)
    assert np.allclose(np.sum(proba, axis=1), 1.0, atol=1e-6)
    assert np.array_equal(head.predict(X), np.array([0, 0, 1, 1], dtype=np.int8))
    assert head.n_updates == 50
    assert head.n_rows_seen == 200


def test_direction_validates_labels():
    X = np.array([[0.0], [1.0]])
    head = lm.DirectionLinearHead(("x",))
    with pytest.raises(ValueError):
        head.partial_fit(X, np.array([-1, 1]))
    with pytest.raises(ValueError):
        head.partial_fit(X, np.array([0, 2]))
    with pytest.raises(ValueError):
        head.partial_fit(X, np.array([0]))
    with pytest.raises(ValueError):
        head.partial_fit(np.array([[np.nan], [1.0]]), np.array([0, 1]))
    before = (head.n_updates, head.n_rows_seen)
    head.partial_fit(np.empty((0, 1)), np.empty((0,)))
    assert (head.n_updates, head.n_rows_seen) == before


def test_magnitude_partial_fit_updates_and_reduces_loss():
    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = np.array([0.0, 0.5, 1.0, 1.5])
    head = lm.MagnitudeLinearHead(lm.MAGNITUDE_UP_HEAD, ("x",), config=lm.LinearModelConfig(learning_rate=0.1, l2=0.0))
    before = head.loss(X, y)
    for _ in range(100):
        head.partial_fit(X, y)
    after = head.loss(X, y)
    assert after < before
    raw = head.predict_raw(X)
    assert raw.shape == (4,)
    assert np.all(head.predict_nonnegative(X) >= 0.0)


def test_magnitude_validates_targets():
    X = np.array([[0.0], [1.0]])
    head = lm.MagnitudeLinearHead(lm.MAGNITUDE_UP_HEAD, ("x",))
    with pytest.raises(ValueError):
        head.partial_fit(X, np.array([-0.1, 0.2]))
    with pytest.raises(ValueError):
        head.partial_fit(X, np.array([np.nan, 0.2]))
    with pytest.raises(ValueError):
        head.partial_fit(X, np.array([0.1]))
    with pytest.raises(ValueError):
        lm.MagnitudeLinearHead("bad", ("x",))
    before = (head.n_updates, head.n_rows_seen)
    head.partial_fit(np.empty((0, 1)), np.empty((0,)))
    assert (head.n_updates, head.n_rows_seen) == before


def test_gradient_clipping_keeps_params_finite():
    X = np.array([[1e6, -1e6], [-1e6, 1e6]], dtype=np.float64)
    y = np.array([1, 0])
    cfg = lm.LinearModelConfig(learning_rate=1.0, l2=0.0, max_grad_norm=0.1)
    head = lm.DirectionLinearHead(("a", "b"), cfg)
    old_w = head.weights.copy()
    old_b = head.intercept
    head.partial_fit(X, y)
    assert np.isfinite(head.weights).all() and np.isfinite(head.intercept)
    delta = np.concatenate([head.weights - old_w, np.array([head.intercept - old_b])])
    assert np.linalg.norm(delta) <= 0.11


def test_l2_regularizes_weights_not_intercept_smoke():
    cfg = lm.LinearModelConfig(learning_rate=0.1, l2=1.0)
    head = lm.MagnitudeLinearHead(lm.MAGNITUDE_UP_HEAD, ("x",), cfg)
    head.weights[:] = 2.0
    head.intercept = 3.0
    X = np.zeros((8, 1))
    y = np.full(8, 3.0)
    head.partial_fit(X, y)
    assert head.weights[0] < 2.0
    assert abs(head.intercept - 3.0) < 1e-8


def test_bundle_predictions_and_validation():
    bundle = lm.make_linear_model_bundle(("a", "b"))
    assert isinstance(bundle.no_move, lm.NoMoveLinearHead)
    assert isinstance(bundle.direction, lm.DirectionLinearHead)
    assert isinstance(bundle.magnitude_up, lm.MagnitudeLinearHead)
    assert isinstance(bundle.magnitude_down, lm.MagnitudeLinearHead)
    assert bundle.feature_columns_by_head[lm.DIRECTION_HEAD] == ("a", "b")
    assert bundle.heads_share_feature_columns()
    assert bundle.n_features == 2
    out = bundle.predict(np.zeros((3, 2)))
    assert set(out) == {"no_move_proba", "no_move_pred", "direction_proba", "direction_pred", "magnitude_up", "magnitude_down"}
    assert out["direction_proba"].shape == (3, 2)
    assert out["direction_pred"].shape == (3,)
    assert out["magnitude_up"].shape == (3,)
    assert out["magnitude_down"].shape == (3,)
    assert bundle.direction.config == bundle.magnitude_up.config == bundle.magnitude_down.config
    with pytest.raises(ValueError):
        lm.LinearModelBundle(
            direction=lm.DirectionLinearHead(("a",)),
            magnitude_up=lm.MagnitudeLinearHead(lm.MAGNITUDE_UP_HEAD, ("a",)),
            magnitude_down=lm.MagnitudeLinearHead(lm.MAGNITUDE_DOWN_HEAD, ("b",)),
        )


def test_bundle_n_features_rejects_nonshared_feature_columns():
    bundle = lm.make_linear_model_bundle(
        {
            lm.DIRECTION_HEAD: ("x_a", "x_b"),
            lm.MAGNITUDE_UP_HEAD: ("x_a",),
            lm.MAGNITUDE_DOWN_HEAD: ("x_b",),
        }
    )

    with pytest.raises(ValueError, match="only defined when all heads share"):
        _ = bundle.n_features


def test_bundle_serialization_roundtrip():
    bundle = lm.make_linear_model_bundle(("x",))
    X = np.array([[-1.0], [1.0]])
    bundle.direction.partial_fit(X, np.array([0, 1]))
    bundle.magnitude_up.partial_fit(X, np.array([0.0, 1.0]))
    bundle.magnitude_down.partial_fit(X, np.array([1.0, 0.0]))
    dct = bundle.as_dict()
    loaded = lm.load_linear_model_bundle(dct)
    p1 = bundle.predict(X)
    p2 = loaded.predict(X)
    for key in p1:
        assert np.allclose(p1[key], p2[key])


def test_decision_function_dtype_and_shape():
    X = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    X_before = X.copy()
    h32 = lm.DirectionLinearHead(("a", "b"))
    y32 = h32.decision_function(X)
    assert y32.shape == (2,)
    assert y32.dtype == np.float32
    h64 = lm.DirectionLinearHead(("a", "b"), lm.LinearModelConfig(output_dtype="float64"))
    y64 = h64.decision_function(X)
    assert y64.dtype == np.float64
    with pytest.raises(ValueError):
        h32.decision_function(np.array([[1.0]]))
    with pytest.raises(ValueError):
        h32.decision_function(np.array([[np.inf, 0.0]]))
    assert np.array_equal(X, X_before)


def test_no_fit_train_evaluate_or_io_api():
    forbidden = ["fit", "fit_transform", "train", "evaluate", "save", "load", "read_split", "transform_table", "partial_fit_reader", "StandardScaler", "PCA", "LogisticRegression", "SGDClassifier", "Torch"]
    for name in forbidden:
        if name == "load":
            continue
        assert not hasattr(lm, name)
    for cls in [lm.DirectionLinearHead, lm.MagnitudeLinearHead, lm.LinearModelBundle]:
        assert not hasattr(cls, "fit")
        assert not hasattr(cls, "train")
        assert not hasattr(cls, "evaluate")
        assert not hasattr(cls, "save")


def test_no_future_leakage_or_timestamp_surface():
    source = inspect.getsource(lm)
    forbidden = [
        "local_" + "ts_us", "ts_" + "us", "event_seq", "raw_mid", "row_idx",
        "future_" + "mid", "future_" + "ret", "shuffle", "sort_values", "random",
        "target_column", "label", "timestamp",
    ]
    for token in forbidden:
        assert token not in source


def test_no_old_pipeline_residue():
    source = inspect.getsource(lm)
    forbidden = [
        "BY" + "BIT", "CM" + "SSL", "offline_" + "ingest", "stage" + "1", "stage" + "2",
        "stage" + "3", "stage" + "4", "stage" + "5", "Mini" + "Rocket", "Multi" + "Rocket",
        "Hy" + "dra", "Ae" + "on", "sklearn", "torch", "pandas", "polars", "pyarrow", "PCA", "StandardScaler",
    ]
    for token in forbidden:
        assert token not in source


def test_vectorized_no_row_loop_smoke():
    source = inspect.getsource(lm)
    forbidden = [".iterrows", "to_pandas", "for i in range(len(", "for row in", "for sample in"]
    for token in forbidden:
        assert token not in source
