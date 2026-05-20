import numpy as np
import pytest

from CMSSL17_linear import (
    AeonRocketExtractor,
    RawLinearExtractor,
    _fit_rocket_channel_mask,
    _sanitize_aeon_constant_case_channels,
    build_linear_extractor_from_config,
)


def test_channel_mask_drops_structurally_constant_channels():
    rng = np.random.default_rng(0)
    n, t, f = 20, 100, 5
    X = rng.normal(size=(n, t, f)).astype(np.float32)
    X[:, :, 2] = 1.0
    X[:, :, 4] = 0.0
    keep_mask, summary = _fit_rocket_channel_mask(
        X, std_eps=1e-7, max_const_frac=0.995, min_p95_std=1e-7, min_keep_channels=1
    )
    assert keep_mask.tolist() == [True, True, False, True, False]
    assert summary["input_channels"] == 5
    assert summary["kept_channels"] == 3
    assert summary["dropped_channels"] == 2
    assert summary["dropped_channel_indices"] == [2, 4]


def test_locally_constant_case_does_not_drop_channel():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(20, 100, 5)).astype(np.float32)
    X[0, :, 1] = 5.0
    keep_mask, _summary = _fit_rocket_channel_mask(
        X, std_eps=1e-7, max_const_frac=0.995, min_p95_std=1e-7, min_keep_channels=1
    )
    assert bool(keep_mask[1]) is True


def test_constant_fallback_fixes_case_channel_pair():
    rng = np.random.default_rng(2)
    X_aeon = rng.normal(size=(20, 5, 100)).astype(np.float32)
    X_aeon[0, 1, :] = 3.0
    X_fixed, n_fixed = _sanitize_aeon_constant_case_channels(X_aeon, std_eps=1e-7, ramp_eps=1e-6)
    assert n_fixed == 1
    assert float(X_fixed[0, 1, :].std()) > 1e-7


def test_extractor_applies_same_mask_at_transform(monkeypatch):
    class FakeTransformer:
        def __init__(self):
            self.fit_channels = None
            self.transform_channels = None

        def fit(self, X):
            self.fit_channels = int(X.shape[1])
            return self

        def transform(self, X):
            self.transform_channels = int(X.shape[1])
            return np.zeros((X.shape[0], 4), dtype=np.float32)

    fake = FakeTransformer()
    monkeypatch.setattr(AeonRocketExtractor, "_build_transformer", lambda self, X_probe=None: fake)

    rng = np.random.default_rng(3)
    X = rng.normal(size=(20, 100, 5)).astype(np.float32)
    X[:, :, 2] = 1.0
    X[:, :, 4] = -2.0

    ext = AeonRocketExtractor("minirocket", channel_filter_min_keep_channels=1)
    ext.fit(X)
    assert fake.fit_channels == 3

    _ = ext.transform(X)
    assert fake.transform_channels == 3

    with pytest.raises(ValueError, match="input channel mismatch"):
        ext.transform(np.zeros((10, 100, 4), dtype=np.float32))


def test_raw_linear_unaffected():
    rng = np.random.default_rng(4)
    X = rng.normal(size=(8, 100, 5)).astype(np.float32)
    raw = RawLinearExtractor()
    Z = raw.fit_transform(X)
    assert Z.shape[1] == raw.output_dim
    assert raw.name == "raw_linear"


def test_build_linear_extractor_passes_rocket_channel_config():
    cfg = {
        "extractor": "minirocket",
        "n_kernels": 100,
        "hydra_n_kernels": 8,
        "n_groups": 64,
        "n_jobs": 1,
        "random_state": 17,
        "rocket_channel_filter": 1,
        "rocket_channel_filter_std_eps": 2e-7,
        "rocket_channel_filter_max_const_frac": 0.9,
        "rocket_channel_filter_min_p95_std": 3e-7,
        "rocket_channel_filter_min_keep_channels": 2,
        "rocket_constant_fallback": 1,
        "rocket_constant_fallback_eps": 5e-6,
    }
    ext = build_linear_extractor_from_config(cfg)
    assert isinstance(ext, AeonRocketExtractor)
    assert ext.channel_filter_enabled is True
    assert ext.channel_filter_std_eps == 2e-7
    assert ext.channel_filter_max_const_frac == 0.9
    assert ext.channel_filter_min_p95_std == 3e-7
    assert ext.channel_filter_min_keep_channels == 2
    assert ext.constant_fallback_enabled is True
    assert ext.constant_fallback_eps == 5e-6


def test_build_multirocket_hydra_passes_rocket_channel_config():
    cfg = {
        "extractor": "multirocket_hydra",
        "n_kernels": 100,
        "hydra_n_kernels": 8,
        "n_groups": 64,
        "n_jobs": 1,
        "random_state": 17,
        "rocket_channel_filter": 1,
        "rocket_channel_filter_std_eps": 2e-7,
        "rocket_channel_filter_max_const_frac": 0.9,
        "rocket_channel_filter_min_p95_std": 3e-7,
        "rocket_channel_filter_min_keep_channels": 2,
        "rocket_constant_fallback": 1,
        "rocket_constant_fallback_eps": 5e-6,
    }
    ext = build_linear_extractor_from_config(cfg)
    assert isinstance(ext, AeonRocketExtractor)
    combined = ext._build_transformer()
    for child in combined.extractors:
        assert child.channel_filter_enabled is True
        assert child.channel_filter_std_eps == 2e-7
        assert child.channel_filter_max_const_frac == 0.9
        assert child.channel_filter_min_p95_std == 3e-7
        assert child.channel_filter_min_keep_channels == 2
        assert child.constant_fallback_enabled is True
        assert child.constant_fallback_eps == 5e-6
