import numpy as np
import pytest

try:
    from test_feature_event_result_contract import _install_optional_dependency_stubs
except Exception:
    _install_optional_dependency_stubs = None

if _install_optional_dependency_stubs is not None:
    _install_optional_dependency_stubs()

from CMSSL17_linear import (
    AeonRocketExtractor,
    RawLinearExtractor,
    build_linear_extractor_from_config,
    cmssl_windows_to_aeon,
)


def test_raw_linear_extractor_lag_bank_stats_shape():
    N, T, F = 4, 12, 3
    X = np.arange(N * T * F, dtype=np.float32).reshape(N, T, F)
    ext = RawLinearExtractor(
        mode="lag_bank_stats",
        lags=(1, 2, 5),
        windows=(3, 6),
        include_std=True,
        include_slope=False,
    )
    Z = ext.fit_transform(X)
    expected_blocks = 1 + 3 + 2 + 2  # last + deltas + means + stds
    assert Z.shape == (N, expected_blocks * F)
    assert Z.dtype == np.float32
    assert np.isfinite(Z).all()


def test_raw_linear_lag_uses_previous_row_not_current():
    X = np.zeros((1, 5, 1), dtype=np.float32)
    X[0, :, 0] = [10, 20, 30, 40, 50]
    ext = RawLinearExtractor(mode="lag_bank", lags=(1,), windows=(), include_std=False)
    Z = ext.fit_transform(X)
    # blocks: last=50, delta lag1=50-40=10
    assert Z[0, 0] == 50
    assert Z[0, 1] == 10


def test_cmssl_windows_to_aeon_transposes_to_nct():
    X = np.zeros((2, 5, 3), dtype=np.float32)
    Y = cmssl_windows_to_aeon(X)
    assert Y.shape == (2, 3, 5)
    assert Y.flags["C_CONTIGUOUS"]


@pytest.mark.parametrize("name", ["minirocket", "multirocket", "hydra"])
def test_aeon_extractor_smoke(name):
    pytest.importorskip("aeon")
    X = np.random.default_rng(17).normal(size=(8, 32, 4)).astype(np.float32)
    ext = AeonRocketExtractor(
        name=name,
        n_kernels=128,
        hydra_n_kernels=4,
        n_groups=8,
        n_jobs=1,
        random_state=17,
    )
    Z = ext.fit_transform(X)
    assert Z.shape[0] == X.shape[0]
    assert Z.ndim == 2
    assert Z.dtype == np.float32
    assert np.isfinite(Z).all()


def test_multirocket_hydra_smoke():
    pytest.importorskip("aeon")
    X = np.random.default_rng(17).normal(size=(8, 32, 4)).astype(np.float32)
    ext = AeonRocketExtractor(
        name="multirocket_hydra",
        n_kernels=128,
        hydra_n_kernels=4,
        n_groups=8,
        n_jobs=1,
        random_state=17,
    )
    Z = ext.fit_transform(X)
    assert Z.shape[0] == X.shape[0]
    assert Z.ndim == 2
    assert Z.dtype == np.float32
    assert np.isfinite(Z).all()


def test_build_linear_extractor_factory_raw_linear():
    ext = build_linear_extractor_from_config(
        {
            "extractor": "raw_linear",
            "raw_mode": "lag_bank_stats",
            "raw_lags": [1, 2, 5],
            "raw_windows": [3, 6],
            "raw_include_std": True,
            "raw_include_slope": False,
            "n_kernels": 128,
            "hydra_n_kernels": 4,
            "n_groups": 8,
            "n_jobs": 1,
            "random_state": 17,
        }
    )
    assert isinstance(ext, RawLinearExtractor)
