import numpy as np
import pytest
import torch

try:
    from test_feature_event_result_contract import _install_optional_dependency_stubs
except Exception:
    _install_optional_dependency_stubs = None

if _install_optional_dependency_stubs is not None:
    _install_optional_dependency_stubs()

from CMSSL17 import NUM_HORIZONS
from CMSSL17_linear import (
    LinearConstantPriorModel,
    build_constant_priors_from_train_labels,
    safe_logit_np,
)


def test_safe_logit_np_clips_extreme_probabilities():
    p = np.array([0.5, 1e-12, 1.0 - 1e-12], dtype=np.float32)
    logits = safe_logit_np(p)
    assert np.isfinite(logits).all()
    assert abs(float(logits[0])) < 1e-6


def test_linear_constant_prior_model_output_schema():
    model = LinearConstantPriorModel(
        dir_logit_prior=np.linspace(-0.1, 0.1, NUM_HORIZONS, dtype=np.float32),
        mag_up_sqrt_prior=np.linspace(0.5, 0.7, NUM_HORIZONS, dtype=np.float32),
        mag_down_sqrt_prior=np.linspace(0.4, 0.6, NUM_HORIZONS, dtype=np.float32),
    )
    x = torch.randn(8, 100, 193)
    pred = model(x)
    assert set(pred.keys()) == {"dir_logits", "mag_up_sqrt", "mag_down_sqrt"}
    for v in pred.values():
        assert v.shape == (8, NUM_HORIZONS)
        assert torch.isfinite(v).all()
    assert torch.all(pred["mag_up_sqrt"] > 0)
    assert torch.all(pred["mag_down_sqrt"] > 0)


@pytest.mark.parametrize(
    "kwargs",
    [
        {
            "dir_logit_prior": np.zeros(NUM_HORIZONS + 1, dtype=np.float32),
            "mag_up_sqrt_prior": np.ones(NUM_HORIZONS, dtype=np.float32),
            "mag_down_sqrt_prior": np.ones(NUM_HORIZONS, dtype=np.float32),
        },
        {
            "dir_logit_prior": np.zeros(NUM_HORIZONS, dtype=np.float32),
            "mag_up_sqrt_prior": np.ones((1, NUM_HORIZONS), dtype=np.float32),
            "mag_down_sqrt_prior": np.ones(NUM_HORIZONS, dtype=np.float32),
        },
        {
            "dir_logit_prior": np.zeros(NUM_HORIZONS, dtype=np.float32),
            "mag_up_sqrt_prior": np.ones(NUM_HORIZONS, dtype=np.float32),
            "mag_down_sqrt_prior": np.ones(NUM_HORIZONS - 1, dtype=np.float32),
        },
    ],
)
def test_linear_constant_prior_model_rejects_invalid_prior_shapes(kwargs):
    with pytest.raises(ValueError):
        LinearConstantPriorModel(**kwargs)


def _prior_stats() -> dict[str, np.ndarray]:
    return {
        "pos_lo_raw_bps": np.zeros(NUM_HORIZONS, dtype=np.float32),
        "pos_hi_raw_bps": np.full(NUM_HORIZONS, 999.0, dtype=np.float32),
        "neg_lo_abs_bps": np.zeros(NUM_HORIZONS, dtype=np.float32),
        "neg_hi_abs_bps": np.full(NUM_HORIZONS, 999.0, dtype=np.float32),
    }


def test_build_constant_priors_uses_precomputed_masks():
    y = np.vstack(
        [
            np.where(np.arange(NUM_HORIZONS) % 2 == 0, 1.0, -1.0),
            np.where(np.arange(NUM_HORIZONS) % 2 == 0, -1.0, 1.0),
            np.where(np.arange(NUM_HORIZONS) % 3 == 0, 3.0, -3.0),
        ]
    ).astype(np.float32)

    keep_pos = y > 0
    keep_neg = y < 0
    keep_signed = keep_pos | keep_neg

    priors = build_constant_priors_from_train_labels(
        y_train=y,
        stats=_prior_stats(),
        mag_up_sqrt_prior=np.ones(NUM_HORIZONS, dtype=np.float32),
        mag_down_sqrt_prior=np.ones(NUM_HORIZONS, dtype=np.float32),
        keep_pos=keep_pos,
        keep_neg=keep_neg,
        keep_signed=keep_signed,
    )

    expected_p_up = keep_pos.sum(axis=0) / keep_signed.sum(axis=0)
    np.testing.assert_allclose(priors["p_up_prior"], expected_p_up.astype(np.float32))


def test_build_constant_priors_rejects_inconsistent_masks():
    y = np.vstack(
        [
            np.ones(NUM_HORIZONS, dtype=np.float32),
            -np.ones(NUM_HORIZONS, dtype=np.float32),
        ]
    )
    keep_pos = y > 0
    keep_neg = y < 0
    keep_signed = keep_pos | keep_neg
    bad_keep_signed = np.ones_like(keep_signed, dtype=bool)
    bad_keep_signed[0, 0] = False

    with pytest.raises(ValueError):
        build_constant_priors_from_train_labels(
            y_train=y,
            stats=_prior_stats(),
            mag_up_sqrt_prior=np.ones(NUM_HORIZONS, dtype=np.float32),
            mag_down_sqrt_prior=np.ones(NUM_HORIZONS, dtype=np.float32),
            keep_pos=keep_pos,
            keep_neg=keep_neg,
            keep_signed=bad_keep_signed,
        )
