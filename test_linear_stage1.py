import numpy as np
import pytest
import torch

from CMSSL17 import NUM_HORIZONS
from CMSSL17_linear import LinearConstantPriorModel, safe_logit_np


def test_safe_logit_np_clips_extreme_probabilities():
    p = np.array([0.5, 1e-12, 1.0 - 1e-12], dtype=np.float32)
    logits = safe_logit_np(p)
    assert np.isfinite(logits).all()
    assert abs(float(logits[0])) < 1e-6


def test_linear_constant_prior_model_output_schema():
    model = LinearConstantPriorModel(
        dir_logit_prior=np.array([0.0, 0.1, -0.1], dtype=np.float32),
        mag_up_sqrt_prior=np.array([0.5, 0.6, 0.7], dtype=np.float32),
        mag_down_sqrt_prior=np.array([0.4, 0.5, 0.6], dtype=np.float32),
    )
    x = torch.randn(8, 100, 193)
    pred = model(x)
    assert set(pred.keys()) == {"dir_logits", "mag_up_sqrt", "mag_down_sqrt"}
    for v in pred.values():
        assert v.shape == (8, 3)
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
