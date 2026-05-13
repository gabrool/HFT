import os, math, copy, json, csv, zipfile, io, gzip, contextlib, time, heapq, hashlib
from pathlib import Path
from collections import deque
from bisect import bisect_left, bisect_right, insort
from decimal import Decimal, ROUND_HALF_EVEN, InvalidOperation
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from enum import IntEnum
from datetime import datetime, timezone
from typing import Deque, Any, List, Dict, Tuple, Generator, Optional, Iterable, Union, Sequence
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import math
from einops import rearrange, repeat
import torch._functorch.config as ft_config

try:
    from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
except ImportError:
    causal_conv1d_fn, causal_conv1d_update = None, None

try:
    from causal_conv1d.causal_conv1d_varlen import causal_conv1d_varlen_states
except ImportError:
    causal_conv1d_varlen_states = None

try:
    from mamba_ssm.ops.triton.selective_state_update import selective_state_update
except ImportError:
    selective_state_update = None

from mamba_ssm.ops.triton.layernorm_gated import RMSNorm as RMSNormGated

from mamba_ssm.distributed.tensor_parallel import ColumnParallelLinear, RowParallelLinear
from mamba_ssm.distributed.distributed_utils import all_reduce, reduce_scatter

from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined
from mamba_ssm.ops.triton.ssd_combined import mamba_split_conv1d_scan_combined

from huggingface_hub import PyTorchModelHubMixin

# NOTE FOR CONTRIBUTORS:
# This file is a library module. Offline dataset creation is implemented in
# offline_ingest.py. Do not add ingestion scripts here.

ft_config.donated_buffer = False
torch.cuda.empty_cache()

# ==============================  MAMBA2  ==============================
class Mamba2(nn.Module, PyTorchModelHubMixin):
    def __init__(
        self,
        d_model,
        d_state=128,
        d_conv=4,
        conv_init=None,
        expand=2,
        headdim=64,
        d_ssm=None,
        ngroups=1,
        A_init_range=(1, 16),
        D_has_hdim=False,
        rmsnorm=True,
        norm_before_gate=False,
        dt_min=0.001,
        dt_max=0.1,
        dt_init_floor=1e-4,
        dt_limit=(0.0, float("inf")),
        bias=False,
        conv_bias=True,
        chunk_size=256,
        use_mem_eff_path=True,
        layer_idx=None,
        process_group=None,
        sequence_parallel=True,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.conv_init = conv_init
        self.expand = expand
        self.process_group = process_group
        self.sequence_parallel = sequence_parallel
        self.world_size = 1 if process_group is None else process_group.size()
        self.local_rank = 0 if process_group is None else process_group.rank()
        self.d_inner = (self.expand * self.d_model) // self.world_size
        assert self.d_inner * self.world_size == self.expand * self.d_model
        self.headdim = headdim
        self.d_ssm = self.d_inner if d_ssm is None else d_ssm // self.world_size
        assert ngroups % self.world_size == 0
        self.ngroups = ngroups // self.world_size
        assert self.d_ssm % self.headdim == 0
        self.nheads = self.d_ssm // self.headdim
        self.D_has_hdim = D_has_hdim
        self.rmsnorm = rmsnorm
        self.norm_before_gate = norm_before_gate
        self.dt_limit = dt_limit
        self.activation = "silu"
        self.chunk_size = chunk_size
        self.use_mem_eff_path = True
        self.layer_idx = layer_idx

        d_in_proj = 2 * self.d_inner + 2 * self.ngroups * self.d_state + self.nheads
        if self.process_group is None:
            self.in_proj = nn.Linear(self.d_model, d_in_proj, bias=bias, **factory_kwargs)
        else:
            self.in_proj = ColumnParallelLinear(self.d_model, d_in_proj * self.world_size, bias=bias,
                                                process_group=self.process_group, sequence_parallel=self.sequence_parallel,
                                                **factory_kwargs)

        conv_dim = self.d_ssm + 2 * self.ngroups * self.d_state
        self.conv1d = nn.Conv1d(
            in_channels=conv_dim,
            out_channels=conv_dim,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=conv_dim,
            padding=d_conv - 1,
            **factory_kwargs,
        )
        if self.conv_init is not None:
            nn.init.uniform_(self.conv1d.weight, -self.conv_init, self.conv_init)

        self.act = nn.SiLU()

        dt = torch.exp(
            torch.rand(self.nheads, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        )
        dt = torch.clamp(dt, min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        self.dt_bias = nn.Parameter(inv_dt)
        self.dt_bias._no_weight_decay = True

        A = torch.empty(self.nheads, dtype=torch.float32, device=device).uniform_(*A_init_range)
        A_log = torch.log(A).to(dtype=dtype)
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True

        self.D = nn.Parameter(torch.ones(self.d_ssm if self.D_has_hdim else self.nheads, device=device))
        self.D._no_weight_decay = True

        if self.rmsnorm:
            self.norm = RMSNormGated(self.d_ssm, eps=1e-5, norm_before_gate=self.norm_before_gate,
                                     group_size=self.d_ssm // ngroups, **factory_kwargs)

        if self.process_group is None:
            self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        else:
            self.out_proj = RowParallelLinear(self.d_inner * self.world_size, self.d_model, bias=bias,
                                              process_group=self.process_group, sequence_parallel=self.sequence_parallel,
                                              **factory_kwargs)

    def _pure_chunk_scan(self, x, dt, A, B, C, D, z, dt_bias, dt_softplus=True, dt_limit=(0.0, float("inf"))):
        b, l, h, p = x.shape
        n = self.d_state
        g = self.ngroups
        if g != 1:
            raise NotImplementedError("Pure PyTorch only supports ngroups=1")

        if self.D_has_hdim:
            D = rearrange(D, "(h p) -> h p", p=p)
        else:
            D = rearrange(D, "h -> h 1")

        B = B[:, :, 0, :]
        C = C[:, :, 0, :]

        if dt_softplus:
            dt = F.softplus(dt + dt_bias)
        dt = dt.clamp(min=dt_limit[0], max=dt_limit[1])

        state = torch.zeros(b, h, p, n, device=x.device, dtype=x.dtype)
        y = torch.zeros_like(x)

        for t in range(l):
            xt = x[:, t]
            dtt = dt[:, t]
            Bt = B[:, t]
            Ct = C[:, t]
            dA = torch.exp(dtt * A)
            dBx = dtt.unsqueeze(-1).unsqueeze(-1) * xt.unsqueeze(-1) * Bt.unsqueeze(1).unsqueeze(1)
            state = state * dA.unsqueeze(-1).unsqueeze(-1) + dBx
            yt = torch.einsum("bhpn,bn->bhp", state, Ct) + D * xt
            y[:, t] = yt

        if z is not None:
            y = y * self.act(z)

        return y

    def forward(self, u, seqlen=None, seq_idx=None, cu_seqlens=None, inference_params=None):
        seqlen_og = seqlen
        if seqlen is None:
            batch, seqlen, dim = u.shape
        else:
            batch_seqlen, dim = u.shape
            batch = batch_seqlen // seqlen

        conv_state, ssm_state = None, None
        if inference_params is not None:
            inference_batch = cu_seqlens.shape[0] - 1 if cu_seqlens is not None else batch
            conv_state, ssm_state = self._get_states_from_cache(inference_params, inference_batch)
            if inference_params.seqlen_offset > 0:
                out, _, _ = self.step(u, conv_state, ssm_state)
                return out

        zxbcdt = self.in_proj(u)
        if seqlen_og is not None:
            zxbcdt = rearrange(zxbcdt, "(b l) d -> b l d", l=seqlen)
        A = -torch.exp(self.A_log.float())
        dt_limit_kwargs = {} if self.dt_limit == (0.0, float("inf")) else dict(dt_limit=self.dt_limit)
        if self.use_mem_eff_path and inference_params is None:
            out = mamba_split_conv1d_scan_combined(
                zxbcdt,
                rearrange(self.conv1d.weight, "d 1 w -> d w"),
                self.conv1d.bias,
                self.dt_bias,
                A,
                D=rearrange(self.D, "(h p) -> h p", p=self.headdim) if self.D_has_hdim else self.D,
                chunk_size=self.chunk_size,
                seq_idx=seq_idx,
                activation=self.activation,
                rmsnorm_weight=self.norm.weight if self.rmsnorm else None,
                rmsnorm_eps=self.norm.eps if self.rmsnorm else 1e-6,
                outproj_weight=self.out_proj.weight,
                outproj_bias=self.out_proj.bias,
                headdim=None if self.D_has_hdim else self.headdim,
                ngroups=self.ngroups,
                norm_before_gate=self.norm_before_gate,
                **dt_limit_kwargs,
            )
            if seqlen_og is not None:
                out = rearrange(out, "b l d -> (b l) d")
            if self.process_group is not None:
                reduce_fn = reduce_scatter if self.sequence_parallel else all_reduce
                out = reduce_fn(out, self.process_group)
        else:
            d_mlp = (zxbcdt.shape[-1] - 2 * self.d_ssm - 2 * self.ngroups * self.d_state - self.nheads) // 2
            z0, x0, z, xBC, dt = torch.split(
                zxbcdt,
                [d_mlp, d_mlp, self.d_ssm, self.d_ssm + 2 * self.ngroups * self.d_state, self.nheads],
                dim=-1
            )
            if conv_state is not None:
                if cu_seqlens is None:
                    xBC_t = rearrange(xBC, "b l d -> b d l")
                    conv_state.copy_(F.pad(xBC_t, (self.d_conv - xBC_t.shape[-1], 0)))
                else:
                    assert causal_conv1d_varlen_states is not None, "varlen inference requires causal_conv1d package"
                    assert batch == 1, "varlen inference only supports batch dimension 1"
                    conv_varlen_states = causal_conv1d_varlen_states(
                        xBC.squeeze(0), cu_seqlens, state_len=conv_state.shape[-1]
                    )
                    conv_state.copy_(conv_varlen_states)
            assert self.activation in ["silu", "swish"]
            if causal_conv1d_fn is None or self.activation not in ["silu", "swish"]:
                assert seq_idx is None, "varlen conv1d requires the causal_conv1d package"
                xBC = self.act(
                    self.conv1d(xBC.transpose(1, 2)).transpose(1, 2)[:, :-(self.d_conv - 1)]
                )
            else:
                xBC = causal_conv1d_fn(
                    xBC.transpose(1, 2),
                    rearrange(self.conv1d.weight, "d 1 w -> d w"),
                    bias=self.conv1d.bias,
                    activation=self.activation,
                    seq_idx=seq_idx,
                ).transpose(1, 2)
            x, B, C = torch.split(xBC, [self.d_ssm, self.ngroups * self.d_state, self.ngroups * self.d_state], dim=-1)
            y = self._pure_chunk_scan(
                rearrange(x, "b l (h p) -> b l h p", p=self.headdim),
                dt,
                A,
                rearrange(B, "b l (g n) -> b l g n", g=self.ngroups),
                rearrange(C, "b l (g n) -> b l g n", g=self.ngroups),
                rearrange(self.D, "(h p) -> h p", p=self.headdim) if self.D_has_hdim else self.D,
                rearrange(z, "b l (h p) -> b l h p", p=self.headdim) if not self.rmsnorm else None,
                self.dt_bias,
                True,
                self.dt_limit,
            )
            if ssm_state is not None:
                y, last_state, *rest = y
                if cu_seqlens is None:
                    ssm_state.copy_(last_state)
                else:
                    varlen_states = rest[0]
                    ssm_state.copy_(varlen_states)
            y = rearrange(y, "b l h p -> b l (h p)")
            if self.rmsnorm:
                y = self.norm(y, z)
            if d_mlp > 0:
                y = torch.cat([F.silu(z0) * x0, y], dim=-1)
            if seqlen_og is not None:
                y = rearrange(y, "b l d -> (b l) d")
            out = self.out_proj(y)
        return out

    def step(self, hidden_states, conv_state, ssm_state):
        dtype = hidden_states.dtype
        assert hidden_states.shape[1] == 1, "Only support decoding with 1 token at a time for now"
        zxbcdt = self.in_proj(hidden_states.squeeze(1))
        d_mlp = (zxbcdt.shape[-1] - 2 * self.d_ssm - 2 * self.ngroups * self.d_state - self.nheads) // 2
        z0, x0, z, xBC, dt = torch.split(
            zxbcdt,
            [d_mlp, d_mlp, self.d_ssm, self.d_ssm + 2 * self.ngroups * self.d_state, self.nheads],
            dim=-1
        )

        if causal_conv1d_update is None:
            conv_state.copy_(torch.roll(conv_state, shifts=-1, dims=-1))
            conv_state[:, :, -1] = xBC
            xBC = torch.sum(conv_state * rearrange(self.conv1d.weight, "d 1 w -> d w"), dim=-1)
            if self.conv1d.bias is not None:
                xBC = xBC + self.conv1d.bias
            xBC = self.act(xBC).to(dtype=dtype)
        else:
            xBC = causal_conv1d_update(
                xBC,
                conv_state,
                rearrange(self.conv1d.weight, "d 1 w -> d w"),
                self.conv1d.bias,
                self.activation,
            )

        x, B, C = torch.split(xBC, [self.d_ssm, self.ngroups * self.d_state, self.ngroups * self.d_state], dim=-1)
        A = -torch.exp(self.A_log.float())

        if selective_state_update is None:
            assert self.ngroups == 1, "Only support ngroups=1 for this inference code path"
            dt = F.softplus(dt + self.dt_bias.to(dtype=dt.dtype))
            dA = torch.exp(dt * A)
            x = rearrange(x, "b (h p) -> b h p", p=self.headdim)
            dBx = torch.einsum("bh,bn,bhp->bhpn", dt, B, x)
            ssm_state.copy_(ssm_state * rearrange(dA, "b h -> b h 1 1") + dBx)
            y = torch.einsum("bhpn,bn->bhp", ssm_state.to(dtype), C)
            y = y + rearrange(self.D.to(dtype), "h -> h 1") * x
            y = rearrange(y, "b h p -> b (h p)")
            if not self.rmsnorm:
                y = y * self.act(z)
        else:
            A = repeat(A, "h -> h p n", p=self.headdim, n=self.d_state).to(dtype=torch.float32)
            dt = repeat(dt, "b h -> b h p", p=self.headdim)
            dt_bias = repeat(self.dt_bias, "h -> h p", p=self.headdim)
            D = repeat(self.D, "h -> h p", p=self.headdim)
            B = rearrange(B, "b (g n) -> b g n", g=self.ngroups)
            C = rearrange(C, "b (g n) -> b g n", g=self.ngroups)
            x_reshaped = rearrange(x, "b (h p) -> b h p", p=self.headdim)
            if not self.rmsnorm:
                z = rearrange(z, "b (h p) -> b h p", p=self.headdim)
            y = selective_state_update(
                ssm_state, x_reshaped, dt, A, B, C, D, z=z if not self.rmsnorm else None,
                dt_bias=dt_bias, dt_softplus=True
            )
            y = rearrange(y, "b h p -> b (h p)")
        if self.rmsnorm:
            y = self.norm(y, z)
        if d_mlp > 0:
            y = torch.cat([F.silu(z0) * x0, y], dim=-1)
        out = self.out_proj(y)
        return out.unsqueeze(1), conv_state, ssm_state

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        device = self.out_proj.weight.device
        conv_dtype = self.conv1d.weight.dtype if dtype is None else dtype
        conv_state = torch.zeros(
            batch_size, self.d_conv, self.conv1d.weight.shape[0], device=device, dtype=conv_dtype
        ).transpose(1, 2)
        ssm_dtype = self.in_proj.weight.dtype if dtype is None else dtype
        ssm_state = torch.zeros(
            batch_size, self.nheads, self.headdim, self.d_state, device=device, dtype=ssm_dtype
        )
        return conv_state, ssm_state

    def _get_states_from_cache(self, inference_params, batch_size, initialize_states=False):
        assert self.layer_idx is not None
        if self.layer_idx not in inference_params.key_value_memory_dict:
            batch_shape = (batch_size,)
            conv_state = torch.zeros(
                batch_size,
                self.d_conv,
                self.conv1d.weight.shape[0],
                device=self.conv1d.weight.device,
                dtype=self.conv1d.weight.dtype,
            ).transpose(1, 2)
            ssm_state = torch.zeros(
                batch_size,
                self.nheads,
                self.headdim,
                self.d_state,
                device=self.in_proj.weight.device,
                dtype=self.in_proj.weight.dtype,
            )
            inference_params.key_value_memory_dict[self.layer_idx] = (conv_state, ssm_state)
        else:
            conv_state, ssm_state = inference_params.key_value_memory_dict[self.layer_idx]
            if initialize_states:
                conv_state.zero_()
                ssm_state.zero_()
        return conv_state, ssm_state



_TS_SECONDS_THRESHOLD = Decimal("1e12")
_TS_MILLI_SCALE = Decimal("1000")


def timestamp_to_ms_half_even(value: Union[int, float, str]) -> int:
    """Convert timestamp-like values to integer milliseconds.

    Policy:
      - Values with absolute magnitude < 1e12 are interpreted as seconds and
        scaled by 1000.
      - Larger magnitudes are interpreted as millisecond-like values.
      - Conversion to integer milliseconds uses bankers rounding
        (ROUND_HALF_EVEN) so .5 ties are deterministic and unbiased.
    """
    if value is None:
        raise ValueError("Timestamp value is missing (None)")

    if isinstance(value, bool):
        raise ValueError(f"Unparseable timestamp value: {value!r}")

    if isinstance(value, str):
        value = value.strip()
        if not value:
            raise ValueError("Timestamp value is missing (empty string)")

    try:
        if isinstance(value, str):
            numeric = Decimal(value)
        elif isinstance(value, int):
            numeric = Decimal(value)
        elif isinstance(value, float):
            if not math.isfinite(value):
                raise ValueError(f"Unparseable timestamp value: {value!r}")
            numeric = Decimal(str(value))
        else:
            numeric = Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError) as exc:
        raise ValueError(f"Unparseable timestamp value: {value!r}") from exc

    if not numeric.is_finite():
        raise ValueError(f"Unparseable timestamp value: {value!r}")

    scaled = numeric * _TS_MILLI_SCALE if abs(numeric) < _TS_SECONDS_THRESHOLD else numeric
    return int(scaled.to_integral_value(rounding=ROUND_HALF_EVEN))


def coerce_ts_ms(value: Union[int, float, str]) -> int:
    """Convert a timestamp-like value to integer milliseconds."""
    return timestamp_to_ms_half_even(value)



# ---------------------------  Core hyper-params  ---------------------------
LOOKBACK        = 100        # canonical event-time token lookback
WINDOW_MS       = 10_000     # canonical rolling window span (10s)
PAD_DT_FOR_LEFT = 0.0
BATCH_SIZE      = 512
DMODEL          = 1024
MAMBA_LAYERS    = 2
DEPATCH_OFFSET_MODE = "learnable"
ACT_DIAG_P95_MAX_ELEMS = max(
    1024,
    int(os.environ.get("BYBIT_ACT_DIAG_P95_MAX_ELEMS", "1000000")),
)
print(f"[model-diag-config-model] act_p95_max_elems={ACT_DIAG_P95_MAX_ELEMS}", flush=True)

MODEL_ARCH_SCHEMA = "cmssl17_1s_maker_old_ctn6ci_linearproj_no_gate_no_mixed_v1"
# ConvTimeNet topology intentionally matches original 1s path:
# Depatch -> 6 CI ConvEncoder layers -> per-patch flatten(F*C) -> linear final_proj -> [B, patch_count, DMODEL].
# No gate, no MLP projection, no mixed ConvTimeNet stack.
CTN_CI_KERNELS = [3, 3, 5, 5, 7, 7]
CTN_MIXED_KERNELS = []
CTN_CI_INTERNAL_DIM = 8
CTN_CI_FF_MULT = 8
CTN_CI_DFF = CTN_CI_INTERNAL_DIM * CTN_CI_FF_MULT
CTN_POST_GATE_HIDDEN = 2 * DMODEL
CTN_MIXED_DIM = DMODEL
CTN_MIXED_DFF = 2 * DMODEL
CTN_PATCH_SIZE = 2
CTN_PATCH_STRIDE = 1

# Prediction horizons (in milliseconds)
HORIZONS_MS     = [200, 500, 1000]
NUM_HORIZONS    = len(HORIZONS_MS)
HORIZON_WEIGHTS = [0.25, 0.5, 1.0]

LOW_ABS_TRIM_FRACTION = 0.02
HIGH_ABS_TRIM_FRACTION = 0.02
TARGET_TRANSFORM = "raw_signed_bps_to_direction_and_conditional_abs_sqrt_bps"
TARGET_TASK = "direction_and_conditional_magnitude_raw_bps_targets"
LABEL_TRIM_SCHEMA = "signed_nonzero_per_side_quantile_v1"
FEATURE_SCHEMA = "cmssl17_1s_maker_rtcore_v5_raw_no_" + "p" + "ca" + "_pruned245_xformv2"
FEATURE_TRANSFORM = "feature_transform_spec_v2_pruned235"
FEATURE_TRANSFORM_POLICY = "deterministic_transform_plus_selective_causal_preupdate_ewma_v1"
FEATURE_TRANSFORM_WARMUP_ROWS = 50
FEATURE_TRANSFORM_OUTPUT_CLIP_DEFAULT = 8.0
FEATURE_TRANSFORM_SPEC_VERSION = "feature_transform_spec_v2_pruned235"
AUX_SCHEMA = "cmssl17_aux_ob_decision_density_1s_v1"
CHECKPOINT_SCHEMA = "cmssl17-dir-mag-v1-1s-maker-rtcore-raw-no-" + "p" + "ca" + "-pruned245-xformv2"

FOUR_WEEK_PROTOCOL = "four_week_cmssl_val_test_rl_eval_v2"
FIVE_WEEK_PROTOCOL = "five_week_cmssl2w_val_test_rl_eval_v1"
CMSSL_TRAIN_VAL_PROTOCOL = "cmssl_train_val_only_v1"
CMSSL_TRAIN_VAL_TEST_PROTOCOL = "cmssl_train_val_test_only_v1"
FULL_SPLIT_PROTOCOLS = {FOUR_WEEK_PROTOCOL, FIVE_WEEK_PROTOCOL}
CMSSL_ONLY_SPLIT_PROTOCOLS = {CMSSL_TRAIN_VAL_PROTOCOL, CMSSL_TRAIN_VAL_TEST_PROTOCOL}
SUPPORTED_SPLIT_PROTOCOLS = FULL_SPLIT_PROTOCOLS | CMSSL_ONLY_SPLIT_PROTOCOLS
EPOCHS          = 200
LR              = 4e-4
CLIP_GRAD       = 10000
PATIENCE        = 15
# Primary metric config (used for checkpointing + early stopping)
PRIMARY_METRIC = "dir_auc_kept_1000ms"
PRIMARY_METRIC_HORIZON_MS = 1000
PRIMARY_DIR_BAL_ACC_GUARD = 0.505
MODEL_OUTPUT_SCHEMA = "dir_logits_mag_up_down_sqrt_v1"
MAG_SQRT_EPS = 1e-6


def inverse_softplus_np(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    x = np.maximum(x, eps)
    # Stable inverse softplus: for large x, log(expm1(x)) ~= x.
    out = x.copy()
    small = x <= 20.0
    out[small] = np.log(np.expm1(x[small]))
    return out.astype(np.float32)


def inverse_softplus_torch(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    x = torch.clamp(x, min=eps)
    return torch.where(x > 20.0, x, torch.log(torch.expm1(x)))
DIR_LOSS_WEIGHT = 1.00
MAG_LOSS_WEIGHT = 0.00
MAG_CORR_LOSS_WEIGHT = 0.00
SINGLE_WEEK_PATIENCE = 1
# Number of auxiliary channels appended after the raw core feature vector.
AUX_DIM        = 6
FEATURE_AUX_TAIL = (
    "log_dt_decision_ms",
    "log_events_100ms",
    "log_events_200ms",
    "log_events_500ms",
    "log_events_1000ms",
    "log_events_3000ms",
)


PRICE_WINDOWS_MS = (
    200,
    500,
    1_000,
)

FAST_WINDOWS_MS = (
    200,
    500,
    1_000,
)

FLOW_WINDOWS_MS = (
    200,
    500,
    1_000,
)

ROLLING_OFI_WINDOWS_MS = (
    200,
    500,
    1_000,
)

ROLLING_OBI_WINDOWS_MS = (
    200,
    500,
    1_000,
)

REGIME_WINDOWS_MS = (
    500,
    1_000,
    3_000,
)

EVENT_DENSITY_WINDOWS_MS = (
    100,
    200,
    500,
    1_000,
    3_000,
)


SPREAD_DEPTH_REGIME_WINDOWS_MS = (
    500,
    1_000,
    3_000,
)

NORMALIZED_OFI_LEVELS = (1, 3, 5, 10)
BPS_DEPTH_BANDS = (1.0, 2.0, 5.0, 10.0)
CALENDAR_CONTEXT_FEATURES = (
    "utc_hour_sin",
    "utc_hour_cos",
    "utc_dow_sin",
    "utc_dow_cos",
    "is_weekend",
)
NOTIONAL_CONTEXT_FEATURES = (
    "bid_l1_notional_usd",
    "ask_l1_notional_usd",
    "bid_depth_notional_5bps",
    "ask_depth_notional_5bps",
    "total_depth_notional_5bps",
)
ROLLING_OFI_LEVELS = (1, 3, 5, 10)
ROLLING_OBI_LEVELS = (3, 5, 10)
DEEP_MICRO_LEVELS = (3, 5, 10)
OFI_ACCEL_PAIRS_MS = (
    (200, 500),
    (500, 1_000),
)
BOOK_DEPTH_FEATURE_LEVELS = (1, 3, 5, 10, 20)
BOOK_SIGNAL_LEVELS = (1, 3, 5, 10)
MAX_BOOK_FEATURE_LEVEL = max(BOOK_DEPTH_FEATURE_LEVELS)

NUM_HEADS       = 16
# Loss mixing (fixed lambdas), with EMA normalization per loss
EMA_DECAY       = 0.99


# ---------------------------  Building blocks  ----------------------------

@dataclass(frozen=True)
class FeatureEventResult:
    ts_ms: int
    features: np.ndarray
    dt_ms: float
    is_decision: bool
    raw_mid: float
    event_type: str

    def __post_init__(self) -> None:
        if self.event_type not in {"ob", "trade"}:
            raise ValueError(f"event_type must be 'ob' or 'trade', got {self.event_type!r}")
        if self.is_decision != (self.event_type == "ob"):
            raise ValueError(
                f"is_decision must be True exactly for OB events; "
                f"got event_type={self.event_type!r}, is_decision={self.is_decision!r}"
            )


class BookValidationError(ValueError):
    pass


@dataclass
class ModelArgs:
    d_model: int
    n_layer: int
    vocab_size: int
    seq_in: int
    d_state: int = 128
    expand: int = 2
    d_conv: int = 3
    headdim: int = DMODEL // NUM_HEADS

    def __post_init__(self):
        assert self.d_model % NUM_HEADS == 0, "d_model must be divisible by NUM_HEADS"
        assert self.headdim == (self.d_model // NUM_HEADS), "headdim must equal d_model // NUM_HEADS"

class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-8):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(d))
    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.scale

def _init_small(m: nn.Module):
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, 0, .02)
        if m.bias is not None: nn.init.zeros_(m.bias)

def get_activation_fn(activation):
    if callable(activation): return activation()
    elif activation.lower() == "relu": return nn.ReLU()
    elif activation.lower() == "gelu": return nn.GELU()
    raise ValueError(f'{activation} is not available. You can use "relu", "gelu", or a callable')

class SublayerConnection(nn.Module):
    def __init__(self, enable_res_parameter, dropout=0.1):
        super(SublayerConnection, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.enable = enable_res_parameter
        if enable_res_parameter:
            self.a = nn.Parameter(torch.tensor(1e-8))
    def forward(self, x, out_x):
        if not self.enable:
            return x + self.dropout(out_x)
        else:
            return x + self.dropout(self.a * out_x)

# ------------  ConvTimeNet  ------------  
def zero_init(m):
    if type(m) == nn.Linear or type(m) == nn.Conv1d:
        m.weight.data.fill_(0)
        if m.bias is not None:
            m.bias.data.fill_(0)

class BoxCoder(nn.Module):
    def __init__(self, patch_count, patch_stride, patch_size, seq_len, channels):
        super().__init__()
        self.seq_len = seq_len
        self.channels = channels
        self.patch_size = patch_size
        self.patch_count = patch_count
        self.patch_stride = patch_stride
        self._generate_anchor()

    def _generate_anchor(self):
        anchors = []
        self.S_bias = (self.patch_size - 1) / 2
        for i in range(self.patch_count):
            x = i * self.patch_stride + 0.5 * (self.patch_size - 1)
            anchors.append(x)
        anchors = torch.as_tensor(anchors, dtype=torch.float32)
        self.register_buffer("anchor", anchors)

    def forward(self, boxes):
        bound = self.decode(boxes)
        points = self.meshgrid(bound)
        return points, bound

    def decode(self, rel_codes):
        boxes = self.anchor.to(device=rel_codes.device, dtype=rel_codes.dtype)
        dx = rel_codes[:, :, :, 0]
        ds = torch.relu(rel_codes[:, :, :, 1] + self.S_bias)

        pred_boxes = torch.zeros_like(rel_codes)
        ref_x = boxes.view(1, boxes.shape[0], 1)
        pred_boxes[:, :, :, 0] = dx + ref_x - ds
        pred_boxes[:, :, :, 1] = dx + ref_x + ds
        pred_boxes = pred_boxes / (self.seq_len - 1)
        return pred_boxes.clamp(min=0.0, max=1.0)

    def meshgrid(self, boxes):
        B, patch_count, C = boxes.shape[0], boxes.shape[1], boxes.shape[2]
        channel_boxes = torch.zeros(
            (B, patch_count, 2),
            device=boxes.device,
            dtype=boxes.dtype,
        )
        channel_boxes[:, :, 1] = 1.0

        xs = boxes.view(B * patch_count, C, 2)
        xs = F.interpolate(xs, size=self.patch_size, mode="linear", align_corners=True)

        ys = F.interpolate(channel_boxes, size=self.channels, mode="linear", align_corners=True)

        xs = xs.view(B, patch_count, C, self.patch_size, 1)
        ys = ys.unsqueeze(3).expand(B, patch_count, C, self.patch_size).unsqueeze(-1)

        return torch.stack([xs, ys], dim=-1)

class OffsetPredictor(nn.Module):
    def __init__(self, in_feats, patch_size, stride, use_zero_init=True):
        super().__init__()
        self.stride = stride
        self.channel = in_feats
        self.patch_size = patch_size
        hid_dim = 64
        self.offset_predictor = nn.Sequential(
            nn.Conv1d(1, hid_dim, patch_size, stride=stride, padding=0),
            nn.GELU(),
            nn.Conv1d(hid_dim, 2, 1, 1, padding=0)
        )
        if use_zero_init:
            self.offset_predictor.apply(zero_init)
    def forward(self, X):
        patch_X = X.unsqueeze(1).permute(0, 1, 3, 2)
        patch_X = F.unfold(patch_X, kernel_size=(self.patch_size, self.channel), stride=self.stride).permute(0, 2, 1)
        B, patch_count = patch_X.shape[0], patch_X.shape[1] 
        patch_X = patch_X.contiguous().view(B, patch_count, self.patch_size, self.channel)
        patch_X = patch_X.permute(0, 1, 3, 2)
        patch_X = patch_X.contiguous().view(B*patch_count*self.channel, 1, self.patch_size)
        pred_offset = self.offset_predictor(patch_X)
        pred_offset = pred_offset.view(B, patch_count, self.channel, 2).contiguous()
        return pred_offset

class DepatchSampling(nn.Module):
    def __init__(self, in_feats, seq_len, patch_size, stride):	 
        super(DepatchSampling, self).__init__()
        self.in_feats = in_feats
        self.seq_len = seq_len
        self.patch_size = patch_size
        self.patch_count = (seq_len - patch_size) // stride + 1
        self.dropout = nn.Dropout(0.1)
        self.offset_mode = DEPATCH_OFFSET_MODE
        self.offset_predictor = OffsetPredictor(in_feats, patch_size, stride)
        self.box_coder = BoxCoder(self.patch_count, stride, patch_size, self.seq_len, in_feats)
        if not hasattr(DepatchSampling, "_printed_offset_mode"):
            print(
                f"[depatch-config] offset_mode=learnable offset_predictor_hid_dim=64",
                flush=True,
            )
            DepatchSampling._printed_offset_mode = True
    def get_sampling_location(self, X):
        pred_offset = self.offset_predictor(X)
        sampling_locations, bound = self.box_coder(pred_offset)
        return sampling_locations, bound

    @torch.no_grad()
    def diagnostics(self, X: torch.Tensor) -> dict:
        pred_offset = self.offset_predictor(X.float())
        dx = pred_offset[..., 0].detach().float()
        ds_raw = pred_offset[..., 1].detach().float()
        _, bound = self.box_coder(pred_offset)
        bf = bound.detach().float()
        span_samples = (bf[..., 1] - bf[..., 0]) * float(self.seq_len - 1)
        dx_abs = dx.abs().reshape(-1)
        ds_abs = ds_raw.abs().reshape(-1)
        span_flat = span_samples.reshape(-1)
        dx_q = _bounded_flat_sample_for_quantile(dx_abs, max_elems=ACT_DIAG_P95_MAX_ELEMS)
        ds_q = _bounded_flat_sample_for_quantile(ds_abs, max_elems=ACT_DIAG_P95_MAX_ELEMS)
        span_q = _bounded_flat_sample_for_quantile(
            span_flat,
            max_elems=ACT_DIAG_P95_MAX_ELEMS,
        )
        return {
            "offset_dx_mean": float(dx.mean().cpu()),
            "offset_dx_std": float(dx.std(unbiased=False).cpu()),
            "offset_dx_abs_p95": float(torch.quantile(dx_q, 0.95).cpu()),
            "offset_dx_abs_max": float(dx_abs.max().cpu()),
            "offset_ds_raw_mean": float(ds_raw.mean().cpu()),
            "offset_ds_raw_std": float(ds_raw.std(unbiased=False).cpu()),
            "offset_ds_raw_abs_p95": float(torch.quantile(ds_q, 0.95).cpu()),
            "span_samples_mean": float(span_flat.mean().cpu()),
            "span_samples_p05": float(torch.quantile(span_q, 0.05).cpu()),
            "span_samples_p50": float(torch.quantile(span_q, 0.50).cpu()),
            "span_samples_p95": float(torch.quantile(span_q, 0.95).cpu()),
            "bound_left_clip_frac": float((bf[..., 0] <= 1e-6).float().mean().cpu()),
            "bound_right_clip_frac": float((bf[..., 1] >= 1.0 - 1e-6).float().mean().cpu()),
            "depatch_quantile_sampled": bool(
                max(dx_abs.numel(), ds_abs.numel(), span_flat.numel()) > ACT_DIAG_P95_MAX_ELEMS
            ),
            "depatch_quantile_elems": int(span_q.numel()),
        }

    def forward(self, X, return_bound=False):
        orig_dtype = X.dtype
        ctx = (
            torch.amp.autocast(device_type="cuda", enabled=False)
            if X.is_cuda
            else contextlib.nullcontext()
        )

        with ctx:
            X32 = X.float()
            img = X32.unsqueeze(1)
            B = img.shape[0]
            sampling_locations, bound = self.get_sampling_location(X32)
            sampling_locations = sampling_locations.view(B, self.patch_count*self.in_feats, self.patch_size, 2)
            sampling_locations = (sampling_locations - 0.5) * 2
            output = F.grid_sample(img, sampling_locations, align_corners=True)
            output = output.view(B, self.patch_count, self.in_feats, self.patch_size)
            output = output.permute(0, 2, 1, 3).contiguous()
        return output.to(dtype=orig_dtype) # (B, C, patch_count, patch_size)

class _ConvEncoderLayer(nn.Module):
    def __init__(self, kernel_size, d_model, d_ff=256, dropout=0.1, activation="gelu", 
                 enable_res_param=True, norm='batch', re_param=True, small_ks=3):
        super(_ConvEncoderLayer, self).__init__()
        self.norm_tp = norm
        self.re_param = re_param
        if not re_param: 
            self.DW_conv = nn.Conv1d(d_model, d_model, kernel_size, 1, 'same', groups=d_model)
        else:
            self.large_ks = kernel_size
            self.small_ks = small_ks
            self.DW_conv_large = nn.Conv1d(d_model, d_model, kernel_size, stride=1, padding='same', groups=d_model)
            self.DW_conv_small = nn.Conv1d(d_model, d_model, small_ks, stride=1, padding='same', groups=d_model)
        self.dw_act = get_activation_fn(activation)
        self.sublayerconnect1 = SublayerConnection(enable_res_param, dropout)
        self.dw_norm = nn.LayerNorm(d_model) if norm != 'batch' else nn.BatchNorm1d(d_model)
        
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff), 
            get_activation_fn(activation), 
            nn.Dropout(dropout), 
            nn.Linear(d_ff, d_model)
        )
        self.sublayerconnect2 = SublayerConnection(enable_res_param, dropout)
        self.norm_ffn = nn.LayerNorm(d_model) if norm != 'batch' else nn.BatchNorm1d(d_model)

    def forward(self, src:torch.Tensor) -> torch.Tensor: # [B, C, L]
        if self.re_param:
            out_x = self.DW_conv_large(src) + self.DW_conv_small(src)
        else:
            out_x = self.DW_conv(src)

        residual_src = self.sublayerconnect1(src, self.dw_act(out_x))
        
        if self.norm_tp != 'batch':
            # LayerNorm natively operates on C, so permute to (B, L, C)
            normed_src = residual_src.permute(0, 2, 1).contiguous()
            normed_src = self.dw_norm(normed_src)
            
            # Apply Linear FFN directly on (B, L, C)
            ff_out = self.ff(normed_src)
            
            residual_src2 = self.sublayerconnect2(normed_src, ff_out)
            normed_src2 = self.norm_ffn(residual_src2)
            
            # Return cleanly as (B, C, L)
            return normed_src2.permute(0, 2, 1).contiguous()
        else:
            normed_src = self.dw_norm(residual_src)
            
            # Must temporarily permute to (B, L, C) for Linear FFN
            normed_src_t = normed_src.permute(0, 2, 1).contiguous()
            ff_out_t = self.ff(normed_src_t)
            ff_out = ff_out_t.permute(0, 2, 1).contiguous()
            
            residual_src2 = self.sublayerconnect2(normed_src, ff_out)
            return self.norm_ffn(residual_src2)

class ConvEncoder(nn.Module):
    def __init__(self, d_model, d_ff, kernel_size=[3,3,5,5,7,7], dropout=0.1, activation='gelu', 
                 n_layers=6, enable_res_param=True, norm='batch', re_param=True, small_ks=3):
        super(ConvEncoder, self).__init__()
        self.layers = nn.ModuleList([_ConvEncoderLayer(kernel_size[i], d_model, d_ff=d_ff, dropout=dropout, 
                                                        activation=activation, enable_res_param=enable_res_param, norm=norm, 
                                                        re_param=re_param, small_ks=small_ks) \
                                                        for i in range(n_layers)])
    def forward(self, src):
        output = src
        for mod in self.layers: 
            output = mod(output)
        return output

class FeatureReliabilityGate(nn.Module):
    """
    Dynamic per-feature reliability gate with a static learned per-feature prior.

    Input:
        z: [B, L, F, C]
           B = batch
           L = patch_count
           F = in_feats
           C = d_model_internal

    Output:
        gated_z: same shape as z
    """

    def __init__(self, in_feats: int, d_internal: int, dropout: float = 0.05, init_keep_prob: float = 0.90):
        super().__init__()
        self.in_feats = int(in_feats)
        self.d_internal = int(d_internal)
        self.init_keep_prob = float(init_keep_prob)
        self.register_buffer(
            "gate_alpha",
            torch.tensor(1.0, dtype=torch.float32),
            persistent=False,
        )

        hidden = max(8, 2 * d_internal)
        self.norm = nn.LayerNorm(d_internal)
        self.dynamic = nn.Sequential(
            nn.Linear(d_internal, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )
        self.feature_prior_logit = nn.Parameter(torch.empty(in_feats))

        prior_init = math.log(init_keep_prob / (1.0 - init_keep_prob))
        nn.init.zeros_(self.dynamic[-1].weight)
        nn.init.zeros_(self.dynamic[-1].bias)
        nn.init.constant_(self.feature_prior_logit, prior_init)

    def _compute_gate(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        assert z.ndim == 4
        _, _, F, C = z.shape
        assert F == self.in_feats
        assert C == self.d_internal
        dyn = self.dynamic(self.norm(z))  # [B, L, F, 1]
        prior = self.feature_prior_logit.view(1, 1, F, 1).to(dtype=dyn.dtype, device=dyn.device)
        gate = torch.sigmoid(dyn + prior)  # [B, L, F, 1]
        return gate, dyn, prior

    def set_gate_alpha(self, alpha: float) -> None:
        alpha = float(alpha)
        if not math.isfinite(alpha):
            raise ValueError(f"gate alpha must be finite, got {alpha}")
        alpha = max(0.0, min(1.0, alpha))
        self.gate_alpha.fill_(alpha)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        gate, _, _ = self._compute_gate(z)
        multiplier = gate / self.init_keep_prob
        alpha = self.gate_alpha.to(device=z.device, dtype=z.dtype)
        effective_multiplier = (1.0 - alpha) + alpha * multiplier
        return z * effective_multiplier

    @torch.no_grad()
    def gate_diagnostics(self, z: torch.Tensor) -> dict:
        gate, dyn, prior = self._compute_gate(z)
        gf = gate.detach().float()
        gf_q = _bounded_flat_sample_for_quantile(
            gf,
            max_elems=ACT_DIAG_P95_MAX_ELEMS,
        )
        q = torch.quantile(
            gf_q,
            torch.tensor([0.01, 0.05, 0.50, 0.95, 0.99], device=gf_q.device),
        )
        feature_gate = gf.mean(dim=(0, 1, 3))
        k = min(8, feature_gate.numel())
        top = torch.topk(feature_gate, k=k, largest=True).indices.detach().cpu().tolist()
        bottom = torch.topk(feature_gate, k=k, largest=False).indices.detach().cpu().tolist()
        return {
            "gate_mean": float(gf.mean().cpu()),
            "gate_std": float(gf.std(unbiased=False).cpu()),
            "gate_min": float(gf.min().cpu()),
            "gate_max": float(gf.max().cpu()),
            "gate_p01": float(q[0].cpu()),
            "gate_p05": float(q[1].cpu()),
            "gate_p50": float(q[2].cpu()),
            "gate_p95": float(q[3].cpu()),
            "gate_p99": float(q[4].cpu()),
            "gate_quantile_sampled": bool(gf.numel() > ACT_DIAG_P95_MAX_ELEMS),
            "gate_quantile_elems": int(gf_q.numel()),
            "gate_frac_lt_0p2": float((gf < 0.2).float().mean().cpu()),
            "gate_frac_lt_0p5": float((gf < 0.5).float().mean().cpu()),
            "gate_frac_gt_0p95": float((gf > 0.95).float().mean().cpu()),
            "dyn_mean": float(dyn.detach().float().mean().cpu()),
            "dyn_std": float(dyn.detach().float().std(unbiased=False).cpu()),
            "prior_mean": float(prior.detach().float().mean().cpu()),
            "prior_std": float(prior.detach().float().std(unbiased=False).cpu()),
            "top_gate_feature_idx_8": top,
            "bottom_gate_feature_idx_8": bottom,
            "alpha": float(self.gate_alpha.detach().cpu().item()),
        }

    @torch.no_grad()
    def gate_stats(self, z: torch.Tensor) -> dict:
        diag = self.gate_diagnostics(z)
        return {
            "mean": diag["gate_mean"],
            "std": diag["gate_std"],
            "min": diag["gate_min"],
            "max": diag["gate_max"],
            "alpha": float(self.gate_alpha.detach().cpu().item()),
        }


@torch.no_grad()
def _bounded_flat_sample_for_quantile(
    x: torch.Tensor,
    *,
    max_elems: int,
) -> torch.Tensor:
    """Return a bounded deterministic flattened sample for diagnostic quantiles.

    This is only for diagnostics. It avoids calling torch.quantile on huge tensors.
    Sampling is deterministic stride sampling, not random.
    """
    xf = x.detach().reshape(-1)
    n = int(xf.numel())
    if n <= max_elems:
        return xf

    max_elems = int(max(1, max_elems))
    stride = max(1, n // max_elems)
    return xf[::stride][:max_elems]


@torch.no_grad()
def _activation_summary(t: torch.Tensor) -> dict:
    td = t.detach()
    tf = td.float()
    finite = torch.isfinite(tf)
    total = max(1, tf.numel())
    vals = tf[finite]
    if vals.numel() == 0:
        out = {
            "shape": list(td.shape),
            "dtype": str(td.dtype),
            "mean": float("nan"),
            "std": float("nan"),
            "rms": float("nan"),
            "abs_p95": float("nan"),
            "abs_p95_sampled": False,
            "abs_p95_elems": 0,
            "abs_max": float("nan"),
            "zero_frac_abs_lt_1e_minus_6": float("nan"),
            "finite_bad_frac": 1.0,
        }
    else:
        abs_vals = vals.abs()
        abs_q = _bounded_flat_sample_for_quantile(
            abs_vals,
            max_elems=ACT_DIAG_P95_MAX_ELEMS,
        )
        out = {
            "shape": list(td.shape),
            "dtype": str(td.dtype),
            "mean": float(vals.mean().cpu()),
            "std": float(vals.std(unbiased=False).cpu()),
            "rms": float(torch.sqrt((vals * vals).mean()).cpu()),
            "abs_p95": float(torch.quantile(abs_q, 0.95).cpu()),
            "abs_p95_sampled": bool(abs_vals.numel() > ACT_DIAG_P95_MAX_ELEMS),
            "abs_p95_elems": int(abs_q.numel()),
            "abs_max": float(abs_vals.max().cpu()),
            "zero_frac_abs_lt_1e_minus_6": float((tf.abs() < 1e-6).float().mean().cpu()),
            "finite_bad_frac": float((~finite).float().mean().cpu()),
        }
    if td.ndim == 3:
        out["time_std_mean"] = float(tf.std(dim=1, unbiased=False).mean().cpu())
        out["token_std_mean"] = float(tf.std(dim=-1, unbiased=False).mean().cpu())
    return out


class ConvTimeNetFeatureExtractor(nn.Module):
    def __init__(
        self,
        in_feats: int,
        seq_len: int,
        d_model: int,
        dropout: float = 0.1,
        act: str = "gelu",
        norm: str = "layer",
        re_param: bool = True,
        re_param_kernel: int = 3,
    ):
        super(ConvTimeNetFeatureExtractor, self).__init__()
        assert d_model == DMODEL
        assert len(CTN_CI_KERNELS) == 6
        assert len(CTN_MIXED_KERNELS) == 0

        self.depatch = DepatchSampling(
            in_feats=in_feats,
            seq_len=seq_len,
            patch_size=CTN_PATCH_SIZE,
            stride=CTN_PATCH_STRIDE,
        )
        self.patch_count = (seq_len - CTN_PATCH_SIZE) // CTN_PATCH_STRIDE + 1
        self.patch_size = CTN_PATCH_SIZE
        self.d_model_internal = CTN_CI_INTERNAL_DIM
        self.d_ff_internal = CTN_CI_DFF
        self.final_in_dim = in_feats * self.d_model_internal

        self.output_linear = nn.Linear(CTN_PATCH_SIZE, CTN_CI_INTERNAL_DIM)
        self.ci_encoder = ConvEncoder(
            d_model=CTN_CI_INTERNAL_DIM,
            d_ff=CTN_CI_DFF,
            kernel_size=CTN_CI_KERNELS,
            dropout=dropout,
            activation=act,
            n_layers=len(CTN_CI_KERNELS),
            enable_res_param=True,
            norm=norm,
            re_param=re_param,
            small_ks=re_param_kernel,
        )
        self.final_proj = nn.Linear(self.final_in_dim, d_model)
        # Match the original _1_sec ConvTimeNet path: no post-final_proj LayerNorm.
        self.output_norm = None

        effective_rf_samples = CTN_PATCH_SIZE + sum(int(k) - 1 for k in CTN_CI_KERNELS)
        sample_ms = float(WINDOW_MS) / float(LOOKBACK)
        effective_rf_ms = int(round(effective_rf_samples * sample_ms))
        print(
            f"[ctn-config] arch={MODEL_ARCH_SCHEMA} patch_size={CTN_PATCH_SIZE} stride={CTN_PATCH_STRIDE} "
            f"patch_count={self.patch_count} effective_rf_samples={effective_rf_samples} effective_rf_ms={effective_rf_ms}",
            flush=True,
        )
        ci_kernel_str = ",".join(str(k) for k in CTN_CI_KERNELS)
        print(
            f"[ctn-config] ci_layers={len(CTN_CI_KERNELS)} ci_kernels=[{ci_kernel_str}] "
            f"ci_dim={CTN_CI_INTERNAL_DIM} ci_dff={CTN_CI_DFF} ci_res_param=1",
            flush=True,
        )
        print(
            f"[ctn-config] final_in_dim={self.final_in_dim} final_proj_out={d_model} gate=disabled mixed=disabled",
            flush=True,
        )

    def set_gate_alpha(self, alpha: float) -> None:
        # Kept as a no-op compatibility hook for training code that may still
        # call it during gate-warmup setup.
        return None

    def _ci_encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # x: [B, LOOKBACK, F]. DepatchSampling operates on [B, F, LOOKBACK].
        B = x.size(0)
        F_in = x.size(-1)
        x_permuted = x.permute(0, 2, 1).contiguous()
        out_patch = self.depatch(x_permuted).contiguous()  # [B, F, patch_count, patch_size]
        out = self.output_linear(out_patch).contiguous()  # [B, F, patch_count, internal_dim]

        B_out, F_out, L, C = out.shape
        assert B_out == B
        assert F_out == F_in
        assert C == self.d_model_internal
        assert L == self.patch_count

        u = out.reshape(B * F_in, self.patch_count, self.d_model_internal)
        u = u.permute(0, 2, 1).contiguous()  # [B*F, C_internal, patch_count]

        ci_t = self.ci_encoder(u)
        assert ci_t.ndim == 3
        assert ci_t.shape[0] == B * x.size(-1)
        assert ci_t.shape[1] == self.d_model_internal
        assert ci_t.shape[2] == self.patch_count
        return out_patch, out, ci_t

    def forward(self, x):
        # x: [B, LOOKBACK, F]
        B = x.size(0)
        F_in = x.size(-1)
        _, _, ci_t = self._ci_encode(x)

        # Preserve patch/time tokens; final_proj is applied per patch.
        out = ci_t.permute(0, 2, 1).contiguous()  # [B*F, patch_count, C_internal]
        out = out.reshape(B, F_in, self.patch_count, self.d_model_internal)
        out = out.permute(0, 2, 3, 1).contiguous()  # [B, patch_count, C_internal, F]
        out = out.reshape(B, self.patch_count, self.final_in_dim)  # [B, patch_count, F*C_internal]
        out = self.final_proj(out)  # [B, patch_count, DMODEL]

        if hasattr(self, "output_norm") and self.output_norm is not None:
            out = self.output_norm(out)

        return out

    @torch.no_grad()
    def residual_scalar_diagnostics(self) -> dict:
        vals = []
        for mod in self.ci_encoder.modules():
            if isinstance(mod, SublayerConnection) and getattr(mod, "enable", False) and hasattr(mod, "a"):
                vals.append(mod.a.detach().float().reshape(1))
        ci = torch.cat(vals) if vals else torch.empty(0)

        def stats(prefix: str, vals: torch.Tensor) -> dict:
            if vals.numel() == 0:
                return {f"{prefix}_res_a_mean": float("nan"), f"{prefix}_res_a_min": float("nan"), f"{prefix}_res_a_max": float("nan"), f"{prefix}_res_a_absmax": float("nan")}
            return {
                f"{prefix}_res_a_mean": float(vals.mean().cpu()),
                f"{prefix}_res_a_min": float(vals.min().cpu()),
                f"{prefix}_res_a_max": float(vals.max().cpu()),
                f"{prefix}_res_a_absmax": float(vals.abs().max().cpu()),
            }
        out = {}
        out.update(stats("ci", ci))
        out.update({
            "mixed_res_a_mean": float("nan"),
            "mixed_res_a_min": float("nan"),
            "mixed_res_a_max": float("nan"),
            "mixed_res_a_absmax": float("nan"),
        })
        return out

    @torch.no_grad()
    def forward_with_diagnostics(self, x: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        diag = {"activations": {}}
        diag["activations"]["x_input"] = _activation_summary(x)
        out_patch, patch_embed, ci_t = self._ci_encode(x)
        B = x.size(0)
        F_in = x.size(-1)
        diag["activations"]["depatch_out"] = _activation_summary(out_patch)
        diag["activations"]["patch_embed"] = _activation_summary(patch_embed)
        diag["activations"]["ci_out"] = _activation_summary(ci_t)

        out = ci_t.permute(0, 2, 1).contiguous()
        out = out.reshape(B, F_in, self.patch_count, self.d_model_internal)
        out = out.permute(0, 2, 3, 1).contiguous()
        ci_patch_flat = out.reshape(B, self.patch_count, self.final_in_dim)
        final = self.final_proj(ci_patch_flat)
        if hasattr(self, "output_norm") and self.output_norm is not None:
            final = self.output_norm(final)

        diag["activations"]["ci_patch_flat"] = _activation_summary(ci_patch_flat)
        diag["activations"]["ci_pooled_flat"] = diag["activations"]["ci_patch_flat"]
        diag["activations"]["final_proj"] = _activation_summary(final)
        diag["activations"]["extractor_out"] = diag["activations"]["final_proj"]

        # Compatibility aliases for older diagnostic print code. These summarize
        # the inactive stages without performing gate or mixed-conv computation.
        diag["activations"]["pre_gate"] = diag["activations"]["ci_out"]
        diag["activations"]["post_gate"] = diag["activations"]["ci_patch_flat"]
        diag["activations"]["post_gate_flat"] = diag["activations"]["ci_patch_flat"]
        diag["activations"]["post_proj"] = diag["activations"]["final_proj"]
        diag["activations"]["post_mixed"] = diag["activations"]["final_proj"]

        diag["ci"] = diag["activations"]["ci_out"]
        diag["ci_patch_flat"] = diag["activations"]["ci_patch_flat"]
        diag["final_proj"] = diag["activations"]["final_proj"]
        diag["out"] = diag["activations"]["final_proj"]
        diag["post_gate"] = diag["activations"]["ci_patch_flat"]
        diag["post_proj"] = diag["activations"]["final_proj"]
        diag["post_mixed"] = diag["activations"]["final_proj"]

        eps = 1e-12
        diag["ratios"] = {
            "gate_over_ci_rms": 1.0,
            "proj_over_flat_rms": diag["activations"]["final_proj"]["rms"] / (diag["activations"]["ci_patch_flat"]["rms"] + eps),
            "mixed_over_proj_rms": 1.0,
        }
        diag["gate"] = {
            "enabled": 0.0,
            "gate_mean": float("nan"),
            "gate_std": float("nan"),
            "gate_min": float("nan"),
            "gate_max": float("nan"),
            "gate_p05": float("nan"),
            "gate_p50": float("nan"),
            "gate_p95": float("nan"),
            "gate_frac_lt_0p5": float("nan"),
            "gate_frac_gt_0p95": float("nan"),
            "dyn_mean": float("nan"),
            "dyn_std": float("nan"),
            "prior_mean": float("nan"),
            "prior_std": float("nan"),
            "alpha": float("nan"),
        }
        diag["mixed"] = {"enabled": 0.0}
        diag["depatch"] = self.depatch.diagnostics(x.permute(0, 2, 1).contiguous())
        diag["residual_scalars"] = self.residual_scalar_diagnostics()
        return final, diag

def _debug_check_ctn_token_diversity(model, feature_dim: int, device: str = "cuda"):
    """Smoke check that ConvTimeNet preserves non-identical patch tokens."""
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    model.eval()
    x = torch.randn(2, LOOKBACK, feature_dim, device=device)
    with torch.no_grad():
        extractor = getattr(model, "feature_extractor", None)
        if extractor is None:
            extractor = model.depatch_proj_encoder
        h = extractor(x)
    assert h.ndim == 3
    assert h.shape[0] == 2
    assert h.shape[-1] == DMODEL
    if h.shape[1] > 1:
        diff = (h[:, 1:, :] - h[:, :-1, :]).abs().mean().item()
        print(f"[ctn-token-check] shape={tuple(h.shape)} adjacent_token_absdiff_mean={diff:.6g}", flush=True)
        assert diff > 0.0, "ConvTimeNet output tokens are identical; patch dimension was collapsed."



# ------------  Mamba wrapper + pooling ------------
class ResidualBlock(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.n = nn.LayerNorm(args.d_model)
        self.m = Mamba2(
            d_model=args.d_model,
            d_state=args.d_state,
            d_conv=args.d_conv,
            expand=args.expand,
            headdim=args.headdim,
        )
    def forward(self, x):
        return x + self.m(self.n(x))

class ChannelFFN(nn.Module):
    """Channel-wise FFN (LN over channels) applied per time step."""
    def __init__(self, d_model, ff_hid, dropout=0.1):
        super().__init__()
        self.ln = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, ff_hid),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_hid, d_model)
        )
    def forward(self, x):  # x: [B, L, D]
        return x + self.ff(self.ln(x))

class GatedPooling(nn.Module):
    """O(L) learned-query / gated pooling."""
    def __init__(self, d_in, d_hidden=None):
        super().__init__()
        d_hidden = d_hidden or d_in
        self.W = nn.Linear(d_in, d_hidden)
        self.u = nn.Parameter(torch.randn(d_hidden))

    def forward(self, h):  # h: [B, L, D]
        g = torch.tanh(self.W(h))        # [B, L, H]
        scores = torch.matmul(g, self.u) # [B, L]
        alpha = torch.softmax(scores, dim=1)
        z = torch.einsum('bl,bld->bd', alpha, h)
        return z

class TaskTokenDecoder(nn.Module):
    """Lightweight task-specific temporal refinement block over shared Mamba token states."""

    def __init__(self, dim: int, kernel_size: int = 5, ff_mult: int = 2, dropout: float = 0.1):
        super().__init__()
        assert kernel_size % 2 == 1, "TaskTokenDecoder requires odd kernel_size for same-length padding"
        self.norm1 = nn.LayerNorm(dim)
        self.temporal = nn.Conv1d(
            in_channels=dim,
            out_channels=dim,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=dim,
            bias=True,
        )
        self.drop1 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, ff_mult * dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_mult * dim, dim),
        )
        self.drop2 = nn.Dropout(dropout)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        z = self.norm1(h)
        z = self.temporal(z.transpose(1, 2)).transpose(1, 2)
        h = h + self.drop1(z)
        h = h + self.drop2(self.ff(self.norm2(h)))
        return h

class Mamba(nn.Module):
    """Bidirectional Mamba stacks returning fused token states."""

    def __init__(self, args: ModelArgs, ff_hid: int):
        super().__init__()
        self.args = args
        self.emb = nn.Linear(args.vocab_size, args.d_model)
        _init_small(self.emb)

        self.blocks_fwd = nn.ModuleList([ResidualBlock(args) for _ in range(args.n_layer)])
        self.ffns_fwd = nn.ModuleList([ChannelFFN(args.d_model, ff_hid) for _ in range(args.n_layer)])

        self.blocks_bwd = nn.ModuleList([ResidualBlock(args) for _ in range(args.n_layer)])
        self.ffns_bwd = nn.ModuleList([ChannelFFN(args.d_model, ff_hid) for _ in range(args.n_layer)])

        self.norm_fwd = nn.LayerNorm(args.d_model)
        self.norm_bwd = nn.LayerNorm(args.d_model)

    def _run_stack(self, x, blocks, ffns):
        for blk, ffn in zip(blocks, ffns):
            x = blk(x)
            x = ffn(x)
        return x

    def forward(self, x, embedded=False):
        if not embedded:
            x = self.emb(x)  # project features to d_model

        x_fwd = self._run_stack(x, self.blocks_fwd, self.ffns_fwd)

        x_bwd_in = torch.flip(x, dims=[1])
        x_bwd = self._run_stack(x_bwd_in, self.blocks_bwd, self.ffns_bwd)
        x_bwd = torch.flip(x_bwd, dims=[1])

        h_fwd = self.norm_fwd(x_fwd)
        h_bwd = self.norm_bwd(x_bwd)
        h = torch.cat([h_fwd, h_bwd], dim=-1)
        return h, h_fwd

# -------------  SAMBA -------------
class SAMBA(nn.Module):
    def __init__(self, args: ModelArgs, feature_dim: Optional[int] = None):
        super().__init__()
        self.args = args
        assert args.d_model == DMODEL, f"Expected args.d_model ({args.d_model}) == DMODEL ({DMODEL})"
        assert args.d_model % NUM_HEADS == 0, "args.d_model must be divisible by NUM_HEADS"
        assert args.headdim == (args.d_model // NUM_HEADS), "args.headdim must match d_model // NUM_HEADS"
        extractor_in_feats = int(feature_dim) if feature_dim is not None else args.vocab_size
        self.depatch_proj_encoder = ConvTimeNetFeatureExtractor(
            in_feats=extractor_in_feats,
            seq_len=args.seq_in,
            d_model=args.d_model,
            dropout=0.1,
            act="gelu",
            norm="layer",
            re_param=True,
            re_param_kernel=3,
        )
        self.feature_extractor = self.depatch_proj_encoder
        # Mamba backbone (forward/backward fusion)
        self.mamba = Mamba(args, ff_hid=4*DMODEL)

        fused_dim = args.d_model * 2
        head_hidden_dim = fused_dim * 2
        self.dir_token_decoder = TaskTokenDecoder(
            dim=fused_dim,
            kernel_size=5,
            ff_mult=2,
            dropout=0.1,
        )
        self.mag_token_decoder = TaskTokenDecoder(
            dim=fused_dim,
            kernel_size=5,
            ff_mult=2,
            dropout=0.1,
        )
        self.dir_pool = GatedPooling(fused_dim)
        self.mag_pool = GatedPooling(fused_dim)
        self.dir_head = nn.Sequential(
            nn.Linear(fused_dim, head_hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(head_hidden_dim, NUM_HORIZONS)
        )
        self.mag_up_head = nn.Sequential(
            nn.Linear(fused_dim, head_hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(head_hidden_dim, NUM_HORIZONS),
        )
        self.mag_down_head = nn.Sequential(
            nn.Linear(fused_dim, head_hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(head_hidden_dim, NUM_HORIZONS),
        )
        for head in (self.dir_head, self.mag_up_head, self.mag_down_head):
            for mod in head:
                if isinstance(mod, nn.Linear):
                    _init_small(mod)

    def set_gate_alpha(self, alpha: float) -> None:
        self.depatch_proj_encoder.set_gate_alpha(alpha)

    @torch.no_grad()
    def initialize_magnitude_head_bias(self, pos_target_sqrt, neg_target_sqrt) -> dict:
        mag_up_final = self.mag_up_head[-1]
        mag_down_final = self.mag_down_head[-1]
        assert isinstance(mag_up_final, nn.Linear)
        assert isinstance(mag_down_final, nn.Linear)
        assert mag_up_final.out_features == NUM_HORIZONS
        assert mag_down_final.out_features == NUM_HORIZONS

        device = mag_up_final.bias.device
        pos = torch.as_tensor(pos_target_sqrt, dtype=torch.float32, device=device).reshape(-1)
        neg = torch.as_tensor(neg_target_sqrt, dtype=torch.float32, device=device).reshape(-1)
        if pos.numel() != NUM_HORIZONS or neg.numel() != NUM_HORIZONS:
            raise ValueError(
                f"magnitude init targets must have {NUM_HORIZONS} elements, "
                f"got pos={pos.numel()} neg={neg.numel()}"
            )
        pos = torch.clamp(pos, min=1e-4)
        neg = torch.clamp(neg, min=1e-4)
        pos_bias = inverse_softplus_torch(pos)
        neg_bias = inverse_softplus_torch(neg)

        mag_up_final.weight.zero_()
        mag_down_final.weight.zero_()
        mag_up_final.bias.copy_(pos_bias)
        mag_down_final.bias.copy_(neg_bias)
        return {
            "pos_target_sqrt": [float(v) for v in pos.detach().cpu().tolist()],
            "neg_target_sqrt": [float(v) for v in neg.detach().cpu().tolist()],
            "pos_bias": [float(v) for v in pos_bias.detach().cpu().tolist()],
            "neg_bias": [float(v) for v in neg_bias.detach().cpu().tolist()],
        }

    def forward(self, x):
        h_tokens = self.depatch_proj_encoder(x).contiguous()        # [B, L, D] (ConvTimeNet projection applied)
        h, _ = self.mamba(h_tokens, embedded=True)
        h_dir = self.dir_token_decoder(h)
        h_mag = self.mag_token_decoder(h)
        pooled_dir = self.dir_pool(h_dir)
        pooled_mag = self.mag_pool(h_mag)
        dir_logits = self.dir_head(pooled_dir)
        mag_up_raw = self.mag_up_head(pooled_mag).float()
        mag_down_raw = self.mag_down_head(pooled_mag).float()
        mag_up_sqrt = F.softplus(mag_up_raw) + MAG_SQRT_EPS
        mag_down_sqrt = F.softplus(mag_down_raw) + MAG_SQRT_EPS
        assert dir_logits.shape[-1] == NUM_HORIZONS
        assert mag_up_sqrt.shape == dir_logits.shape
        assert mag_down_sqrt.shape == dir_logits.shape
        return {
            "dir_logits": dir_logits,
            "mag_up_sqrt": mag_up_sqrt,
            "mag_down_sqrt": mag_down_sqrt,
        }

    @torch.no_grad()
    def forward_with_diagnostics(self, x: torch.Tensor) -> Tuple[Dict[str, torch.Tensor], dict]:
        diag = {"activations": {}, "prediction": {}}
        h_tokens, ext_diag = self.depatch_proj_encoder.forward_with_diagnostics(x)
        diag["extractor"] = ext_diag
        diag["activations"]["mamba_tokens"] = _activation_summary(h_tokens)
        h, _ = self.mamba(h_tokens, embedded=True)
        diag["activations"]["mamba_fused"] = _activation_summary(h)
        h_dir = self.dir_token_decoder(h)
        h_mag = self.mag_token_decoder(h)
        diag["activations"]["dir_decoder_out"] = _activation_summary(h_dir)
        diag["activations"]["mag_decoder_out"] = _activation_summary(h_mag)
        pooled_dir = self.dir_pool(h_dir)
        pooled_mag = self.mag_pool(h_mag)
        diag["activations"]["pooled_dir"] = _activation_summary(pooled_dir)
        diag["activations"]["pooled_mag"] = _activation_summary(pooled_mag)
        dir_logits = self.dir_head(pooled_dir)
        mag_up_raw = self.mag_up_head(pooled_mag).float()
        mag_down_raw = self.mag_down_head(pooled_mag).float()
        mag_up_sqrt = F.softplus(mag_up_raw) + MAG_SQRT_EPS
        mag_down_sqrt = F.softplus(mag_down_raw) + MAG_SQRT_EPS
        diag["activations"]["dir_logits"] = _activation_summary(dir_logits)
        diag["activations"]["mag_up_raw"] = _activation_summary(mag_up_raw)
        diag["activations"]["mag_down_raw"] = _activation_summary(mag_down_raw)
        diag["activations"]["mag_up_sqrt"] = _activation_summary(mag_up_sqrt)
        diag["activations"]["mag_down_sqrt"] = _activation_summary(mag_down_sqrt)
        probs = torch.sigmoid(dir_logits.float())
        dir_logit_abs_q = _bounded_flat_sample_for_quantile(
            dir_logits.float().abs(),
            max_elems=ACT_DIAG_P95_MAX_ELEMS,
        )
        def _raw_stats(prefix: str, t: torch.Tensor) -> dict:
            tf = t.detach().float()
            tq = _bounded_flat_sample_for_quantile(tf, max_elems=ACT_DIAG_P95_MAX_ELEMS)
            qs = torch.quantile(tq, torch.tensor([0.05, 0.50, 0.95], device=tq.device))
            return {
                f"{prefix}_mean": float(tf.mean().cpu()),
                f"{prefix}_std": float(tf.std(unbiased=False).cpu()),
                f"{prefix}_p05": float(qs[0].cpu()),
                f"{prefix}_p50": float(qs[1].cpu()),
                f"{prefix}_p95": float(qs[2].cpu()),
                f"{prefix}_min": float(tf.min().cpu()),
                f"{prefix}_max": float(tf.max().cpu()),
            }

        pred_stats = {
            "dir_logit_mean": float(dir_logits.float().mean().cpu()),
            "dir_logit_std": float(dir_logits.float().std(unbiased=False).cpu()),
            "dir_logit_abs_p95": float(torch.quantile(dir_logit_abs_q, 0.95).cpu()),
            "dir_prob_mean": float(probs.mean().cpu()),
            "dir_prob_std": float(probs.std(unbiased=False).cpu()),
            "mag_up_mean": float(mag_up_sqrt.float().mean().cpu()),
            "mag_up_std": float(mag_up_sqrt.float().std(unbiased=False).cpu()),
            "mag_down_mean": float(mag_down_sqrt.float().mean().cpu()),
            "mag_down_std": float(mag_down_sqrt.float().std(unbiased=False).cpu()),
            "mag_up_floor_frac": float((mag_up_sqrt.float() <= MAG_SQRT_EPS * 1.01).float().mean().cpu()),
            "mag_down_floor_frac": float((mag_down_sqrt.float() <= MAG_SQRT_EPS * 1.01).float().mean().cpu()),
        }
        pred_stats.update(_raw_stats("mag_up_raw", mag_up_raw))
        pred_stats.update(_raw_stats("mag_down_raw", mag_down_raw))
        diag["prediction"] = pred_stats
        eps = 1e-12
        diag["ratios"] = {
            "fused_over_tokens_rms": diag["activations"]["mamba_fused"]["rms"] / (diag["activations"]["mamba_tokens"]["rms"] + eps),
        }
        pred = {
            "dir_logits": dir_logits,
            "mag_up_sqrt": mag_up_sqrt,
            "mag_down_sqrt": mag_down_sqrt,
        }
        return pred, diag

# --------------------  SAM Optimiser  ---------------------
class SAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.01, adaptive=False, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"
        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super(SAM, self).__init__(params, defaults)
        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)
            for p in group["params"]:
                if p.grad is None: continue
                self.state[p]["old_p"] = p.data.clone()
                e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale.to(p)
                p.add_(e_w)
        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                p.copy_(self.state[p]["old_p"])
        self.base_optimizer.step()
        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None):
        assert closure is not None, "Sharpness Aware Minimization requires closure"
        closure = torch.enable_grad()(closure)
        self.first_step(zero_grad=True)
        closure()
        self.second_step()

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device
        norm = torch.norm(
            torch.stack([
                ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad).norm(p=2).to(shared_device)
                for group in self.param_groups for p in group["params"]
                if p.grad is not None
            ]),
            p=2
        )
        return norm

# ------------------------  Data  --------------------------

"""Event-driven data pipeline for Bybit L2 order book and trade history."""

def _detect_container(path: str) -> str:
    # magic header sniffing is robust even if extension is wrong
    with open(path, "rb") as fh:
        sig = fh.read(4)
    if sig[:2] == b"PK":       # zip
        return "zip"
    if sig[:2] == b"\x1f\x8b": # gzip
        return "gz"
    return "plain"

@contextlib.contextmanager
def _open_text(path: str):
    kind = _detect_container(path)
    if kind == "zip":
        with zipfile.ZipFile(path) as z:
            name = z.namelist()[0]
            with z.open(name) as f:
                yield io.TextIOWrapper(f, encoding="utf-8")
    elif kind == "gz":
        with gzip.open(path, "rt", encoding="utf-8") as f:
            yield f
    else:
        with open(path, "rt", encoding="utf-8") as f:
            yield f

# --------------------  Data ingestion & merging  ---------------------

class BybitRawIter:
    """Iterate over Bybit L2 order book (.data) and trade history (.csv) files."""

    def __init__(self, ob_zip: str, th_zip: str):
        self.ob_zip = ob_zip
        self.th_zip = th_zip

    def ob_iter(self):
        # OB is line-delimited JSON.
        # Missing/invalid timestamps are hard errors to preserve monotonicity guarantees.
        with _open_text(self.ob_zip) as f:
            for line in f:
                if not line:
                    continue
                obj = json.loads(line)
                ts_raw = obj.get("ts") or obj.get("cts")
                if ts_raw is None or (isinstance(ts_raw, str) and not ts_raw.strip()):
                    raise ValueError(f"Missing OB timestamp in payload: {obj}")
                try:
                    ts = coerce_ts_ms(ts_raw)
                except ValueError as exc:
                    raise ValueError(
                        f"Invalid OB timestamp {ts_raw!r} in payload: {obj}"
                    ) from exc
                seq = obj["data"].get("seq", 0)
                yield ts, seq, obj

    def trade_iter(self):
        # TH is CSV with a 'timestamp' column in seconds
        with _open_text(self.th_zip) as f:
            reader = csv.DictReader(f)
            seq = 0

            for row in reader:
                seq += 1
                t_raw = row.get("timestamp")
                try:
                    ts_ms = timestamp_to_ms_half_even(t_raw)
                except ValueError as exc:
                    raise ValueError(
                        f"Invalid trade timestamp {t_raw!r} in row: {row}"
                    ) from exc
                row["seq"] = seq
                yield ts_ms, seq, row


# ---------------------  Rolling normalization  ---------------------

class RollingZScore:
    def __init__(self, window_ms: int = 10_000):
        self.window = window_ms
        self.buf = deque()
        self.sum = None
        self.sumsq = None

    def update(self, x: np.ndarray, t: int) -> np.ndarray:
        x = x.astype(np.float32)
        if self.sum is None:
            self.sum = np.zeros_like(x)
            self.sumsq = np.zeros_like(x)
        self.buf.append((t, x))
        self.sum += x
        self.sumsq += x * x
        while self.buf and t - self.buf[0][0] > self.window:
            _, x0 = self.buf.popleft()
            self.sum -= x0
            self.sumsq -= x0 * x0
        n = max(1, len(self.buf))
        mean = self.sum / n
        var = self.sumsq / n - mean * mean
        std = np.sqrt(np.clip(var, 1e-6, None))
        return (x - mean) / std


class RollingWindowStats:
    def __init__(self, window_ms: int):
        self.window_ms = int(window_ms)
        self.deq: Deque[Tuple[int, float]] = deque()
        self.sum: float = 0.0
        self.sumsq: float = 0.0

    def add(self, ts_ms: int, val: float) -> None:
        v = float(val)
        self.deq.append((int(ts_ms), v))
        self.sum += v
        self.sumsq += v * v
        while self.deq and (ts_ms - self.deq[0][0] > self.window_ms):
            _, old = self.deq.popleft()
            self.sum -= old
            self.sumsq -= old * old

    def mean_var(self) -> Tuple[float, float]:
        n = len(self.deq)
        if n <= 1:
            return 0.0, 0.0
        mean = self.sum / n
        var = (self.sumsq - (self.sum * self.sum) / n) / (n - 1)
        return mean, max(0.0, var)


class RollingScalarWindowState:
    def __init__(
        self,
        window_ms: int,
        *,
        track_sorted: bool = False,
        track_sign: bool = False,
        above_thresholds: Tuple[float, ...] = (),
    ):
        self.window_ms: int = int(window_ms)
        self.deq: Deque[Tuple[int, float, float]] = deque()
        self.n: int = 0
        self.sum: float = 0.0
        self.sumsq: float = 0.0
        self.sum_t: float = 0.0
        self.sum_t2: float = 0.0
        self.sum_ty: float = 0.0
        self.anchor_ms: Optional[int] = None

        self.track_sorted: bool = bool(track_sorted)
        self.sorted_vals: List[float] = []

        self.track_sign: bool = bool(track_sign)
        self.pos_count: int = 0
        self.neg_count: int = 0

        self.above_thresholds: Tuple[float, ...] = tuple(float(t) for t in above_thresholds)
        self.above_counts: Dict[float, int] = {float(t): 0 for t in self.above_thresholds}

    def _remove_sorted_value(self, value: float) -> None:
        idx = bisect_left(self.sorted_vals, value)
        if idx < len(self.sorted_vals) and self.sorted_vals[idx] == value:
            self.sorted_vals.pop(idx)
            return
        for j in range(max(0, idx - 3), min(len(self.sorted_vals), idx + 4)):
            if self.sorted_vals[j] == value:
                self.sorted_vals.pop(j)
                return
        for j, v in enumerate(self.sorted_vals):
            if v == value:
                self.sorted_vals.pop(j)
                return
        raise RuntimeError("RollingScalarWindowState sorted value removal failed")

    def add(self, ts_ms: int, value: float) -> None:
        v = float(value)
        if not math.isfinite(v):
            return
        ts = int(ts_ms)
        if self.anchor_ms is None:
            self.anchor_ms = ts
        t_rel_sec = (float(ts) - float(self.anchor_ms)) / 1000.0
        self.deq.append((ts, v, t_rel_sec))
        self.n += 1
        self.sum += v
        self.sumsq += v * v
        self.sum_t += t_rel_sec
        self.sum_t2 += t_rel_sec * t_rel_sec
        self.sum_ty += t_rel_sec * v
        if self.track_sorted:
            insort(self.sorted_vals, v)
        if self.track_sign:
            s = 1 if v > 0.0 else (-1 if v < 0.0 else 0)
            if s > 0:
                self.pos_count += 1
            elif s < 0:
                self.neg_count += 1
        if self.above_thresholds:
            for thr in self.above_thresholds:
                if v > thr:
                    self.above_counts[thr] += 1

    def prune(self, now_ms: int) -> None:
        cutoff = int(now_ms) - self.window_ms
        while self.deq and self.deq[0][0] < cutoff:
            _ts, v, t_rel_sec = self.deq.popleft()
            self.n -= 1
            self.sum -= v
            self.sumsq -= v * v
            self.sum_t -= t_rel_sec
            self.sum_t2 -= t_rel_sec * t_rel_sec
            self.sum_ty -= t_rel_sec * v
            if self.track_sorted:
                self._remove_sorted_value(v)
            if self.track_sign:
                s = 1 if v > 0.0 else (-1 if v < 0.0 else 0)
                if s > 0:
                    self.pos_count -= 1
                elif s < 0:
                    self.neg_count -= 1
            if self.above_thresholds:
                for thr in self.above_thresholds:
                    if v > thr:
                        self.above_counts[thr] -= 1
        if self.n <= 0:
            self.n = 0
            self.sum = 0.0
            self.sumsq = 0.0
            self.sum_t = 0.0
            self.sum_t2 = 0.0
            self.sum_ty = 0.0
            self.pos_count = 0
            self.neg_count = 0
            for thr in self.above_thresholds:
                self.above_counts[thr] = 0

    def update(self, ts_ms: int, value: float) -> None:
        self.add(ts_ms, value)
        self.prune(ts_ms)

    def mean(self) -> float:
        return self.sum / float(self.n) if self.n > 0 else 0.0

    def std(self) -> float:
        if self.n <= 0:
            return 0.0
        mean = self.sum / float(self.n)
        var = max(0.0, self.sumsq / float(self.n) - mean * mean)
        return math.sqrt(var)

    def mean_std(self) -> Tuple[float, float]:
        if self.n <= 0:
            return 0.0, 0.0
        mean = self.sum / float(self.n)
        var = max(0.0, self.sumsq / float(self.n) - mean * mean)
        return mean, math.sqrt(var)

    def min(self) -> float:
        if self.n <= 0:
            return 0.0
        if self.track_sorted:
            return float(self.sorted_vals[0])
        return float(min(v for _, v, _ in self.deq))

    def max(self) -> float:
        if self.n <= 0:
            return 0.0
        if self.track_sorted:
            return float(self.sorted_vals[-1])
        return float(max(v for _, v, _ in self.deq))

    def quantile(self, q: float) -> float:
        if self.n <= 0:
            return 0.0
        if not self.track_sorted:
            vals = sorted(v for _, v, _ in self.deq)
        else:
            vals = self.sorted_vals
        if self.n == 1:
            return float(vals[0])
        h = (self.n - 1) * float(q)
        lo = int(math.floor(h))
        hi = int(math.ceil(h))
        if lo == hi:
            return float(vals[lo])
        return float(vals[lo] * (hi - h) + vals[hi] * (h - lo))

    def p90(self) -> float:
        return self.quantile(0.90)

    def slope(self) -> float:
        n = self.n
        if n < 2:
            return 0.0
        den = self.sum_t2 - (self.sum_t * self.sum_t) / float(n)
        if den <= 1e-12 or not math.isfinite(den):
            return 0.0
        num = self.sum_ty - (self.sum_t * self.sum) / float(n)
        out = num / den
        return float(out) if math.isfinite(out) else 0.0

    def frac_above(self, threshold: float) -> float:
        if self.n <= 0:
            return 0.0
        thr = float(threshold)
        if thr in self.above_counts:
            return float(self.above_counts[thr]) / float(self.n)
        count = sum(1 for _, v, _ in self.deq if v > thr)
        return float(count) / float(self.n)

    def sum_value(self) -> float:
        return float(self.sum)

    def persistence(self, current_sign: int) -> float:
        if current_sign == 0 or self.n <= 0:
            return 0.0
        if current_sign > 0:
            return float(self.pos_count) / float(self.n)
        if current_sign < 0:
            return float(self.neg_count) / float(self.n)
        return 0.0


class RollingPriceWindowState:
    def __init__(self, window_ms: int):
        self.window_ms = int(window_ms)
        self.deq: Deque[Tuple[int, float, float, float]] = deque()
        self.anchor_ms: Optional[int] = None
        self.n: int = 0

        self.sum_t: float = 0.0
        self.sum_t2: float = 0.0
        self.sum_logp: float = 0.0
        self.sum_logp2: float = 0.0
        self.sum_t_logp: float = 0.0

        self.sorted_prices: List[float] = []


    def _remove_sorted_price(self, value: float) -> None:
        idx = bisect_left(self.sorted_prices, value)
        if idx < len(self.sorted_prices) and self.sorted_prices[idx] == value:
            self.sorted_prices.pop(idx)
            return
        for j in range(max(0, idx - 3), min(len(self.sorted_prices), idx + 4)):
            if self.sorted_prices[j] == value:
                self.sorted_prices.pop(j)
                return
        for j, v in enumerate(self.sorted_prices):
            if v == value:
                self.sorted_prices.pop(j)
                return
        raise RuntimeError("RollingPriceWindowState sorted price removal failed")

    def add(self, ts_ms: int, price: float) -> None:
        p = float(price)
        if p <= 0.0 or not math.isfinite(p):
            return
        ts = int(ts_ms)
        if self.anchor_ms is None:
            self.anchor_ms = ts
        t_rel_sec = (ts - self.anchor_ms) / 1000.0
        logp = math.log(p)

        self.deq.append((ts, p, logp, t_rel_sec))
        self.n += 1
        self.sum_t += t_rel_sec
        self.sum_t2 += t_rel_sec * t_rel_sec
        self.sum_logp += logp
        self.sum_logp2 += logp * logp
        self.sum_t_logp += t_rel_sec * logp
        insort(self.sorted_prices, p)


    def prune(self, now_ms: int) -> None:
        cutoff = int(now_ms) - self.window_ms
        while self.deq and self.deq[0][0] < cutoff:
            _ts, p, logp, t_rel_sec = self.deq.popleft()
            self.n -= 1
            self.sum_t -= t_rel_sec
            self.sum_t2 -= t_rel_sec * t_rel_sec
            self.sum_logp -= logp
            self.sum_logp2 -= logp * logp
            self.sum_t_logp -= t_rel_sec * logp
            self._remove_sorted_price(p)
        if self.n <= 0:
            self.n = 0
            self.sum_t = 0.0
            self.sum_t2 = 0.0
            self.sum_logp = 0.0
            self.sum_logp2 = 0.0
            self.sum_t_logp = 0.0

    def slope_bps_per_sec(self) -> float:
        if self.n < 3:
            return 0.0
        n = float(self.n)
        x_mean = self.sum_t / n
        y_mean = self.sum_logp / n
        x_var = self.sum_t2 / n - x_mean * x_mean
        if x_var <= 1e-12:
            return 0.0
        cov = self.sum_t_logp / n - x_mean * y_mean
        return float(1e4 * cov / x_var)

    def range_bps(self) -> float:
        if self.n < 3:
            return 0.0
        low = self.sorted_prices[0]
        high = self.sorted_prices[-1]
        return float(1e4 * math.log(high / low)) if high > 0.0 and low > 0.0 else 0.0


class PriceAsofHistory:
    def __init__(self, keep_ms: int):
        self.keep_ms = int(keep_ms)
        self.ts: List[int] = []
        self.values: List[float] = []
        self.start_idx: int = 0

    def append(self, ts_ms: int, value: float) -> None:
        ts = int(ts_ms)
        v = float(value)
        if not math.isfinite(v):
            return
        self.ts.append(ts)
        self.values.append(v)
        cutoff = ts - self.keep_ms
        self.start_idx = bisect_left(self.ts, cutoff, lo=self.start_idx)
        if self.start_idx > 0 and (self.start_idx >= 4096 or self.start_idx >= len(self.ts) // 2):
            self.ts = self.ts[self.start_idx:]
            self.values = self.values[self.start_idx:]
            self.start_idx = 0

    def asof(self, ts_query: int) -> Optional[float]:
        if not self.ts:
            return None
        idx = bisect_right(self.ts, int(ts_query), lo=self.start_idx) - 1
        if idx < self.start_idx:
            return None
        return float(self.values[idx])


@dataclass
class LargeTradeWindowState:
    window_ms: int
    max_heap: List[Tuple[float, int, float]]

    @classmethod
    def create(cls, window_ms: int) -> "LargeTradeWindowState":
        return cls(
            window_ms=int(window_ms),
            max_heap=[],
        )


@dataclass
class CVDWindowState:
    window_ms: int
    points: Deque[Tuple[int, float, float]]
    asof_before_window_value: float
    n: int
    sum_t: float
    sum_y: float
    sum_t2: float
    sum_ty: float
    anchor_ms: Optional[int]

    @classmethod
    def create(cls, window_ms: int, initial_cvd: float = 0.0) -> "CVDWindowState":
        return cls(int(window_ms), deque(), float(initial_cvd), 0, 0.0, 0.0, 0.0, 0.0, None)

    def add(self, ts_ms: int, cvd_value: float) -> None:
        ts = int(ts_ms)
        y = float(cvd_value)
        if not math.isfinite(y):
            raise ValueError(f"Non-finite CVD value for window={self.window_ms}: {cvd_value!r}")
        if self.anchor_ms is None:
            self.anchor_ms = ts
        t_rel_sec = (float(ts) - float(self.anchor_ms)) / 1000.0
        self.points.append((ts, y, t_rel_sec))
        self.n += 1
        self.sum_t += t_rel_sec
        self.sum_y += y
        self.sum_t2 += t_rel_sec * t_rel_sec
        self.sum_ty += t_rel_sec * y

    def _remove_left_point(self) -> Tuple[int, float, float]:
        ts, y, t_rel_sec = self.points.popleft()
        self.n -= 1
        self.sum_t -= t_rel_sec
        self.sum_y -= y
        self.sum_t2 -= t_rel_sec * t_rel_sec
        self.sum_ty -= t_rel_sec * y
        if self.n <= 0:
            self.n = 0
            if abs(self.sum_t) < 1e-9:
                self.sum_t = 0.0
            if abs(self.sum_y) < 1e-9:
                self.sum_y = 0.0
            if abs(self.sum_t2) < 1e-9:
                self.sum_t2 = 0.0
            if abs(self.sum_ty) < 1e-9:
                self.sum_ty = 0.0
        return ts, y, t_rel_sec

    def prune(self, now_ms: int) -> None:
        cutoff = int(now_ms) - int(self.window_ms)
        while self.points and self.points[0][0] < cutoff:
            _ts, y, _t_rel_sec = self._remove_left_point()
            self.asof_before_window_value = y

    def asof_cutoff_value(self, now_ms: int) -> float:
        cutoff = int(now_ms) - int(self.window_ms)
        baseline = float(self.asof_before_window_value)
        for ts, y, _t_rel_sec in self.points:
            if ts == cutoff:
                baseline = y
                continue
            if ts > cutoff:
                break
        return float(baseline)

    def slope_usd_per_sec(self) -> float:
        if self.n < 3:
            return 0.0
        n = float(self.n)
        den = self.sum_t2 - (self.sum_t * self.sum_t) / n
        if den <= 1e-12 or not math.isfinite(den):
            return 0.0
        num = self.sum_ty - (self.sum_t * self.sum_y) / n
        slope = num / den
        return float(slope) if math.isfinite(float(slope)) else 0.0

    def change_usd(self, now_ms: int, current_cvd: float) -> float:
        return float(current_cvd) - self.asof_cutoff_value(now_ms)

    def current_points_debug(self) -> List[Tuple[int, float, float]]:
        return list(self.points)


@dataclass
class RollingReturnDistributionState:
    window_ms: int
    deq: Deque[Tuple[int, float, float, int]]
    n: int
    sum1: float
    sum2: float
    sum3: float
    sum4: float
    up_sumsq: float
    down_sumsq: float
    bipower: float
    max_abs_q: Deque[Tuple[int, int, float]]
    seq: int



# -------------------------  Feature transforms v2  -------------------------
XFORM_HL_FAST_MS = 5_000
XFORM_HL_MEDIUM_MS = 30_000
XFORM_HL_SLOW_MS = 120_000
AUX_TRANSFORM = "prelog1p_no_ewma_v1"


class RawTransformKind(IntEnum):
    IDENTITY = 0
    LOG1P_POS = 1
    SIGNED_LOG1P = 2
    FIXED_SCALE = 3
    TANH_SCALE = 4
    LOG1P_MS = 5


class NormalizeKind(IntEnum):
    NONE = 0
    EWMA_Z = 1


@dataclass(frozen=True)
class FeatureTransformSpec:
    name: str
    raw_transform: RawTransformKind
    normalize: NormalizeKind
    half_life_ms: int
    scale: float
    input_clip_abs: float
    output_clip_abs: float

    def __post_init__(self) -> None:
        if self.normalize == NormalizeKind.NONE and int(self.half_life_ms) != 0:
            raise ValueError(f"half_life_ms must be 0 when normalize=NONE for {self.name!r}")
        if self.normalize == NormalizeKind.EWMA_Z and int(self.half_life_ms) not in {XFORM_HL_FAST_MS, XFORM_HL_MEDIUM_MS, XFORM_HL_SLOW_MS}:
            raise ValueError(f"invalid EWMA half_life_ms={self.half_life_ms} for {self.name!r}")
        if self.raw_transform in {RawTransformKind.FIXED_SCALE, RawTransformKind.TANH_SCALE} and float(self.scale) <= 0.0:
            raise ValueError(f"scale must be positive for {self.raw_transform.name} on {self.name!r}")
        if float(self.input_clip_abs) < 0.0:
            raise ValueError(f"input_clip_abs must be nonnegative for {self.name!r}")
        if float(self.output_clip_abs) <= 0.0:
            raise ValueError(f"output_clip_abs must be positive for {self.name!r}")


def build_feature_transform_specs(feature_names: Sequence[str]) -> List[FeatureTransformSpec]:
    def spec(name: str, raw: RawTransformKind, norm: NormalizeKind, hl: int = 0, scale: float = 1.0, inp: float = 0.0, out: float = FEATURE_TRANSFORM_OUTPUT_CLIP_DEFAULT) -> FeatureTransformSpec:
        return FeatureTransformSpec(name, raw, norm, int(hl), float(scale), float(inp), float(out))

    def clip(name, out=8.0): return spec(name, RawTransformKind.IDENTITY, NormalizeKind.NONE, out=out)
    def bounded(name, out=1.5): return spec(name, RawTransformKind.IDENTITY, NormalizeKind.NONE, out=out)
    def ratio(name, out=5.0): return spec(name, RawTransformKind.IDENTITY, NormalizeKind.NONE, out=out)
    def bps(name, scale=2.0, out=8.0): return spec(name, RawTransformKind.FIXED_SCALE, NormalizeKind.NONE, scale=scale, out=out)
    def bps_tanh(name, scale=2.0, out=1.5): return spec(name, RawTransformKind.TANH_SCALE, NormalizeKind.NONE, scale=scale, out=out)
    def log_ewma(name, hl, out=8.0): return spec(name, RawTransformKind.LOG1P_POS, NormalizeKind.EWMA_Z, hl=hl, out=out)
    def log_no_norm(name, out=20.0): return spec(name, RawTransformKind.LOG1P_POS, NormalizeKind.NONE, out=out)
    def signed_log_ewma(name, hl, out=8.0): return spec(name, RawTransformKind.SIGNED_LOG1P, NormalizeKind.EWMA_Z, hl=hl, out=out)
    def log_ms(name, out=8.0): return spec(name, RawTransformKind.LOG1P_MS, NormalizeKind.NONE, out=out)

    specs: List[FeatureTransformSpec] = []
    for name in feature_names:
        s: Optional[FeatureTransformSpec] = None
        if name in {"utc_hour_sin", "utc_hour_cos", "utc_dow_sin", "utc_dow_cos", "is_weekend"}:
            s = bounded(name, out=1.0)
        elif name in {"bid_l1_notional_usd", "ask_l1_notional_usd", "bid_depth_notional_5bps", "ask_depth_notional_5bps", "total_depth_notional_5bps"}:
            s = log_no_norm(name, out=20.0)
        elif name.startswith("max_abs_return_bps_"):
            s = log_ewma(name, XFORM_HL_MEDIUM_MS)
        elif name.startswith("obi_l") and ("_mean_" in name):
            s = bounded(name)
        elif name.startswith("obi_l"):
            s = bounded(name)
        elif (
            name.startswith("depth_imbalance_within_")
            or name.startswith("trade_imbalance_notional_")
            or name.startswith("regime_flow_imbalance_")
            or name.startswith("depth_imbalance_5bps_mean_")
            or name.startswith("spread_time_above_1bp_frac_")
            or name.startswith("down_up_vol_ratio_")
            or name == "micro_minus_mid_over_spread"
        ):
            s = ratio(name) if name == "micro_minus_mid_over_spread" else bounded(name)
        elif name.startswith("spread_z_") or name.startswith("depth_5bps_z_"):
            s = clip(name)
        elif name.startswith("time_since") or name.startswith("time_since_last_"):
            s = log_ms(name, out=10.0)
        elif name in {"last_trade_side_sign", "last_tick_sign", "last_is_zero_tick", "last_is_rpi"}:
            s = bounded(name, out=1.0)
        elif name.startswith("trade_toxicity_") or name.startswith("tick_imbalance_") or name.startswith("tick_sign_imbalance_") or name.startswith("zero_tick_fraction_"):
            s = bounded(name)
        elif name.startswith("consecutive_buy_trade_count") or name.startswith("consecutive_sell_trade_count"):
            s = log_ewma(name, XFORM_HL_FAST_MS)
        elif name in {"ofi_l1", "ofi_l3", "ofi_l5", "ofi_l10"}:
            s = signed_log_ewma(name, XFORM_HL_FAST_MS)
        elif (name.startswith("ofi_l") and ("_over_depth_" in name or "_sum_over_depth_" in name or "_accel_" in name)) or name.startswith("ofi_l1_pressure_over_depth_5bps_") or name.startswith("ofi_l1_pressure_over_realized_vol_"):
            s = bps_tanh(name, scale=1.0)
        elif name.startswith("ofi_l1_pressure_ewma_"):
            s = signed_log_ewma(name, XFORM_HL_FAST_MS)
        elif name.startswith("spread_delta_over_spread_"):
            s = bps_tanh(name, scale=1.0)
        elif name.startswith("mid_ret_bps_") or name.startswith("micro_ret_bps_"):
            s = bps(name, scale=2.0)
        elif name.startswith("mid_slope_bps_per_sec_") or name.startswith("micro_l5_slope_") or name.startswith("spread_widening_slope_bps_per_sec_"):
            s = bps(name, scale=10.0)
        elif name == "micro_premia":
            s = FeatureTransformSpec(
                name="micro_premia",
                raw_transform=RawTransformKind.IDENTITY,
                normalize=NormalizeKind.NONE,
                half_life_ms=0,
                scale=1.0,
                input_clip_abs=0.0,
                output_clip_abs=1.5,
            )
        elif name == "micro_minus_mid_bps" or name.startswith("micro_l") and name.endswith("_minus_mid_bps") or name.startswith("vamp_l") and name.endswith("_minus_mid_bps") or name == "micro_l1_minus_micro_l10_bps":
            s = bps(name, scale=2.0)
        elif name.startswith("vwap_vs_mid_bps_") or name.startswith("signed_trade_premium_bps_volume_weighted_"):
            s = bps(name, scale=3.0)
        elif name.startswith("depth_imbalance_5bps_slope_"):
            s = bps(name, scale=1.0)
        elif name in {"spread_bps", "gap_a_bps", "gap_b_bps"}:
            s = log_ewma(name, XFORM_HL_SLOW_MS)
        elif name.startswith("mid_range_bps_") or name.startswith("return_std_bps_") or name.startswith("regime_realized_vol_bps_"):
            s = log_ewma(name, XFORM_HL_MEDIUM_MS)
        elif name in {"bsz1", "asz1"} or name.startswith("bid_depth_within_") or name.startswith("ask_depth_within_"):
            s = log_ewma(name, XFORM_HL_MEDIUM_MS)
        elif name.startswith("regime_volume_ewma_"):
            s = log_ewma(name, XFORM_HL_SLOW_MS)
        elif (
            name.startswith("spread_change_count_")
            or name.startswith("bid_price_change_rate_")
            or name.startswith("ask_price_change_rate_")
            or name.startswith("ob_update_count_")
            or name.startswith("ob_update_rate_")
            or name.startswith("trade_count_")
            or name.startswith("trade_count_per_second_")
            or name.startswith("buy_trade_count_")
            or name.startswith("sell_trade_count_")
        ):
            s = log_ewma(name, XFORM_HL_FAST_MS)
        elif (
            name.startswith("l1_bid_depletion_rate_")
            or name.startswith("l1_ask_depletion_rate_")
            or name.startswith("l1_bid_add_rate_")
            or name.startswith("l1_ask_add_rate_")
            or name.startswith("bid_l1_depletion_")
            or name.startswith("ask_l1_depletion_")
            or name.startswith("bid_l1_depletion_over_depth_")
            or name.startswith("ask_l1_depletion_over_depth_")
            or name.startswith("bid_l1_add_rate_over_depth_")
            or name.startswith("ask_l1_add_rate_over_depth_")
            or name.startswith("bid_l1_rem_rate_over_depth_")
            or name.startswith("ask_l1_rem_rate_over_depth_")
            or "replen" in name
        ):
            s = log_ewma(name, XFORM_HL_FAST_MS)
        elif name.startswith("cvd_imbalance_") or name.startswith("signed_trade_count_imbalance_"):
            s = bounded(name)
        elif name.startswith("signed_notional_flow_usd_") or name.startswith("cvd_") or name.startswith("max_signed_trade_notional_usd_"):
            s = signed_log_ewma(name, XFORM_HL_MEDIUM_MS)
        elif name == "last_trade_notional_usd" or name.startswith("top5_trade_notional_sum_usd_") or name.startswith("top_trade_notional_sum_usd_"):
            s = log_ewma(name, XFORM_HL_MEDIUM_MS)
        elif name.startswith("buy_flow_without_price_up_") or name.startswith("sell_flow_without_price_down_") or name.startswith("absorption_bid_") or name.startswith("absorption_ask_"):
            # These retained absorption/no-price-move features are strictly nonnegative scaled notional measures.
            s = log_ewma(name, XFORM_HL_FAST_MS)
        if s is None:
            raise ValueError(f"No transform spec assigned for feature {name!r}")
        specs.append(s)

    if len(specs) != len(feature_names):
        raise ValueError(f"Feature transform spec length mismatch: {len(specs)} != {len(feature_names)}")
    if len({spec.name for spec in specs}) != len(specs):
        raise ValueError("Duplicate feature transform spec names")
    return specs


def feature_transform_spec_records(feature_names: Sequence[str]) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for s in build_feature_transform_specs(feature_names):
        records.append({
            "name": s.name,
            "raw_transform": s.raw_transform.name,
            "normalize": s.normalize.name,
            "half_life_ms": int(s.half_life_ms),
            "scale": float(s.scale),
            "input_clip_abs": float(s.input_clip_abs),
            "output_clip_abs": float(s.output_clip_abs),
        })
    return records


def feature_transform_spec_hash(feature_names: Sequence[str]) -> str:
    records = feature_transform_spec_records(feature_names)
    return hashlib.sha256(json.dumps(records, sort_keys=True).encode()).hexdigest()[:12]


class FeatureTransformEngine:
    def __init__(self, feature_names: Sequence[str], *, enable_diagnostics: bool = True):
        self.feature_names = list(feature_names)
        self.specs = build_feature_transform_specs(self.feature_names)
        self.dim = len(self.specs)
        self.enable_diagnostics = bool(enable_diagnostics)
        self.raw_kind = np.asarray([int(s.raw_transform) for s in self.specs], dtype=np.int32)
        self.norm_kind = np.asarray([int(s.normalize) for s in self.specs], dtype=np.int32)
        self.half_life_ms = np.asarray([float(s.half_life_ms) for s in self.specs], dtype=np.float64)
        self.scale = np.asarray([float(s.scale) for s in self.specs], dtype=np.float64)
        self.input_clip_abs = np.asarray([float(s.input_clip_abs) for s in self.specs], dtype=np.float64)
        self.output_clip_abs = np.asarray([float(s.output_clip_abs) for s in self.specs], dtype=np.float64)
        self.ewma_mask = self.norm_kind == int(NormalizeKind.EWMA_Z)
        self.mean = np.zeros(self.dim, dtype=np.float64)
        self.m2 = np.zeros(self.dim, dtype=np.float64)
        self.obs_count = np.zeros(self.dim, dtype=np.int64)
        self.initialized = False
        self.last_apply_dt_ms: float = 0.0
        self.raw_nonfinite_count = np.zeros(self.dim, dtype=np.int64)
        self.input_clip_count = np.zeros(self.dim, dtype=np.int64)
        self.output_clip_count = np.zeros(self.dim, dtype=np.int64)
        self.ewma_warmup_output_count = np.zeros(self.dim, dtype=np.int64)
        self.n_rows_seen = 0
        self.sum_final = np.zeros(self.dim, dtype=np.float64)
        self.sumsq_final = np.zeros(self.dim, dtype=np.float64)
        self.max_abs_final = np.zeros(self.dim, dtype=np.float64)
        self.hl_diag: Dict[int, Dict[str, float]] = {
            hl: {"feature_count": int(np.sum((self.half_life_ms == float(hl)) & self.ewma_mask)), "rows_after_warmup": 0.0, "output_clip_count": 0.0, "sum_abs_z": 0.0, "sum_z": 0.0, "sumsq_z": 0.0, "max_abs_z": 0.0}
            for hl in (XFORM_HL_FAST_MS, XFORM_HL_MEDIUM_MS, XFORM_HL_SLOW_MS)
        }
        self.non_ewma_diag: Dict[str, Dict[str, float]] = {}

    def apply(self, raw: np.ndarray, dt_ms: float) -> np.ndarray:
        dt_ms_f = max(1.0, float(dt_ms))
        self.last_apply_dt_ms = dt_ms_f
        x = np.asarray(raw, dtype=np.float64)
        if x.ndim != 1 or x.shape[0] != self.dim:
            raise ValueError(f"FeatureTransformEngine expects shape ({self.dim},), got {x.shape}")
        finite = np.isfinite(x)
        if not np.all(finite):
            self.raw_nonfinite_count += (~finite).astype(np.int64)
            raise ValueError("Non-finite raw feature values passed to FeatureTransformEngine")
        clip_mask = self.input_clip_abs > 0.0
        if np.any(clip_mask):
            before = x.copy()
            x[clip_mask] = np.clip(x[clip_mask], -self.input_clip_abs[clip_mask], self.input_clip_abs[clip_mask])
            self.input_clip_count += ((before != x) & clip_mask).astype(np.int64)
        tx = x.copy()
        for kind in RawTransformKind:
            mask = self.raw_kind == int(kind)
            if not np.any(mask):
                continue
            xm = x[mask]
            if kind == RawTransformKind.IDENTITY:
                tx[mask] = xm
            elif kind == RawTransformKind.LOG1P_POS:
                tx[mask] = np.log1p(np.maximum(xm, 0.0))
            elif kind == RawTransformKind.SIGNED_LOG1P:
                tx[mask] = np.sign(xm) * np.log1p(np.abs(xm))
            elif kind == RawTransformKind.FIXED_SCALE:
                tx[mask] = xm / self.scale[mask]
            elif kind == RawTransformKind.TANH_SCALE:
                tx[mask] = np.tanh(xm / self.scale[mask])
            elif kind == RawTransformKind.LOG1P_MS:
                tx[mask] = np.log1p(np.maximum(xm, 0.0))
        out = tx.copy()
        if np.any(self.ewma_mask):
            warm = self.obs_count < int(FEATURE_TRANSFORM_WARMUP_ROWS)
            score_mask = self.ewma_mask & (~warm)
            out[self.ewma_mask & warm] = 0.0
            self.ewma_warmup_output_count += (self.ewma_mask & warm).astype(np.int64)
            if np.any(score_mask):
                var = np.maximum(self.m2[score_mask] - self.mean[score_mask] * self.mean[score_mask], 1e-9)
                out[score_mask] = (tx[score_mask] - self.mean[score_mask]) / np.sqrt(var)
        preclip_out = out.copy()
        out = np.clip(out, -self.output_clip_abs, self.output_clip_abs)
        clipped = preclip_out != out
        self.output_clip_count += clipped.astype(np.int64)
        if np.any(self.ewma_mask):
            ew = self.ewma_mask
            if not self.initialized:
                self.mean[ew] = tx[ew]
                self.m2[ew] = tx[ew] * tx[ew]
                self.obs_count[ew] += 1
            else:
                alpha = np.zeros(self.dim, dtype=np.float64)
                alpha[ew] = 1.0 - np.power(0.5, dt_ms_f / self.half_life_ms[ew])
                self.mean[ew] = (1.0 - alpha[ew]) * self.mean[ew] + alpha[ew] * tx[ew]
                self.m2[ew] = (1.0 - alpha[ew]) * self.m2[ew] + alpha[ew] * (tx[ew] * tx[ew])
                self.obs_count[ew] += 1
        self.initialized = True
        self.n_rows_seen += 1
        self.sum_final += out
        self.sumsq_final += out * out
        self.max_abs_final = np.maximum(self.max_abs_final, np.abs(out))
        for hl, d in self.hl_diag.items():
            mask = self.ewma_mask & (self.half_life_ms == float(hl)) & (self.obs_count > FEATURE_TRANSFORM_WARMUP_ROWS)
            if np.any(mask):
                vals = out[mask]
                d["rows_after_warmup"] += int(np.sum(mask))
                d["output_clip_count"] += int(np.sum(clipped[mask]))
                d["sum_abs_z"] += float(np.sum(np.abs(vals)))
                d["sum_z"] += float(np.sum(vals))
                d["sumsq_z"] += float(np.sum(vals * vals))
                d["max_abs_z"] = max(float(d["max_abs_z"]), float(np.max(np.abs(vals))))
        for kind in RawTransformKind:
            mask = (~self.ewma_mask) & (self.raw_kind == int(kind))
            if np.any(mask):
                d = self.non_ewma_diag.setdefault(kind.name, {"feature_count": int(np.sum(mask)), "rows": 0.0, "sum_abs": 0.0, "sumsq": 0.0, "max_abs": 0.0})
                vals = out[mask]
                d["rows"] += int(np.sum(mask))
                d["sum_abs"] += float(np.sum(np.abs(vals)))
                d["sumsq"] += float(np.sum(vals * vals))
                d["max_abs"] = max(float(d["max_abs"]), float(np.max(np.abs(vals))))
        if not np.all(np.isfinite(out)):
            raise ValueError("Non-finite transformed feature values produced")
        return out.astype(np.float32)

    def diagnostics_summary(self) -> Dict[str, Any]:
        rows = max(1, int(self.n_rows_seen))
        counts_raw = {kind.name: int(np.sum(self.raw_kind == int(kind))) for kind in RawTransformKind}
        counts_norm = {kind.name: int(np.sum(self.norm_kind == int(kind))) for kind in NormalizeKind}
        counts_hl = {str(hl): int(np.sum((self.half_life_ms == float(hl)) & self.ewma_mask)) for hl in (XFORM_HL_FAST_MS, XFORM_HL_MEDIUM_MS, XFORM_HL_SLOW_MS)}
        out_frac = self.output_clip_count.astype(np.float64) / float(rows)
        def rows_for(th: float) -> List[Dict[str, Any]]:
            return [{"idx": int(i), "name": self.feature_names[int(i)], "output_clip_frac": float(out_frac[int(i)])} for i in np.flatnonzero(out_frac > th)]
        top_idx = sorted(range(self.dim), key=lambda i: float(out_frac[i]), reverse=True)[:20]
        hl_summary: Dict[str, Any] = {}
        for hl, d in self.hl_diag.items():
            denom = max(1.0, float(d["rows_after_warmup"]))
            mean = float(d["sum_z"] / denom)
            std = math.sqrt(max(0.0, float(d["sumsq_z"] / denom) - mean * mean))
            abs_mean = float(d["sum_abs_z"] / denom)
            clip_frac = float(d["output_clip_count"] / denom)
            if clip_frac > 0.01:
                warning = "too_many_clips_or_half_life_too_slow"
            elif std < 0.5 and d["rows_after_warmup"] > 0:
                warning = "possibly_too_fast_or_overcompressed"
            elif std > 2.0:
                warning = "possibly_too_slow_or_undercompressed"
            else:
                warning = "ok"
            hl_summary[str(hl)] = {"feature_count": int(d["feature_count"]), "std_mean": std, "abs_mean": abs_mean, "clip_frac": clip_frac, "max_abs": float(d["max_abs_z"]), "warning": warning}
        feature_rows = []
        means = self.sum_final / float(rows)
        stds = np.sqrt(np.maximum(0.0, self.sumsq_final / float(rows) - means * means))
        for i, sp in enumerate(self.specs):
            feature_rows.append({
                "idx": int(i), "name": sp.name, "raw_transform": sp.raw_transform.name, "normalize": sp.normalize.name,
                "half_life_ms": int(sp.half_life_ms), "output_clip_abs": float(sp.output_clip_abs),
                "output_clip_frac": float(out_frac[i]), "warmup_frac": float(self.ewma_warmup_output_count[i] / float(rows)),
                "final_mean": float(means[i]), "final_std": float(stds[i]), "final_max_abs": float(self.max_abs_final[i]),
            })
        return {
            "version": "feature_transform_diag_v1",
            "rows_seen": int(self.n_rows_seen),
            "feature_count": int(self.dim),
            "counts_by_raw_transform": counts_raw,
            "counts_by_normalize": counts_norm,
            "counts_by_half_life_ms": counts_hl,
            "clip_summary": {
                "features_with_output_clip_frac_gt_0p001": rows_for(0.001),
                "features_with_output_clip_frac_gt_0p01": rows_for(0.01),
                "top20_output_clip_frac": [{"idx": int(i), "name": self.feature_names[int(i)], "output_clip_frac": float(out_frac[int(i)])} for i in top_idx],
            },
            "half_life_summary": hl_summary,
            "feature_rows": feature_rows,
        }

# -------------------------  Feature engine  -------------------------
class FeatureEngine:

    @dataclass(frozen=True)
    class OBSnapshot:
        # Developer note:
        # - Snapshot history is only for OB horizon-based state(t) vs state(t-h) comparisons.
        # - Instantaneous OB features are always computed from the current in-memory book state.
        # - Trade-history / rolling event-time features (deques, EWMAs, trade windows) do not use snapshots.
        ts_ms: int
        bid1: float
        ask1: float
        bsz1: float
        asz1: float
        spread: float
        cum_bid3: float
        cum_ask3: float
        cum_bid5: float
        cum_ask5: float
    
    def __init__(
        self,
        depth: int = MAX_BOOK_FEATURE_LEVEL,
    ):
        # feature_depth controls the cached top-of-book ladders used for feature extraction.
        # The full source snapshot/update book is stored in self.bids/self.asks so deeper
        # levels can promote into the feature ladders after top-level deletes.
        self.feature_depth = int(depth)
        self.book_state_depth: Optional[int] = None
        if self.feature_depth < MAX_BOOK_FEATURE_LEVEL:
            raise ValueError(
                f"FeatureEngine feature_depth={self.feature_depth} is too small for "
                f"BOOK_DEPTH_FEATURE_LEVELS={BOOK_DEPTH_FEATURE_LEVELS}. "
                f"Need feature_depth >= {MAX_BOOK_FEATURE_LEVEL}."
            )
        # ---------- Book state ----------
        self.bids: Dict[float, float] = {}
        self.asks: Dict[float, float] = {}
        self.bid_lvls: List[Tuple[float, float]] = []  # sorted desc by price
        self.ask_lvls: List[Tuple[float, float]] = []  # sorted asc by price
        self._book_dirty: bool = False
        self.prev_bsz: float = 0.0
        self.prev_asz: float = 0.0
        self.prev_bsz2: float = 0.0
        self.prev_asz2: float = 0.0
        self.prev_cum_bid_by_level: Dict[int, float] = {lvl: 0.0 for lvl in BOOK_SIGNAL_LEVELS}
        self.prev_cum_ask_by_level: Dict[int, float] = {lvl: 0.0 for lvl in BOOK_SIGNAL_LEVELS}

        # Guard against sub-interval jitter when targeting ~100ms OB cadence
        self.ob_jitter_guard_ms: int = 100

        # ---------- Time bookkeeping ----------
        self.last_ts: Optional[int] = None
        self._last_any_event_ts: Optional[int] = None
        self._last_ob_feature_ts: Optional[int] = None
        self.last_trade_ts: Optional[int] = None
        self.last_bid1_update_ts: Optional[int] = None
        self.last_ask1_update_ts: Optional[int] = None
        self.last_bid_price_change_ts: Optional[int] = None
        self.last_ask_price_change_ts: Optional[int] = None
        self.last_mid_change_ts: Optional[int] = None
        self.prev_bid1_price: Optional[float] = None
        self.prev_ask1_price: Optional[float] = None
        self.prev_mid_price_for_age: Optional[float] = None
        self.prev_spread_for_age: Optional[float] = None
        self.price_history_keep_ms = max(PRICE_WINDOWS_MS) + 5_000
        self._price_ts: Deque[int] = deque()
        self._mid_history: Deque[float] = deque()
        self._micro_history: Deque[float] = deque()
        self._mid_asof_history = PriceAsofHistory(self.price_history_keep_ms)
        self._micro_asof_history = PriceAsofHistory(self.price_history_keep_ms)
        self._price_window_mid_states: Dict[int, RollingPriceWindowState] = {
            w: RollingPriceWindowState(w) for w in PRICE_WINDOWS_MS
        }

        # ---------- Rolling return histories ----------
        # Deques of (ts_ms, logret) to compute dispersion and variance-ratio ladders.
        self.return_windows_ms: Tuple[int, ...] = FLOW_WINDOWS_MS
        self.return_histories: Dict[int, RollingWindowStats] = {
            ms: RollingWindowStats(ms) for ms in self.return_windows_ms
        }

        self.regime_windows_ms: Tuple[int, ...] = REGIME_WINDOWS_MS
        self.rv_ewma: Dict[int, float] = {ms: 0.0 for ms in self.regime_windows_ms}
        self.realized_vol: Dict[int, float] = {ms: 0.0 for ms in self.regime_windows_ms}
        self.volume_ewma: Dict[int, float] = {ms: 0.0 for ms in self.regime_windows_ms}
        self.flow_regime: Dict[int, float] = {ms: 0.0 for ms in self.regime_windows_ms}
        self.regime_return_states: Dict[int, RollingReturnDistributionState] = {
            ms: RollingReturnDistributionState(
                window_ms=ms,
                deq=deque(),
                n=0,
                sum1=0.0,
                sum2=0.0,
                sum3=0.0,
                sum4=0.0,
                up_sumsq=0.0,
                down_sumsq=0.0,
                bipower=0.0,
                max_abs_q=deque(),
                seq=0,
            )
            for ms in self.regime_windows_ms
        }
        self.last_mid_for_ret: Optional[float] = None
        self._spread_bps_history: Deque[Tuple[int, float]] = deque()
        self._bid_depth_5bps_history: Deque[Tuple[int, float]] = deque()
        self._ask_depth_5bps_history: Deque[Tuple[int, float]] = deque()
        self._depth_5bps_total_history: Deque[Tuple[int, float]] = deque()
        self._depth_5bps_imbalance_history: Deque[Tuple[int, float]] = deque()
        self._regime_metric_keep_ms = max(SPREAD_DEPTH_REGIME_WINDOWS_MS) + 5_000
        self._spread_bps_regime_states = {
            ms: RollingScalarWindowState(
                ms,
                track_sorted=True,
                above_thresholds=(1.0,),
            )
            for ms in SPREAD_DEPTH_REGIME_WINDOWS_MS
        }
        self._depth_5bps_total_regime_states = {
            ms: RollingScalarWindowState(ms)
            for ms in SPREAD_DEPTH_REGIME_WINDOWS_MS
        }
        self._depth_5bps_imbalance_regime_states = {
            ms: RollingScalarWindowState(ms)
            for ms in SPREAD_DEPTH_REGIME_WINDOWS_MS
        }
        self._bid_depth_5bps_regime_states = {
            ms: RollingScalarWindowState(ms)
            for ms in SPREAD_DEPTH_REGIME_WINDOWS_MS
        }
        self._ask_depth_5bps_regime_states = {
            ms: RollingScalarWindowState(ms)
            for ms in SPREAD_DEPTH_REGIME_WINDOWS_MS
        }

        # ---------- Spread ----------
        self.last_spread: Optional[float] = None
        self.last_spread_ts: Optional[int] = None
        self.spread_delta_windows: Tuple[int, ...] = FAST_WINDOWS_MS
        self.ob_horizon_compare_windows_ms: Tuple[int, ...] = FAST_WINDOWS_MS
        self.ob_snapshot_margin_ms: int = 200
        self._ob_snapshot_keep_ms: int = max(self.ob_horizon_compare_windows_ms) + self.ob_snapshot_margin_ms
        self._ob_snapshots: List[FeatureEngine.OBSnapshot] = []
        self._ob_snapshot_ts_ms: List[int] = []
        self._spread_change_deques: Dict[int, Deque[int]] = {
            ms: deque() for ms in self.spread_delta_windows
        }

        # ---------- Best-level churn & depletion ----------
        self.bestlvl_windows: Tuple[int, ...] = FAST_WINDOWS_MS
        self._bid_price_change_deques: Dict[int, Deque[int]] = {ms: deque() for ms in self.bestlvl_windows}
        self._ask_price_change_deques: Dict[int, Deque[int]] = {ms: deque() for ms in self.bestlvl_windows}
        self.last_bid1 = None; self.last_ask1 = None
        self._bid_l1_depletion_deques: Dict[int, Deque[Tuple[int, float]]] = {ms: deque() for ms in self.bestlvl_windows}
        self._ask_l1_depletion_deques: Dict[int, Deque[Tuple[int, float]]] = {ms: deque() for ms in self.bestlvl_windows}
        self._bid_l1_depletion_sums: Dict[int, float] = {ms: 0.0 for ms in self.bestlvl_windows}
        self._ask_l1_depletion_sums: Dict[int, float] = {ms: 0.0 for ms in self.bestlvl_windows}

        # ---------- Liquidity replenishment tracking (L1/L2) ----------
        self.replen_windows_ms: Tuple[int, ...] = FAST_WINDOWS_MS
        self._replen_keys: Tuple[Tuple[str, int, str], ...] = tuple(
            (side, level, kind)
            for side in ("bid", "ask")
            for level in (1, 2)
            for kind in ("add", "rem")
        )
        self.replen_deques: Dict[int, Dict[Tuple[str, int, str], Deque[Tuple[int, float]]]] = {
            window: {key: deque() for key in self._replen_keys}
            for window in self.replen_windows_ms
        }
        self.replen_sums: Dict[int, Dict[Tuple[str, int, str], float]] = {
            window: {key: 0.0 for key in self._replen_keys}
            for window in self.replen_windows_ms
        }

        # ---------- Trades windows ----------
        # self.trade_windows includes FLOW_WINDOWS_MS plus REGIME_WINDOWS_MS because
        # 3000ms trade stats are needed for regime features. Large-trade and CVD
        # states are intentionally restricted to FLOW_WINDOWS_MS because only those
        # windows are emitted as features.
        # (ts_ms, price, size, notional_usd, side, side_sign, tick_sign, is_zero_tick)
        self.trade_windows: Tuple[int, ...] = tuple(sorted(set(FLOW_WINDOWS_MS) | set(REGIME_WINDOWS_MS)))
        self._trade_window_deques: Dict[int, Deque[Tuple[int, float, float, float, str, float, float, float]]] = {
            ms: deque() for ms in self.trade_windows
        }
        self._trade_seq: int = 0
        self.trade_window_state: Dict[int, Dict[str, Any]] = {
            window: self._new_trade_window_state()
            for window in self.trade_windows
        }
        self.large_trade_windows: Tuple[int, ...] = FLOW_WINDOWS_MS
        self.large_trade_states: Dict[int, LargeTradeWindowState] = {
            ms: LargeTradeWindowState.create(ms) for ms in self.large_trade_windows
        }
        self.cvd_notional = 0.0
        self._cvd_ema = {ms: 0.0 for ms in FLOW_WINDOWS_MS}
        self._cvd_ema_initialized = {ms: False for ms in FLOW_WINDOWS_MS}
        self.cvd_windows: Tuple[int, ...] = FLOW_WINDOWS_MS
        self.cvd_window_states: Dict[int, CVDWindowState] = {
            ms: CVDWindowState.create(ms, initial_cvd=0.0) for ms in self.cvd_windows
        }
        self.last_cvd_update_ts: Optional[int] = None
        self.consecutive_buy_trade_count: int = 0
        self.consecutive_sell_trade_count: int = 0
        self.last_ob_ofi_l5: float = 0.0
        self.last_ob_trade_imbalance_1000ms: float = 0.0

        # Tick-direction & RPI tracking
        self.last_tick_sign: int = 0
        self.last_is_zero_tick: int = 0
        self.last_trade_price: Optional[float] = None
        self.last_is_rpi: int = 0
        self.last_trade_side_sign: float = 0.0
        self.last_trade_notional_usd: float = 0.0
        self.last_buy_trade_ts: Optional[int] = None
        self.last_sell_trade_ts: Optional[int] = None

        # ---------- Quote / OB update windows ----------
        self._quote_window_deques: Dict[int, Deque[int]] = {
            ms: deque() for ms in FLOW_WINDOWS_MS
        }
        self._ob_update_deques: Dict[int, Deque[int]] = {
            ms: deque() for ms in FAST_WINDOWS_MS
        }

        # ---------- Event density windows ----------
        self._event_density_deques: Dict[int, Deque[int]] = {
            ms: deque() for ms in EVENT_DENSITY_WINDOWS_MS
        }

        # ---------- Decayed pressure (EWMA of OFI L1) ----------
        self.ofi_pressure_by_window: Dict[int, float] = {ms: 0.0 for ms in FAST_WINDOWS_MS}
        self.ofi_level_histories: Dict[int, Deque[Tuple[int, float]]] = {
            level: deque() for level in ROLLING_OFI_LEVELS
        }
        self.obi_level_histories: Dict[int, Deque[Tuple[int, float]]] = {
            level: deque() for level in ROLLING_OBI_LEVELS
        }
        self._rolling_ofi_states = {
            (level, window): RollingScalarWindowState(window)
            for level in ROLLING_OFI_LEVELS
            for window in ROLLING_OFI_WINDOWS_MS
        }
        self._rolling_obi_states = {
            (level, window): RollingScalarWindowState(window)
            for level in ROLLING_OBI_LEVELS
            for window in ROLLING_OBI_WINDOWS_MS
        }
        self.deep_micro_histories: Dict[int, Deque[Tuple[int, float]]] = {
            5: deque(),
            10: deque(),
        }


        # ---------- Feature transform v2 state ----------
        self._feature_names_cache: Optional[List[str]] = None
        self._feature_transform_engine: Optional[FeatureTransformEngine] = None
        # Empty immutable-ish placeholder returned for trade events.
        # Trade rows return this after updating trade/flow state only, avoiding per-trade allocation.
        self._trade_fast_path_empty_feature = np.empty((0,), dtype=np.float32)
        self.trade_fast_path_count: int = 0
        self.ob_feature_build_count: int = 0
        self.strict_feature_validation = os.environ.get("BYBIT_STRICT_FEATURE_VALIDATION", "0") == "1"

    def feature_names(self) -> List[str]:
        if self._feature_names_cache is not None:
            return list(self._feature_names_cache)
        names: List[str] = []
        names.extend(CALENDAR_CONTEXT_FEATURES)
        for w in PRICE_WINDOWS_MS:
            names.extend([
                f"mid_ret_bps_{w}ms",
                f"micro_ret_bps_{w}ms",
                f"mid_slope_bps_per_sec_{w}ms",
                f"mid_range_bps_{w}ms",
            ])
        names.extend([
            "spread_bps",
            "gap_a_bps",
            "gap_b_bps",
            "bsz1",
            "asz1",
            "micro_premia",
            "micro_minus_mid_bps",
            "micro_minus_mid_over_spread",
            "time_since_trade_ms",
            "time_since_mid_change_ms",
        ])
        names.extend(NOTIONAL_CONTEXT_FEATURES)
        for lvl in BOOK_SIGNAL_LEVELS:
            names.append(f"obi_l{lvl}")
        for lvl in BOOK_SIGNAL_LEVELS:
            names.append(f"ofi_l{lvl}")
        for lvl in NORMALIZED_OFI_LEVELS:
            names.append(f"ofi_l{lvl}_over_depth_l{lvl}")
            names.append(f"ofi_l{lvl}_over_depth_5bps")
        for level in ROLLING_OFI_LEVELS:
            for window in ROLLING_OFI_WINDOWS_MS:
                names.append(f"ofi_l{level}_sum_over_depth_{window}ms")
        for level in ROLLING_OFI_LEVELS:
            for fast_ms, slow_ms in OFI_ACCEL_PAIRS_MS:
                names.append(f"ofi_l{level}_accel_{fast_ms}_minus_{slow_ms}ms")
        for level in ROLLING_OBI_LEVELS:
            for window in ROLLING_OBI_WINDOWS_MS:
                names.append(f"obi_l{level}_mean_{window}ms")
        for level in DEEP_MICRO_LEVELS:
            names.extend([
                f"micro_l{level}_minus_mid_bps",
                f"vamp_l{level}_minus_mid_bps",
            ])
        names.extend([
            "micro_l5_slope_200ms",
            "micro_l5_slope_1000ms",
            "micro_l1_minus_micro_l10_bps",
        ])
        for band in BPS_DEPTH_BANDS:
            b = self._fmt_bps_band(band)
            names.extend([
                f"bid_depth_within_{b}bps",
                f"ask_depth_within_{b}bps",
                f"depth_imbalance_within_{b}bps",
            ])
        for ms in FAST_WINDOWS_MS:
            names.extend([
                f"spread_delta_over_spread_{ms}ms",
                f"spread_change_count_{ms}ms",
                f"bid_price_change_rate_{ms}ms",
                f"ask_price_change_rate_{ms}ms",
                f"bid_l1_depletion_{ms}ms",
                f"ask_l1_depletion_{ms}ms",
                f"bid_l1_depletion_over_depth_{ms}ms",
                f"ask_l1_depletion_over_depth_{ms}ms",
            ])
        for ms in FAST_WINDOWS_MS:
            names.extend([
                f"ob_update_rate_{ms}ms",
            ])
        for ms in FAST_WINDOWS_MS:
            names.extend([
                f"bid_l1_add_rate_over_depth_{ms}ms",
                f"bid_l1_rem_rate_over_depth_{ms}ms",
                f"ask_l1_add_rate_over_depth_{ms}ms",
                f"ask_l1_rem_rate_over_depth_{ms}ms",
            ])
        for ms in FLOW_WINDOWS_MS:
            names.extend([
                f"signed_notional_flow_usd_{ms}ms",
                f"signed_trade_count_imbalance_{ms}ms",
                f"trade_imbalance_notional_{ms}ms",
                f"trade_toxicity_notional_{ms}ms",
                f"zero_tick_fraction_{ms}ms",
                f"tick_sign_imbalance_{ms}ms",
                f"trade_count_per_second_{ms}ms",
                f"vwap_vs_mid_bps_{ms}ms",
                f"signed_trade_premium_bps_volume_weighted_{ms}ms",
            ])
        names.extend([
            "last_trade_side_sign",
            "last_tick_sign",
            "last_is_zero_tick",
            "last_is_rpi",
            "last_trade_notional_usd",
            "time_since_last_buy_trade_ms",
            "time_since_last_sell_trade_ms",
        ])
        for ms in FLOW_WINDOWS_MS:
            names.extend([
                f"cvd_change_usd_{ms}ms",
                f"cvd_slope_usd_per_sec_{ms}ms",
                f"cvd_minus_ema_usd_{ms}ms",
            ])
        names.extend([
            "consecutive_buy_trade_count",
            "consecutive_sell_trade_count",
        ])
        for ms in FLOW_WINDOWS_MS:
            names.extend([
                f"max_signed_trade_notional_usd_{ms}ms",
                f"top5_trade_notional_sum_usd_{ms}ms",
            ])
        for ms in FLOW_WINDOWS_MS:
            names.extend([
                f"buy_flow_without_price_up_{ms}ms",
                f"sell_flow_without_price_down_{ms}ms",
                f"absorption_bid_{ms}ms",
                f"absorption_ask_{ms}ms",
            ])
        for ms in FLOW_WINDOWS_MS:
            names.append(f"return_std_bps_{ms}ms")
        for ms in REGIME_WINDOWS_MS:
            names.extend([
                f"regime_volume_ewma_{ms}ms",
                f"regime_realized_vol_bps_{ms}ms",
                f"regime_flow_imbalance_{ms}ms",
                f"down_up_vol_ratio_{ms}ms",
                f"max_abs_return_bps_{ms}ms",
            ])
        for ms in SPREAD_DEPTH_REGIME_WINDOWS_MS:
            names.extend([
                f"spread_z_{ms}ms",
                f"spread_widening_slope_bps_per_sec_{ms}ms",
                f"spread_time_above_1bp_frac_{ms}ms",
                f"depth_5bps_z_{ms}ms",
                f"depth_imbalance_5bps_mean_{ms}ms",
                f"depth_imbalance_5bps_slope_{ms}ms",
            ])
        for ms in FAST_WINDOWS_MS:
            names.extend([
                f"ofi_l1_pressure_ewma_{ms}ms",
                f"ofi_l1_pressure_over_depth_5bps_{ms}ms",
                f"ofi_l1_pressure_over_realized_vol_{ms}ms",
            ])
        if len(names) != len(set(names)):
            seen = set()
            duplicates = sorted({n for n in names if n in seen or seen.add(n)})
            raise ValueError(f"Duplicate feature names in 1s maker feature schema: {duplicates[:20]}")
        self._feature_names_cache = list(names)
        return list(self._feature_names_cache)

    def core_feature_dim(self) -> int:
        return len(self.feature_names())

    def feature_schema(self) -> str:
        return FEATURE_SCHEMA

    def aux_schema(self) -> str:
        return AUX_SCHEMA

    def feature_dim(self) -> int:
        return self.core_feature_dim() + AUX_DIM

    def _new_trade_window_state(self) -> Dict[str, Any]:
        return {
            # Counts
            "trade_count": 0,
            "buy_cnt": 0,
            "sell_cnt": 0,

            # Base volume
            "buy_vol": 0.0,
            "sell_vol": 0.0,
            "vol_sum": 0.0,

            # Notional
            "buy_notional": 0.0,
            "sell_notional": 0.0,
            "signed_notional": 0.0,

            # VWAP
            "pxv_sum": 0.0,

            # Tick direction
            "plus_tick": 0,
            "minus_tick": 0,
            "zero_tick": 0,

            # signed_premium_weighted =
            #   1e4 * (signed_price_notional_sum / mid - signed_notional)
            "signed_price_notional_sum": 0.0,
            "signed_notional_sum_for_premium": 0.0,


            # Monotonic max queues: entries are (ts_ms, notional_usd).
            "buy_max_q": deque(),
            "sell_max_q": deque(),
        }

    def _fmt_bps_band(self, band: float) -> str:
        if float(band).is_integer():
            return str(int(band))
        return str(float(band)).replace(".", "p")


    # -------------------------------------------------------------------------
    # Helpers (kept inside the class)
    # -------------------------------------------------------------------------
    def _alpha_half_life_ms(self, hl_ms: int) -> int:
        # we return hl_ms but compute alpha using dt at callsite
        return max(1, hl_ms)

    def _ewma_update(self, prev: float, x: float, dt_ms: float, hl_ms_cfg: int) -> float:
        # alpha = 1 - 0.5^(dt/hl)
        hl = self._alpha_half_life_ms(hl_ms_cfg)
        alpha = 1.0 - math.pow(0.5, max(1.0, dt_ms) / float(hl))
        return (1.0 - alpha) * prev + alpha * x

    def _append_metric_history(self, deq: Deque[Tuple[int, float]], ts_ms: int, value: float, keep_ms: int) -> None:
        if math.isfinite(float(value)):
            deq.append((int(ts_ms), float(value)))
        cutoff = int(ts_ms) - int(keep_ms)
        while deq and deq[0][0] < cutoff:
            deq.popleft()

    def _metric_values(self, deq: Deque[Tuple[int, float]], now_ms: int, window_ms: int) -> List[Tuple[int, float]]:
        cutoff = int(now_ms) - int(window_ms)
        now = int(now_ms)
        out: List[Tuple[int, float]] = []

        for ts, val in reversed(deq):
            ts_i = int(ts)
            if ts_i > now:
                continue
            if ts_i < cutoff:
                break
            v = float(val)
            if math.isfinite(v):
                out.append((ts_i, v))

        out.reverse()
        return out

    def _slope_from_points_no_numpy(self, points: List[Tuple[int, float]]) -> float:
        n = 0
        t0: Optional[int] = None
        sum_x = 0.0
        sum_y = 0.0
        sum_x2 = 0.0
        sum_xy = 0.0

        for ts, value in points:
            v = float(value)
            if not math.isfinite(v):
                continue
            ts_i = int(ts)
            if t0 is None:
                t0 = ts_i
            x = (float(ts_i) - float(t0)) / 1000.0
            n += 1
            sum_x += x
            sum_y += v
            sum_x2 += x * x
            sum_xy += x * v

        if n < 3:
            return 0.0

        den = sum_x2 - (sum_x * sum_x) / float(n)
        if den <= 1e-12 or not math.isfinite(den):
            return 0.0

        num = sum_xy - (sum_x * sum_y) / float(n)
        out = num / den
        return float(out) if math.isfinite(out) else 0.0


    def _return_distribution_stats(self, vals: List[float]) -> Dict[str, float]:
        if not vals:
            return {
                "realized_up_vol_bps": 0.0,
                "realized_down_vol_bps": 0.0,
                "down_up_vol_ratio": 0.0,
                "bipower_variation": 0.0,
                "jump_variation": 0.0,
                "max_abs_return_bps": 0.0,
                "return_skew": 0.0,
                "return_kurtosis": 0.0,
            }
        eps = 1e-9
        arr = np.asarray(vals, dtype=np.float64)
        up = float(math.sqrt(np.sum(np.square(arr[arr > 0.0])))) if np.any(arr > 0.0) else 0.0
        down = float(math.sqrt(np.sum(np.square(arr[arr < 0.0])))) if np.any(arr < 0.0) else 0.0
        realized_var = float(np.sum(arr * arr))
        bipower = 0.0
        if arr.shape[0] >= 2:
            bipower = float(np.sum(np.abs(arr[1:]) * np.abs(arr[:-1])))
        jump_var = max(realized_var - bipower, 0.0)
        max_abs = float(np.max(np.abs(arr)))
        mean = float(np.mean(arr))
        std = float(np.std(arr, ddof=0))
        skew = 0.0
        kurt = 0.0
        if arr.shape[0] >= 3 and std > eps:
            z = (arr - mean) / std
            skew = float(np.mean(z ** 3))
        if arr.shape[0] >= 4 and std > eps:
            z = (arr - mean) / std
            kurt = float(np.mean(z ** 4))
        out = {
            "realized_up_vol_bps": up,
            "realized_down_vol_bps": down,
            "down_up_vol_ratio": float(down / max(up, eps)),
            "bipower_variation": bipower,
            "jump_variation": jump_var,
            "max_abs_return_bps": max_abs,
            "return_skew": skew,
            "return_kurtosis": kurt,
        }
        for k, v in out.items():
            if not math.isfinite(float(v)):
                out[k] = 0.0
        return out

    def _update_trade_window_state_with_insert(
        self,
        window_ms: int,
        entry: Tuple[int, float, float, float, str, float, float, float],
    ) -> None:
        ts_ms, price, size, notional_usd, side, side_sign, tick_sign, is_zero_tick = entry
        state = self.trade_window_state[window_ms]

        ts_i = int(ts_ms)
        px = float(price)
        sz = float(size)
        notion = float(notional_usd)
        ss = float(side_sign)
        tick = float(tick_sign)
        zero = float(is_zero_tick)

        state["trade_count"] += 1
        state["pxv_sum"] += px * sz
        state["vol_sum"] += sz

        if tick > 0:
            state["plus_tick"] += 1
        elif tick < 0:
            state["minus_tick"] += 1

        if zero > 0:
            state["zero_tick"] += 1

        if ss > 0:
            state["buy_cnt"] += 1
            state["buy_vol"] += sz
            state["buy_notional"] += notion
            state["signed_notional"] += notion

            state["signed_price_notional_sum"] += px * notion
            state["signed_notional_sum_for_premium"] += notion


            q = state["buy_max_q"]
            while q and q[-1][1] <= notion:
                q.pop()
            q.append((ts_i, notion))

        elif ss < 0:
            state["sell_cnt"] += 1
            state["sell_vol"] += sz
            state["sell_notional"] += notion
            state["signed_notional"] -= notion

            state["signed_price_notional_sum"] -= px * notion
            state["signed_notional_sum_for_premium"] += notion

            q = state["sell_max_q"]
            while q and q[-1][1] <= notion:
                q.pop()
            q.append((ts_i, notion))
        if window_ms in self.large_trade_states:
            self._large_trade_state_insert(window_ms, entry)

    def _update_trade_window_state_with_expire(
        self,
        window_ms: int,
        entry: Tuple[int, float, float, float, str, float, float, float],
    ) -> None:
        ts_ms, price, size, notional_usd, side, side_sign, tick_sign, is_zero_tick = entry
        state = self.trade_window_state[window_ms]

        ts_i = int(ts_ms)
        px = float(price)
        sz = float(size)
        notion = float(notional_usd)
        ss = float(side_sign)
        tick = float(tick_sign)
        zero = float(is_zero_tick)

        state["trade_count"] -= 1
        state["pxv_sum"] -= px * sz
        state["vol_sum"] -= sz

        if tick > 0:
            state["plus_tick"] -= 1
        elif tick < 0:
            state["minus_tick"] -= 1

        if zero > 0:
            state["zero_tick"] -= 1

        if ss > 0:
            state["buy_cnt"] -= 1
            state["buy_vol"] -= sz
            state["buy_notional"] -= notion
            state["signed_notional"] -= notion

            state["signed_price_notional_sum"] -= px * notion
            state["signed_notional_sum_for_premium"] -= notion


            q = state["buy_max_q"]
            if q and q[0][0] == ts_i and abs(q[0][1] - notion) <= 1e-12:
                q.popleft()

        elif ss < 0:
            state["sell_cnt"] -= 1
            state["sell_vol"] -= sz
            state["sell_notional"] -= notion
            state["signed_notional"] += notion

            state["signed_price_notional_sum"] += px * notion
            state["signed_notional_sum_for_premium"] -= notion

            q = state["sell_max_q"]
            if q and q[0][0] == ts_i and abs(q[0][1] - notion) <= 1e-12:
                q.popleft()

        # Clamp tiny floating point residue / impossible negative counters after expiration.
        int_keys = (
            "trade_count",
            "buy_cnt",
            "sell_cnt",
            "plus_tick",
            "minus_tick",
            "zero_tick",
        )
        for key in int_keys:
            if state[key] < 0:
                state[key] = 0

        float_nonnegative_keys = (
            "buy_vol",
            "sell_vol",
            "vol_sum",
            "buy_notional",
            "sell_notional",
            "signed_notional_sum_for_premium",
        )
        for key in float_nonnegative_keys:
            if state[key] < 0.0 and abs(state[key]) <= 1e-9:
                state[key] = 0.0
            elif state[key] < 0.0:
                state[key] = 0.0

        if state["trade_count"] == 0:
            for key in (
                "buy_vol",
                "sell_vol",
                "vol_sum",
                "buy_notional",
                "sell_notional",
                "signed_notional",
                "pxv_sum",
                "signed_price_notional_sum",
                "signed_notional_sum_for_premium",
            ):
                state[key] = 0.0
            state["buy_cnt"] = 0
            state["sell_cnt"] = 0
            state["plus_tick"] = 0
            state["minus_tick"] = 0
            state["zero_tick"] = 0
            state["buy_max_q"].clear()
            state["sell_max_q"].clear()

    def _prune_trade_window(self, now_ms: int, window_ms: int) -> None:
        deq = self._trade_window_deques[window_ms]
        while deq and (now_ms - deq[0][0] > window_ms):
            expired = deq.popleft()
            self._update_trade_window_state_with_expire(window_ms, expired)

    def _prune_trade_max_queue_heads(self, q: Deque[Tuple[int, float]], cutoff_ts_ms: int) -> None:
        cutoff = int(cutoff_ts_ms)
        while q and int(q[0][0]) < cutoff:
            q.popleft()

    def _compute_trade_window_stats(self, ms: int, ts_ms: int, mid: float, micro: float) -> Dict[str, float]:
        eps = 1e-12
        state = self.trade_window_state[ms]
        cutoff_ts_ms = int(ts_ms) - int(ms)

        trade_count = float(state["trade_count"])
        buy_vol = float(state["buy_vol"])
        sell_vol = float(state["sell_vol"])
        buy_notional = float(state["buy_notional"])
        sell_notional = float(state["sell_notional"])
        buy_count = float(state["buy_cnt"])
        sell_count = float(state["sell_cnt"])
        plus_tick = float(state["plus_tick"])
        minus_tick = float(state["minus_tick"])
        zero_tick = float(state["zero_tick"])
        pxv_sum = float(state["pxv_sum"])
        vol_sum = float(state["vol_sum"])
        signed_notional = float(state["signed_notional"])

        buy_max_q = state["buy_max_q"]
        sell_max_q = state["sell_max_q"]
        self._prune_trade_max_queue_heads(buy_max_q, cutoff_ts_ms)
        self._prune_trade_max_queue_heads(sell_max_q, cutoff_ts_ms)
        buy_max_notional = float(buy_max_q[0][1]) if buy_max_q else 0.0
        sell_max_notional = float(sell_max_q[0][1]) if sell_max_q else 0.0

        signed_premium_weighted = 0.0
        signed_notional_sum_for_premium = 0.0

        if mid > 0.0:
            signed_premium_weighted = 1e4 * (
                float(state["signed_price_notional_sum"]) / mid
                - float(state["signed_notional"])
            )
            signed_notional_sum_for_premium = float(state["signed_notional_sum_for_premium"])

        tot_notional = buy_notional + sell_notional
        tot_count_signed = buy_count + sell_count
        window_sec = max(ms / 1000.0, eps)
        stats = {
            "buy_vol_base": buy_vol,
            "sell_vol_base": sell_vol,
            "buy_notional_usd": buy_notional,
            "sell_notional_usd": sell_notional,
            "buy_count": buy_count,
            "sell_count": sell_count,
            "buy_mean_notional_usd": self._safe_div(buy_notional, buy_count, 0.0),
            "sell_mean_notional_usd": self._safe_div(sell_notional, sell_count, 0.0),
            "buy_max_notional_usd": buy_max_notional,
            "sell_max_notional_usd": sell_max_notional,
            "signed_notional_flow_usd": signed_notional,
            "signed_trade_count_imbalance": self._safe_div(buy_count - sell_count, tot_count_signed, 0.0),
            "trade_imbalance_notional": self._safe_div(buy_notional - sell_notional, tot_notional, 0.0),
            "trade_toxicity_notional": self._safe_div(abs(buy_notional - sell_notional), tot_notional, 0.0),
            "plus_tick_fraction": self._safe_div(plus_tick, trade_count, 0.0),
            "minus_tick_fraction": self._safe_div(minus_tick, trade_count, 0.0),
            "zero_tick_fraction": self._safe_div(zero_tick, trade_count, 0.0),
            "tick_sign_imbalance": self._safe_div(plus_tick - minus_tick, plus_tick + minus_tick + zero_tick, 0.0),
            "trade_count": trade_count,
            "trade_count_per_second": self._safe_div(trade_count, window_sec, 0.0),
            "vwap_vs_mid_bps": 0.0,
            "signed_trade_premium_bps_volume_weighted": self._safe_div(signed_premium_weighted, signed_notional_sum_for_premium, 0.0),
            "aggressor_price_impact_bps": self._safe_div(signed_premium_weighted, signed_notional_sum_for_premium, 0.0),
        }
        if vol_sum > eps:
            vwap = pxv_sum / vol_sum
            stats["vwap_vs_mid_bps"] = (1e4 * (vwap / mid - 1.0)) if mid > 0 else 0.0
        for k, v in stats.items():
            if not math.isfinite(float(v)):
                raise ValueError(f"Non-finite trade stat {k}={v!r} window={ms}")
        return stats

    def _lin_slope(self, xs: List[float], ys: List[float], eps: float = 1e-12) -> float:
        n = len(xs)
        if n < 2:
            return 0.0
        mx = sum(xs) / n
        my = sum(ys) / n
        num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
        den = sum((x - mx) * (x - mx) for x in xs) + eps
        return num / den

    def _safe_div(self, num: float, den: float, default: float = 0.0) -> float:
        try:
            n = float(num)
            d = float(den)
        except Exception:
            return float(default)
        if not math.isfinite(n) or not math.isfinite(d) or abs(d) <= 1e-12:
            return float(default)
        out = n / d
        return float(out) if math.isfinite(out) else float(default)

    def _finite(self, x: float, default: float = 0.0) -> float:
        v = float(x)
        return v if math.isfinite(v) else float(default)

    def _bps(self, price: float, ref: float, default: float = 0.0) -> float:
        p = float(price)
        r = float(ref)
        if p > 0.0 and r > 0.0 and math.isfinite(p) and math.isfinite(r):
            out = 1e4 * (p / r - 1.0)
            return out if math.isfinite(out) else float(default)
        return float(default)


    def _weighted_side_price(self, side_levels: Sequence[Tuple[float, float]], n: int) -> Tuple[float, float]:
        px_qty = 0.0
        qty = 0.0
        for price, size in side_levels[:n]:
            p = float(price)
            q = float(size)
            if p <= 0.0 or q <= 0.0 or not math.isfinite(p) or not math.isfinite(q):
                continue
            px_qty += p * q
            qty += q
        if qty <= 1e-12:
            return 0.0, 0.0
        return float(px_qty / qty), float(qty)

    def _sign(self, x: float) -> int:
        v = float(x)
        if v > 0.0:
            return 1
        if v < 0.0:
            return -1
        return 0

    def _large_trade_state_insert(self, window_ms: int, entry: Tuple[int, float, float, float, str, float, float, float]) -> None:
        ts_ms, _price, _size, notional_usd, _side, side_sign, _tick_sign, _is_zero_tick = entry
        state = self.large_trade_states[window_ms]
        notion = float(notional_usd)
        ss = float(side_sign)
        ts_i = int(ts_ms)
        if notion > 0.0:
            heapq.heappush(state.max_heap, (-notion, ts_i, ss))


    def _large_trade_stats_from_state(self, ms: int, now_ms: int) -> Dict[str, float]:
        cutoff = int(now_ms) - int(ms)
        state = self.large_trade_states[ms]
        out: Dict[str, float] = {}
        while state.max_heap and state.max_heap[0][1] < cutoff:
            heapq.heappop(state.max_heap)
        if state.max_heap:
            notion = float(-state.max_heap[0][0])
            ss = float(state.max_heap[0][2])
            out[f"max_signed_trade_notional_usd_{ms}ms"] = ss * notion if ss != 0.0 else 0.0
        else:
            out[f"max_signed_trade_notional_usd_{ms}ms"] = 0.0
        popped: List[Tuple[float, int, float]] = []
        top_sum = 0.0
        for _ in range(5):
            while state.max_heap and state.max_heap[0][1] < cutoff:
                heapq.heappop(state.max_heap)
            if not state.max_heap:
                break
            item = heapq.heappop(state.max_heap)
            popped.append(item)
            top_sum += float(-item[0])
        for item in popped:
            heapq.heappush(state.max_heap, item)
        out[f"top5_trade_notional_sum_usd_{ms}ms"] = top_sum
        return out

    def _bps_return(self, current: float, past: float) -> float:
        c = float(current)
        p = float(past)
        if c <= 0.0 or p <= 0.0 or not math.isfinite(c) or not math.isfinite(p):
            return 0.0
        return float(1e4 * math.log(c / p))

    def _append_price_history(self, ts_ms: int, mid: float, micro: float) -> None:
        ts = int(ts_ms)
        self._price_ts.append(ts)
        self._mid_history.append(float(mid))
        self._micro_history.append(float(micro))
        cutoff = ts - int(self.price_history_keep_ms)
        while self._price_ts and self._price_ts[0] < cutoff:
            self._price_ts.popleft()
            self._mid_history.popleft()
            self._micro_history.popleft()
        self._mid_asof_history.append(ts, mid)
        self._micro_asof_history.append(ts, micro)
        for state in self._price_window_mid_states.values():
            state.add(ts, mid)
            state.prune(ts)

    def _series_asof(self, ts_query: int, series: str) -> Optional[float]:
        history = self._mid_history if series == "mid" else self._micro_history
        for ts, value in zip(reversed(self._price_ts), reversed(history)):
            if ts <= int(ts_query):
                return float(value)
        return None

    def _window_price_points(self, now_ms: int, window_ms: int, series: str = "mid") -> List[Tuple[int, float]]:
        history = self._mid_history if series == "mid" else self._micro_history
        cutoff = int(now_ms) - int(window_ms)
        now = int(now_ms)
        out: List[Tuple[int, float]] = []
        for ts, value in zip(reversed(self._price_ts), reversed(history)):
            ts_i = int(ts)
            if ts_i > now:
                continue
            if ts_i < cutoff:
                break
            v = float(value)
            if v > 0.0 and math.isfinite(v):
                out.append((ts_i, v))
        out.reverse()
        return out

    def _rolling_slope_simple(self, points: List[Tuple[int, float]]) -> float:
        return self._slope_from_points_no_numpy(points)

    def _depth_within_bps(
        self,
        levels: List[Tuple[float, float]],
        mid: float,
        band_bps: float,
        is_bid: bool,
    ) -> float:
        if mid <= 0.0 or not levels:
            return 0.0

        band = float(band_bps)
        total_size = 0.0

        for px, qty in levels:
            price = float(px)
            size = float(qty)
            if price <= 0.0 or size <= 0.0:
                continue

            if is_bid:
                dist_bps = 1e4 * max(0.0, (mid - price) / mid)
            else:
                dist_bps = 1e4 * max(0.0, (price - mid) / mid)

            if dist_bps <= band:
                total_size += size

        return float(total_size)

    def _calendar_context_features(self, ts_ms: int) -> List[float]:
        dt = datetime.fromtimestamp(int(ts_ms) / 1000.0, tz=timezone.utc)

        hour_float = (
            float(dt.hour)
            + float(dt.minute) / 60.0
            + float(dt.second) / 3600.0
            + float(dt.microsecond) / 3_600_000_000.0
        )
        hour_angle = 2.0 * math.pi * hour_float / 24.0

        dow_float = float(dt.weekday()) + hour_float / 24.0
        dow_angle = 2.0 * math.pi * dow_float / 7.0

        return [
            math.sin(hour_angle),
            math.cos(hour_angle),
            math.sin(dow_angle),
            math.cos(dow_angle),
            1.0 if dt.weekday() >= 5 else 0.0,
        ]

    def _notional_depth_within_bps(
        self,
        levels: List[Tuple[float, float]],
        mid: float,
        band_bps: float,
        *,
        is_bid: bool,
    ) -> float:
        if mid <= 0.0 or not levels:
            return 0.0

        band = float(band_bps)
        total_notional = 0.0

        for px, qty in levels:
            price = float(px)
            size = float(qty)
            if price <= 0.0 or size <= 0.0:
                continue

            if is_bid:
                dist_bps = 1e4 * max(0.0, (mid - price) / mid)
            else:
                dist_bps = 1e4 * max(0.0, (price - mid) / mid)

            if dist_bps <= band:
                total_notional += price * size

        return float(total_notional)

    def _sorted_ladders(self):
        # Rebuild the cached top-feature-depth ladders from the full in-memory book.
        self.bid_lvls = sorted(self.bids.items(), key=lambda x: x[0], reverse=True)[: self.feature_depth]
        self.ask_lvls = sorted(self.asks.items(), key=lambda x: x[0], reverse=False)[: self.feature_depth]
        self._book_dirty = False

    def _insert_level(self, levels: List[Tuple[float, float]], price: float, size: float, is_bid: bool) -> bool:
        insert_at = 0
        while insert_at < len(levels):
            px_i = levels[insert_at][0]
            if px_i == price:
                levels[insert_at] = (price, size)
                return insert_at < self.feature_depth
            if (price > px_i) if is_bid else (price < px_i):
                break
            insert_at += 1
        levels.insert(insert_at, (price, size))
        if len(levels) > self.feature_depth:
            levels.pop()
        return insert_at < self.feature_depth

    def _remove_level(self, levels: List[Tuple[float, float]], price: float) -> bool:
        for idx, (px_i, _sz_i) in enumerate(levels):
            if px_i == price:
                levels.pop(idx)
                return idx < self.feature_depth
        return False

    def _side_requires_full_rebuild(
        self,
        book: Dict[float, float],
        levels: List[Tuple[float, float]],
        price: float,
        size: float,
        is_bid: bool,
    ) -> bool:
        prev_size = book.get(price)
        was_tracked = any(px == price for px, _ in levels)
        best_price = levels[0][0] if levels else None
        boundary_price = levels[-1][0] if len(levels) >= self.feature_depth else None
        deleting = size <= 0.0

        if deleting and prev_size is None:
            return False
        if deleting and best_price is not None and price == best_price:
            return True
        if deleting:
            removed = self._remove_level(levels, price)
            if removed and len(book) >= self.feature_depth:
                return True
            return False

        book[price] = size
        if was_tracked:
            self._insert_level(levels, price, size, is_bid)
            return False
        if len(levels) < self.feature_depth:
            self._insert_level(levels, price, size, is_bid)
            return False
        if boundary_price is None:
            self._insert_level(levels, price, size, is_bid)
            return False
        enters_top = (price > boundary_price) if is_bid else (price < boundary_price)
        if enters_top:
            return True
        return False

    def _apply_side_updates(
        self,
        book: Dict[float, float],
        levels: List[Tuple[float, float]],
        updates: Sequence[Tuple[float, float]],
        is_bid: bool,
    ) -> bool:
        rebuild = False
        for price_raw, size_raw in updates:
            try:
                price = float(price_raw)
                size = float(size_raw)
            except (TypeError, ValueError):
                continue
            if not math.isfinite(price) or not math.isfinite(size):
                continue
            if price <= 0.0:
                continue
            if self._side_requires_full_rebuild(book, levels, price, size, is_bid):
                rebuild = True
            if size <= 0.0:
                book.pop(price, None)
            else:
                book[price] = size
        return rebuild

    def _ensure_book_ladders(self) -> None:
        if self._book_dirty:
            self._sorted_ladders()

    def _book_best(self) -> Tuple[float, float, float, float]:
        bid = self.bid_lvls[0][0] if self.bid_lvls else 0.0
        ask = self.ask_lvls[0][0] if self.ask_lvls else 0.0
        bsz = self.bid_lvls[0][1] if self.bid_lvls else 0.0
        asz = self.ask_lvls[0][1] if self.ask_lvls else 0.0
        return bid, ask, bsz, asz

    def _validate_book_health(self, ts_ms: int, tp_code: int) -> None:
        # Minimal invariant check only: non-empty, positive, finite, non-crossed top of book and sorted cached ladders.
        self._ensure_book_ladders()

        if not self.bid_lvls or not self.ask_lvls:
            raise BookValidationError(
                f"Book empty after OB update at ts={int(ts_ms)} "
                f"type={int(tp_code)} bid_levels={len(self.bid_lvls)} ask_levels={len(self.ask_lvls)} "
                f"full_bids={len(self.bids)} full_asks={len(self.asks)}"
            )

        bid1, ask1, bsz1, asz1 = self._book_best()

        values = (bid1, ask1, bsz1, asz1)
        if not all(math.isfinite(float(x)) for x in values):
            raise BookValidationError(
                f"Non-finite top-of-book after OB update at ts={int(ts_ms)} "
                f"type={int(tp_code)} bid1={bid1} ask1={ask1} bsz1={bsz1} asz1={asz1}"
            )

        if bid1 <= 0.0 or ask1 <= 0.0 or bsz1 <= 0.0 or asz1 <= 0.0:
            raise BookValidationError(
                f"Non-positive top-of-book after OB update at ts={int(ts_ms)} "
                f"type={int(tp_code)} bid1={bid1} ask1={ask1} bsz1={bsz1} asz1={asz1}"
            )

        if bid1 >= ask1:
            raise BookValidationError(
                f"Crossed/locked book after OB update at ts={int(ts_ms)} "
                f"type={int(tp_code)} bid1={bid1} ask1={ask1}"
            )

        for i in range(1, len(self.bid_lvls)):
            if self.bid_lvls[i - 1][0] <= self.bid_lvls[i][0]:
                raise BookValidationError(
                    f"Bid ladder not strictly descending after OB update at ts={int(ts_ms)}"
                )

        for i in range(1, len(self.ask_lvls)):
            if self.ask_lvls[i - 1][0] >= self.ask_lvls[i][0]:
                raise BookValidationError(
                    f"Ask ladder not strictly ascending after OB update at ts={int(ts_ms)}"
                )

    def _cum_depth(self, lvls: List[Tuple[float, float]], n: int) -> float:
        return float(sum(s for _, s in lvls[: n]))

    def _prune_deque_ms(self, deq: Deque[Tuple[int, Any]], now_ms: int, window_ms: int):
        while deq and (now_ms - deq[0][0] > window_ms):
            deq.popleft()

    def _entry_ts(self, entry: Any) -> int:
        return int(entry if isinstance(entry, (int, float)) else entry[0])
    
    def _append_ts_with_guard(
        self,
        deq: Deque[int],
        ts_ms: int,
        window_ms: int,
        is_ob_event: bool,
    ) -> None:
        """Append a timestamp to a deque, collapsing sub-100ms OB jitter."""
        if is_ob_event and window_ms == 100 and deq:
            last_ts = self._entry_ts(deq[-1])
            if ts_ms - last_ts < self.ob_jitter_guard_ms:
                deq.pop()
        deq.append(ts_ms)
        self._prune_ts_deque(deq, ts_ms, window_ms)

    def _append_tuple_with_guard(
        self,
        deq: Deque[Tuple[int, Any]],
        entry: Tuple[int, Any],
        ts_ms: int,
        window_ms: int,
        is_ob_event: bool,
    ) -> None:
        """Append (ts, value...) to deque, merging sub-100ms OB arrivals."""
        if is_ob_event and window_ms == 100 and deq:
            last_ts = self._entry_ts(deq[-1])
            if ts_ms - last_ts < self.ob_jitter_guard_ms:
                deq.pop()
        deq.append(entry)
        self._prune_deque_ms(deq, ts_ms, window_ms)

    def _event_density(self, deq: Deque[int], window_ms: int) -> float:
        if not deq:
            return 0.0
        now = deq[-1]
        self._prune_ts_deque(deq, now, window_ms)
        window_secs = window_ms / 1000.0
        return len(deq) / window_secs if window_secs > 0 else 0.0

    def event_density(self, window_ms: int) -> float:
        deq = self._event_density_deques.get(int(window_ms))
        if deq is None:
            return 0.0
        return self._event_density(deq, int(window_ms))

    def event_density_100ms(self) -> float:
        return self.event_density(100)

    def event_density_200ms(self) -> float:
        return self.event_density(200)

    def event_density_500ms(self) -> float:
        return self.event_density(500)

    def event_density_1000ms(self) -> float:
        return self.event_density(1_000)

    def event_density_3000ms(self) -> float:
        return self.event_density(3_000)

    def _prune_ts_deque(self, deq: Deque[int], now_ms: int, window_ms: int):
        while deq and (now_ms - deq[0] > window_ms):
            deq.popleft()

    def _prune_replen_windows(self, now_ms: int):
        for window, key_map in self.replen_deques.items():
            max_age = window
            sums_map = self.replen_sums[window]
            for key, dq in key_map.items():
                while dq and (now_ms - dq[0][0] > max_age):
                    _, val = dq.popleft()
                    sums_map[key] -= val

    def _record_replenishment(self, ts_ms: int, deltas: Dict[Tuple[str, int, str], float]):
        if not deltas:
            return
        for window in self.replen_windows_ms:
            key_map = self.replen_deques[window]
            sums_map = self.replen_sums[window]
            for key, value in deltas.items():
                if value <= 0.0:
                    continue
                dq = key_map[key]
                self._append_tuple_with_guard(dq, (ts_ms, value), ts_ms, window, is_ob_event=True)
                sums_map[key] += value

    def _append_ob_snapshot(
        self,
        ts_ms: int,
        bid1: float,
        ask1: float,
        bsz1: float,
        asz1: float,
        spread: float,
        cum_bid3: float,
        cum_ask3: float,
        cum_bid5: float,
        cum_ask5: float,
    ) -> None:
        snap = self.OBSnapshot(
            ts_ms=int(ts_ms),
            bid1=float(bid1),
            ask1=float(ask1),
            bsz1=float(bsz1),
            asz1=float(asz1),
            spread=float(spread),
            cum_bid3=float(cum_bid3),
            cum_ask3=float(cum_ask3),
            cum_bid5=float(cum_bid5),
            cum_ask5=float(cum_ask5),
        )
        self._ob_snapshots.append(snap)
        self._ob_snapshot_ts_ms.append(snap.ts_ms)
        self._prune_ob_snapshots(ts_ms)

    def _prune_ob_snapshots(self, now_ts_ms: int) -> None:
        cutoff = int(now_ts_ms) - int(self._ob_snapshot_keep_ms)
        while self._ob_snapshot_ts_ms and self._ob_snapshot_ts_ms[0] < cutoff:
            del self._ob_snapshot_ts_ms[0]
            del self._ob_snapshots[0]

    def get_ob_snapshot_asof(self, cutoff_ts_ms: int) -> Optional["FeatureEngine.OBSnapshot"]:
        # Strict as-of lookup: returns latest snapshot with snapshot.ts_ms <= cutoff_ts_ms.
        # Intentionally no nearest-neighbor-by-distance selection.
        if not self._ob_snapshot_ts_ms:
            return None
        idx = bisect_right(self._ob_snapshot_ts_ms, int(cutoff_ts_ms)) - 1
        if idx < 0:
            return None
        return self._ob_snapshots[idx]

    def _replenishment_rates(self) -> Dict[int, Dict[Tuple[str, int, str], float]]:
        rates: Dict[int, Dict[Tuple[str, int, str], float]] = {}
        for window in self.replen_windows_ms:
            scale = float(window) if window > 0 else 1.0
            rates[window] = {
                key: self.replen_sums[window][key] / scale
                for key in self.replen_sums[window]
            }
        return rates

    # -------------------------------------------------------------------------
    # Event ingestion & feature build
    # -------------------------------------------------------------------------
    def _parse_event(self, e: Any) -> Tuple[str, int, Any]:
        """
        Accepts multiple event shapes:
        - Tuple ('ob'|'trade', data:dict, ts_ms:int)
        - Tuple ('ob'|'trade', ts_ms:int, seq:int, data:dict)
        - Dict-like OB: recognized by orderbook shape (`b`/`a` arrays), not topic/type strings.
          Timestamp can be provided via `ts` or `cts`, either on the top-level event dict
          or inside `data`.
        - Dict-like trade: {'timestamp': float|str, 'price': str|float, 'size': str|float, 'side': 'Buy'|'Sell'|'buy'|'sell', ...}
        Tuple input behavior is supported and unchanged.
        Returns: (etype, ts_ms, payload)
        etype in {'ob','trade'}
        """
        # Tuple form
        if isinstance(e, tuple) and len(e) >= 4 and isinstance(e[0], str):
            etype = e[0].lower()
            if etype in {"ob", "trade"}:
                ts_ms = int(e[1])
                payload = e[3:]
                return etype, ts_ms, payload

        if isinstance(e, tuple) and len(e) == 3 and isinstance(e[0], str):
            etype = e[0].lower()
            ts_ms = int(e[2])
            return etype, ts_ms, e[1]

        if isinstance(e, dict):
            data = e.get("data", e)
            if isinstance(data, list) and data and isinstance(data[0], dict):
                data = data[0]

            is_ob = (
                isinstance(data, dict)
                and ("b" in data or "a" in data)
                and (isinstance(data.get("b", []), list) or isinstance(data.get("a", []), list))
            )

            if is_ob:
                ts_raw = e.get("ts")
                if ts_raw is None:
                    ts_raw = e.get("cts")
                if ts_raw is None and isinstance(data, dict):
                    ts_raw = data.get("ts")
                if ts_raw is None and isinstance(data, dict):
                    ts_raw = data.get("cts")
                if ts_raw is None:
                    raise ValueError(f"Missing OB timestamp in event: {e}")
                return "ob", coerce_ts_ms(ts_raw), e
            # Trade event?
            if 'price' in e and 'size' in e and 'side' in e:
                t_raw = e.get("timestamp")
                if t_raw is None:
                    t_raw = e.get("ts")
                if t_raw is None:
                    t_raw = e.get("T")
                if t_raw is None:
                    raise ValueError(f"Missing trade timestamp in event: {e}")

                try:
                    ts_ms = coerce_ts_ms(t_raw)
                except ValueError as exc:
                    raise ValueError(f"Unparseable trade timestamp in event: {e}") from exc

                return 'trade', ts_ms, e

        raise ValueError(f"Unrecognized event shape: {type(e)} :: {e}")

    def _dispatch_parsed_event(
        self,
        etype: str,
        ts_ms: int,
        payload: Any,
    ) -> FeatureEventResult:
        etype = str(etype).lower()
        ts_ms = int(ts_ms)

        any_event_dt_ms = (
            1.0
            if self._last_any_event_ts is None
            else max(1.0, float(ts_ms - int(self._last_any_event_ts)))
        )

        for w, deq in self._event_density_deques.items():
            # Mixed event-density counts all events, including same-ms trade+OB pairs.
            # Do not apply OB jitter collapse here; otherwise a same-ms OB can pop
            # the trade that preceded it.
            self._append_ts_with_guard(deq, ts_ms, w, is_ob_event=False)

        if etype == "trade":
            return self._handle_trade_event(ts_ms, payload, any_event_dt_ms)

        if etype == "ob":
            return self._handle_ob_event(ts_ms, payload, any_event_dt_ms)

        raise ValueError(f"Unsupported event type in _dispatch_parsed_event: {etype!r}")

    def _current_book_mid_or_trade_price(self, payload: Any) -> float:
        bid1, ask1, _, _ = self._book_best()
        if bid1 > 0.0 and ask1 > 0.0:
            return 0.5 * (bid1 + ask1)

        try:
            if isinstance(payload, tuple):
                return float(payload[0])
            return float(payload.get("price", 0.0))
        except Exception:
            return 0.0

    def _handle_trade_event(
        self,
        ts_ms: int,
        payload: Any,
        any_event_dt_ms: float,
    ) -> FeatureEventResult:
        self._update_trade_windows(ts_ms, payload, any_event_dt_ms)
        self.trade_fast_path_count += 1
        self._last_any_event_ts = int(ts_ms)

        raw_mid = self._current_book_mid_or_trade_price(payload)

        return FeatureEventResult(
            ts_ms=int(ts_ms),
            features=self._trade_fast_path_empty_feature,
            dt_ms=float(any_event_dt_ms),
            is_decision=False,
            raw_mid=float(raw_mid),
            event_type="trade",
        )

    def _handle_ob_event(
        self,
        ts_ms: int,
        payload: Any,
        any_event_dt_ms: float,
    ) -> FeatureEventResult:
        prev_bid_l1 = self.prev_bsz
        prev_ask_l1 = self.prev_asz
        prev_bid_l2 = self.prev_bsz2
        prev_ask_l2 = self.prev_asz2

        self._prune_replen_windows(ts_ms)

        for window in self._trade_window_deques:
            self._prune_trade_window(ts_ms, window)

        tp_code, bids, asks = payload
        self._update_book_from_ob(tp_code, bids, asks)
        self._validate_book_health(ts_ms, tp_code)

        for window, deq in self._quote_window_deques.items():
            self._append_ts_with_guard(deq, ts_ms, window, is_ob_event=True)

        for window, deq in self._ob_update_deques.items():
            self._append_ts_with_guard(deq, ts_ms, window, is_ob_event=True)
        bid1, ask1, bsz1, asz1 = self._book_best()
        mid = 0.5 * (bid1 + ask1) if (bid1 > 0 and ask1 > 0) else 0.0

        if (bsz1 + asz1) > 0:
            micro = (ask1 * bsz1 + bid1 * asz1) / (bsz1 + asz1)
        else:
            micro = mid

        spread = max(0.0, ask1 - bid1)
        spread_bps = 1e4 * spread / mid if mid > 0.0 else 0.0

        spread_delta_bps: Dict[int, float] = {}
        for window in self.ob_horizon_compare_windows_ms:
            target_ts_ms = ts_ms - window
            ob_snap = self.get_ob_snapshot_asof(target_ts_ms)
            spread_t_minus_h = ob_snap.spread if ob_snap is not None else spread
            spread_delta_bps[window] = 1e4 * (spread - spread_t_minus_h) / max(mid, 1e-12)

        ask2 = self.ask_lvls[1][0] if len(self.ask_lvls) > 1 else ask1
        bid2 = self.bid_lvls[1][0] if len(self.bid_lvls) > 1 else bid1
        gap_a = max(0.0, ask2 - ask1)
        gap_b = max(0.0, bid1 - bid2)
        gap_a_bps = 1e4 * gap_a / max(mid, 1e-12)
        gap_b_bps = 1e4 * gap_b / max(mid, 1e-12)

        bsz2 = self.bid_lvls[1][1] if len(self.bid_lvls) > 1 else 0.0
        asz2 = self.ask_lvls[1][1] if len(self.ask_lvls) > 1 else 0.0
        replen_deltas = {
            ("bid", 1, "add"): max(bsz1 - prev_bid_l1, 0.0),
            ("bid", 1, "rem"): max(prev_bid_l1 - bsz1, 0.0),
            ("ask", 1, "add"): max(asz1 - prev_ask_l1, 0.0),
            ("ask", 1, "rem"): max(prev_ask_l1 - asz1, 0.0),
            ("bid", 2, "add"): max(bsz2 - prev_bid_l2, 0.0),
            ("bid", 2, "rem"): max(prev_bid_l2 - bsz2, 0.0),
            ("ask", 2, "add"): max(asz2 - prev_ask_l2, 0.0),
            ("ask", 2, "rem"): max(prev_ask_l2 - asz2, 0.0),
        }
        self._record_replenishment(ts_ms, replen_deltas)

        cum_bid_by_level = {lvl: self._cum_depth(self.bid_lvls, lvl) for lvl in BOOK_SIGNAL_LEVELS}
        cum_ask_by_level = {lvl: self._cum_depth(self.ask_lvls, lvl) for lvl in BOOK_SIGNAL_LEVELS}
        cum_bid3 = cum_bid_by_level[3]
        cum_ask3 = cum_ask_by_level[3]
        cum_bid5 = cum_bid_by_level[5]
        cum_ask5 = cum_ask_by_level[5]

        self._append_ob_snapshot(ts_ms, bid1, ask1, bsz1, asz1, spread, cum_bid3, cum_ask3, cum_bid5, cum_ask5)

        obi_by_level = {
            lvl: (cum_bid_by_level[lvl] - cum_ask_by_level[lvl]) / max(cum_bid_by_level[lvl] + cum_ask_by_level[lvl], 1e-12)
            for lvl in BOOK_SIGNAL_LEVELS
        }

        ofi_by_level: Dict[int, float] = {}
        ofi_by_level[1] = (bsz1 - prev_bid_l1) - (asz1 - prev_ask_l1)
        for lvl in BOOK_SIGNAL_LEVELS:
            if lvl == 1:
                continue
            ofi_by_level[lvl] = (
                (cum_bid_by_level[lvl] - self.prev_cum_bid_by_level[lvl])
                - (cum_ask_by_level[lvl] - self.prev_cum_ask_by_level[lvl])
            )
        ofi_l1 = ofi_by_level[1]
        ofi_l3 = ofi_by_level[3]
        ofi_l5 = ofi_by_level[5]
        self.prev_bsz, self.prev_asz = bsz1, asz1
        self.prev_bsz2, self.prev_asz2 = bsz2, asz2
        for lvl in BOOK_SIGNAL_LEVELS:
            self.prev_cum_bid_by_level[lvl] = cum_bid_by_level[lvl]
            self.prev_cum_ask_by_level[lvl] = cum_ask_by_level[lvl]


        micro_premia = (micro - mid) / max(spread, 1e-12)
        micro_minus_mid_bps = 1e4 * (micro / mid - 1.0) if mid > 0.0 and micro > 0.0 else 0.0
        micro_minus_mid_over_spread = (micro - mid) / max(spread, 1e-12)
        if self._last_ob_feature_ts is None:
            ob_dt_ms = 1.0
        else:
            ob_dt_ms = max(1.0, float(ts_ms - int(self._last_ob_feature_ts)))


        for ms in FAST_WINDOWS_MS:
            self.ofi_pressure_by_window[ms] = self._ewma_update(self.ofi_pressure_by_window[ms], ofi_l1, ob_dt_ms, ms)
        ofi_pressure_by_ms = {ms: self.ofi_pressure_by_window[ms] for ms in FAST_WINDOWS_MS}
        trade_stats_by_ms = {ms: self._compute_trade_window_stats(ms, ts_ms, mid, micro) for ms in self.trade_windows}
        self.last_ob_ofi_l5 = float(ofi_l5)
        self.last_ob_trade_imbalance_1000ms = float(trade_stats_by_ms[1_000]["trade_imbalance_notional"])
        cvd_stats_by_ms: Dict[int, Dict[str, float]] = {}
        for ms in FLOW_WINDOWS_MS:
            cvd_state = self.cvd_window_states[ms]
            cvd_state.prune(ts_ms)
            cvd_change = cvd_state.change_usd(ts_ms, self.cvd_notional)
            cvd_slope = cvd_state.slope_usd_per_sec()
            cvd_minus_ema = float(self.cvd_notional - self._cvd_ema[ms])
            cvd_stats_by_ms[ms] = {
                "cvd_change_usd": float(cvd_change),
                "cvd_slope_usd_per_sec": float(cvd_slope),
                "cvd_minus_ema_usd": float(cvd_minus_ema),
            }
            for k, v in cvd_stats_by_ms[ms].items():
                if not math.isfinite(float(v)):
                    raise ValueError(f"Non-finite CVD stat {k}={v!r} at ts_ms={ts_ms} window={ms}")
        large_stats_by_ms = {
            ms: self._large_trade_stats_from_state(ms, ts_ms)
            for ms in self.large_trade_windows
        }
        consecutive_buy_trade_count = float(self.consecutive_buy_trade_count)
        consecutive_sell_trade_count = float(self.consecutive_sell_trade_count)
        if self.prev_bid1_price is None or bid1 != self.prev_bid1_price:
            self.last_bid_price_change_ts = ts_ms
        if self.prev_ask1_price is None or ask1 != self.prev_ask1_price:
            self.last_ask_price_change_ts = ts_ms
        if self.prev_mid_price_for_age is None or mid != self.prev_mid_price_for_age:
            self.last_mid_change_ts = ts_ms
        spread_changed = (self.last_spread is not None and spread != self.last_spread)
        for ms in self._spread_change_deques:
            if spread_changed:
                self._append_ts_with_guard(self._spread_change_deques[ms], ts_ms, ms, is_ob_event=True)
            else:
                self._prune_ts_deque(self._spread_change_deques[ms], ts_ms, ms)
        if self.last_spread is None or spread_changed:
            self.last_spread = spread
            self.last_spread_ts = ts_ms

        bid_level_changed = (self.last_bid1 is None or bid1 != self.last_bid1 or bsz1 != prev_bid_l1)
        ask_level_changed = (self.last_ask1 is None or ask1 != self.last_ask1 or asz1 != prev_ask_l1)
        bid_price_changed = (self.prev_bid1_price is not None and bid1 != self.prev_bid1_price)
        ask_price_changed = (self.prev_ask1_price is not None and ask1 != self.prev_ask1_price)
        for ms, dq in self._bid_price_change_deques.items():
            self._append_ts_with_guard(dq, ts_ms, ms, is_ob_event=True) if bid_price_changed else self._prune_ts_deque(dq, ts_ms, ms)
        for ms, dq in self._ask_price_change_deques.items():
            self._append_ts_with_guard(dq, ts_ms, ms, is_ob_event=True) if ask_price_changed else self._prune_ts_deque(dq, ts_ms, ms)
        if bid_level_changed:
            self.last_bid1_update_ts = ts_ms
        if ask_level_changed:
            self.last_ask1_update_ts = ts_ms
        self.last_bid1, self.last_ask1 = bid1, ask1

        bid_l1_depletion_event = max(prev_bid_l1 - bsz1, 0.0) if prev_bid_l1 > 0.0 else 0.0
        ask_l1_depletion_event = max(prev_ask_l1 - asz1, 0.0) if prev_ask_l1 > 0.0 else 0.0
        for ms in self.bestlvl_windows:
            bid_deq = self._bid_l1_depletion_deques[ms]
            ask_deq = self._ask_l1_depletion_deques[ms]
            bid_deq.append((ts_ms, bid_l1_depletion_event))
            ask_deq.append((ts_ms, ask_l1_depletion_event))
            self._bid_l1_depletion_sums[ms] += bid_l1_depletion_event
            self._ask_l1_depletion_sums[ms] += ask_l1_depletion_event
            while bid_deq and (ts_ms - bid_deq[0][0] > ms):
                _, old_bid = bid_deq.popleft()
                self._bid_l1_depletion_sums[ms] -= old_bid
            while ask_deq and (ts_ms - ask_deq[0][0] > ms):
                _, old_ask = ask_deq.popleft()
                self._ask_l1_depletion_sums[ms] -= old_ask
        l1_depletion = {
            ms: (self._bid_l1_depletion_sums[ms], self._ask_l1_depletion_sums[ms])
            for ms in self.bestlvl_windows
        }

        dt_since_trade = float(ts_ms - self.last_trade_ts) if self.last_trade_ts is not None else 0.0
        time_since_mid_change_ms = float(ts_ms - self.last_mid_change_ts) if self.last_mid_change_ts is not None else 0.0

        self._add_return(ts_ms, mid, ob_dt_ms)
        return_var = {ms: stats.mean_var()[1] for ms, stats in self.return_histories.items()}
        return_std_bps = {ms: math.sqrt(var) for ms, var in return_var.items()}

        regime_vol_ewma = {ms: math.sqrt(max(self.rv_ewma[ms], 1e-18)) for ms in self.regime_windows_ms}
        regime_realized = {ms: self.realized_vol[ms] for ms in self.regime_windows_ms}
        regime_volume = {ms: self.volume_ewma[ms] for ms in self.regime_windows_ms}
        for ms in self.regime_windows_ms:
            self.flow_regime[ms] = trade_stats_by_ms[ms]["trade_imbalance_notional"]
        regime_flow_snapshot = {ms: self.flow_regime[ms] for ms in self.regime_windows_ms}


        replen_rates = self._replenishment_rates()
        price_features_by_window: Dict[int, Tuple[float, ...]] = {}
        for w in PRICE_WINDOWS_MS:
            past_mid = self._mid_asof_history.asof(ts_ms - w)
            past_micro = self._micro_asof_history.asof(ts_ms - w)
            mid_ret_bps = self._bps_return(mid, past_mid) if past_mid is not None else 0.0
            micro_ret_bps = self._bps_return(micro, past_micro) if past_micro is not None else 0.0
            state = self._price_window_mid_states[w]
            state.prune(ts_ms)  # critical: prune to current row timestamp before reading
            mid_slope_bps_per_sec = state.slope_bps_per_sec()
            mid_range_bps = state.range_bps()
            price_features_by_window[w] = (
                mid_ret_bps, micro_ret_bps, mid_slope_bps_per_sec, mid_range_bps,
            )
        bid_depth_5bps = self._depth_within_bps(self.bid_lvls, mid, 5.0, is_bid=True)
        ask_depth_5bps = self._depth_within_bps(self.ask_lvls, mid, 5.0, is_bid=False)
        depth_5bps_total = bid_depth_5bps + ask_depth_5bps
        depth_5bps_imbalance = self._safe_div(
            bid_depth_5bps - ask_depth_5bps,
            depth_5bps_total,
            0.0,
        )
        self._append_metric_history(self._spread_bps_history, ts_ms, spread_bps, self._regime_metric_keep_ms)
        self._append_metric_history(self._bid_depth_5bps_history, ts_ms, bid_depth_5bps, self._regime_metric_keep_ms)
        self._append_metric_history(self._ask_depth_5bps_history, ts_ms, ask_depth_5bps, self._regime_metric_keep_ms)
        self._append_metric_history(self._depth_5bps_total_history, ts_ms, depth_5bps_total, self._regime_metric_keep_ms)
        self._append_metric_history(self._depth_5bps_imbalance_history, ts_ms, depth_5bps_imbalance, self._regime_metric_keep_ms)
        for ms in SPREAD_DEPTH_REGIME_WINDOWS_MS:
            self._spread_bps_regime_states[ms].update(ts_ms, spread_bps)
            self._bid_depth_5bps_regime_states[ms].update(ts_ms, bid_depth_5bps)
            self._ask_depth_5bps_regime_states[ms].update(ts_ms, ask_depth_5bps)
            self._depth_5bps_total_regime_states[ms].update(ts_ms, depth_5bps_total)
            self._depth_5bps_imbalance_regime_states[ms].update(ts_ms, depth_5bps_imbalance)
        rolling_ofi_sums: Dict[Tuple[int, int], float] = {}
        rolling_obi_means: Dict[Tuple[int, int], float] = {}
        for level in ROLLING_OFI_LEVELS:
            value = ofi_by_level[level]
            self._append_metric_history(self.ofi_level_histories[level], ts_ms, value, keep_ms=10_000)
            for window in ROLLING_OFI_WINDOWS_MS:
                self._rolling_ofi_states[(level, window)].update(ts_ms, value)
        for level in ROLLING_OBI_LEVELS:
            value = obi_by_level[level]
            self._append_metric_history(self.obi_level_histories[level], ts_ms, value, keep_ms=10_000)
            for window in ROLLING_OBI_WINDOWS_MS:
                self._rolling_obi_states[(level, window)].update(ts_ms, value)
        for level in ROLLING_OFI_LEVELS:
            for window in ROLLING_OFI_WINDOWS_MS:
                rolling_ofi_sums[(level, window)] = self._rolling_ofi_states[(level, window)].sum_value()
        for level in ROLLING_OBI_LEVELS:
            for window in ROLLING_OBI_WINDOWS_MS:
                rolling_obi_means[(level, window)] = self._rolling_obi_states[(level, window)].mean()
        deep_micro_features: Dict[str, float] = {}
        deep_micro_minus_mid_bps: Dict[int, float] = {}
        micro_price_by_level: Dict[int, float] = {}
        for level in DEEP_MICRO_LEVELS:
            bid_px_n, bid_qty_n = self._weighted_side_price(self.bid_lvls, level)
            ask_px_n, ask_qty_n = self._weighted_side_price(self.ask_lvls, level)
            den = bid_qty_n + ask_qty_n
            if den > 1e-12:
                micro_l = (ask_px_n * bid_qty_n + bid_px_n * ask_qty_n) / den
                # VAMP convention: same-side weighted average
                vamp_l = (bid_px_n * bid_qty_n + ask_px_n * ask_qty_n) / den
            else:
                micro_l = 0.0
                vamp_l = 0.0
            micro_price_by_level[level] = float(micro_l)
            deep_micro_minus_mid_bps[level] = self._bps(micro_l, mid, 0.0)
            deep_micro_features[f"micro_l{level}_minus_mid_bps"] = deep_micro_minus_mid_bps[level]
            deep_micro_features[f"vamp_l{level}_minus_mid_bps"] = self._bps(vamp_l, mid, 0.0)
        self._append_metric_history(self.deep_micro_histories[5], ts_ms, deep_micro_minus_mid_bps.get(5, 0.0), keep_ms=10_000)
        self._append_metric_history(self.deep_micro_histories[10], ts_ms, deep_micro_minus_mid_bps.get(10, 0.0), keep_ms=10_000)
        for window in (200, 1_000):
            points = self._metric_values(self.deep_micro_histories[5], ts_ms, window)
            xs = [(t - ts_ms) / 1000.0 for t, _ in points]
            ys = [v for _, v in points]
            deep_micro_features[f"micro_l5_slope_{window}ms"] = self._lin_slope(xs, ys) if len(ys) >= 2 else 0.0
        deep_micro_features["micro_l1_minus_micro_l10_bps"] = self._bps(micro, micro_price_by_level.get(10, 0.0), 0.0)
        bid_l1_notional_usd = bid1 * bsz1
        ask_l1_notional_usd = ask1 * asz1
        bid_depth_notional_5bps = self._notional_depth_within_bps(
            self.bid_lvls,
            mid,
            5.0,
            is_bid=True,
        )
        ask_depth_notional_5bps = self._notional_depth_within_bps(
            self.ask_lvls,
            mid,
            5.0,
            is_bid=False,
        )
        total_depth_notional_5bps = bid_depth_notional_5bps + ask_depth_notional_5bps

        feat_list: List[float] = []
        feat_list.extend(self._calendar_context_features(ts_ms))
        for w in PRICE_WINDOWS_MS:
            feat_list.extend(price_features_by_window[w])
        feat_list.extend([
            spread_bps, gap_a_bps, gap_b_bps, bsz1, asz1, micro_premia,
            micro_minus_mid_bps, micro_minus_mid_over_spread,
            dt_since_trade,
            time_since_mid_change_ms,
        ])
        feat_list.extend([
            bid_l1_notional_usd,
            ask_l1_notional_usd,
            bid_depth_notional_5bps,
            ask_depth_notional_5bps,
            total_depth_notional_5bps,
        ])
        for lvl in BOOK_SIGNAL_LEVELS:
            feat_list.append(obi_by_level[lvl])
        for lvl in BOOK_SIGNAL_LEVELS:
            feat_list.append(ofi_by_level[lvl])
        for lvl in NORMALIZED_OFI_LEVELS:
            ofi_val = ofi_by_level[lvl]
            level_depth = cum_bid_by_level[lvl] + cum_ask_by_level[lvl]
            feat_list.append(self._safe_div(ofi_val, level_depth, 0.0))
            feat_list.append(self._safe_div(ofi_val, depth_5bps_total, 0.0))
        for level in ROLLING_OFI_LEVELS:
            depth_l = cum_bid_by_level[level] + cum_ask_by_level[level]
            for window in ROLLING_OFI_WINDOWS_MS:
                ofi_sum = rolling_ofi_sums[(level, window)]
                feat_list.append(self._safe_div(ofi_sum, depth_l, 0.0))
        for level in ROLLING_OFI_LEVELS:
            for fast_ms, slow_ms in OFI_ACCEL_PAIRS_MS:
                feat_list.append(rolling_ofi_sums[(level, fast_ms)] - rolling_ofi_sums[(level, slow_ms)])
        for level in ROLLING_OBI_LEVELS:
            for window in ROLLING_OBI_WINDOWS_MS:
                feat_list.append(rolling_obi_means[(level, window)])
        for level in DEEP_MICRO_LEVELS:
            feat_list.extend([
                deep_micro_features[f"micro_l{level}_minus_mid_bps"],
                deep_micro_features[f"vamp_l{level}_minus_mid_bps"],
            ])
        feat_list.extend([
            deep_micro_features["micro_l5_slope_200ms"],
            deep_micro_features["micro_l5_slope_1000ms"],
            deep_micro_features["micro_l1_minus_micro_l10_bps"],
        ])
        for band in BPS_DEPTH_BANDS:
            if float(band) == 5.0:
                bid_size = bid_depth_5bps
                ask_size = ask_depth_5bps
            else:
                bid_size = self._depth_within_bps(self.bid_lvls, mid, band, is_bid=True)
                ask_size = self._depth_within_bps(self.ask_lvls, mid, band, is_bid=False)

            feat_list.extend([
                bid_size,
                ask_size,
                self._safe_div(bid_size - ask_size, bid_size + ask_size, 0.0),
            ])
        for ms in FAST_WINDOWS_MS:
            window_seconds = max(ms / 1000.0, 1e-9)
            bid_price_change_count = float(len(self._bid_price_change_deques[ms]))
            ask_price_change_count = float(len(self._ask_price_change_deques[ms]))
            bid_l1_depletion = l1_depletion[ms][0]
            ask_l1_depletion = l1_depletion[ms][1]
            feat_list.extend([
                self._safe_div(spread_delta_bps[ms], max(abs(spread_bps), 1e-12), 0.0),
                float(len(self._spread_change_deques[ms])),
                bid_price_change_count / window_seconds,
                ask_price_change_count / window_seconds,
                bid_l1_depletion,
                ask_l1_depletion,
                self._safe_div(bid_l1_depletion, max(bsz1, 1e-9), 0.0),
                self._safe_div(ask_l1_depletion, max(asz1, 1e-9), 0.0),
            ])
        for ms in FAST_WINDOWS_MS:
            window_seconds = max(ms / 1000.0, 1e-12)
            ob_count = float(len(self._ob_update_deques[ms]))
            feat_list.extend([
                ob_count / window_seconds,
            ])
        for ms in FAST_WINDOWS_MS:
            rates = replen_rates[ms]
            bid_add_rate = rates[("bid", 1, "add")] * 1000.0
            bid_rem_rate = rates[("bid", 1, "rem")] * 1000.0
            ask_add_rate = rates[("ask", 1, "add")] * 1000.0
            ask_rem_rate = rates[("ask", 1, "rem")] * 1000.0
            feat_list.extend([
                self._safe_div(bid_add_rate, max(bsz1, 1e-9), 0.0),
                self._safe_div(bid_rem_rate, max(bsz1, 1e-9), 0.0),
                self._safe_div(ask_add_rate, max(asz1, 1e-9), 0.0),
                self._safe_div(ask_rem_rate, max(asz1, 1e-9), 0.0),
            ])

        for ms in FLOW_WINDOWS_MS:
            s = trade_stats_by_ms[ms]
            feat_list.extend([
                s["signed_notional_flow_usd"],
                s["signed_trade_count_imbalance"],
                s["trade_imbalance_notional"],
                s["trade_toxicity_notional"],
                s["zero_tick_fraction"],
                s["tick_sign_imbalance"],
                s["trade_count_per_second"],
                s["vwap_vs_mid_bps"],
                s["signed_trade_premium_bps_volume_weighted"],
            ])
        feat_list.extend([
            float(self.last_trade_side_sign),
            float(self.last_tick_sign),
            float(self.last_is_zero_tick),
            float(self.last_is_rpi),
            float(self.last_trade_notional_usd),
            float(ts_ms - self.last_buy_trade_ts) if self.last_buy_trade_ts is not None else 0.0,
            float(ts_ms - self.last_sell_trade_ts) if self.last_sell_trade_ts is not None else 0.0,
        ])
        for ms in FLOW_WINDOWS_MS:
            c = cvd_stats_by_ms[ms]
            feat_list.extend([
                c["cvd_change_usd"],
                c["cvd_slope_usd_per_sec"],
                c["cvd_minus_ema_usd"],
            ])
        feat_list.extend([
            consecutive_buy_trade_count,
            consecutive_sell_trade_count,
        ])
        for ms in FLOW_WINDOWS_MS:
            large_stats = large_stats_by_ms[ms]
            feat_list.extend([
                large_stats[f"max_signed_trade_notional_usd_{ms}ms"],
                large_stats[f"top5_trade_notional_sum_usd_{ms}ms"],
            ])
        for ms in FLOW_WINDOWS_MS:
            s = trade_stats_by_ms[ms]
            mid_ret_bps = price_features_by_window[ms][0] if ms in price_features_by_window else (
                self._bps_return(mid, self._series_asof(ts_ms - ms, "mid")) if self._series_asof(ts_ms - ms, "mid") is not None else 0.0
            )
            buy_notional_scaled = s["buy_notional_usd"] / 100_000.0
            sell_notional_scaled = s["sell_notional_usd"] / 100_000.0
            buy_flow_without_price_up = buy_notional_scaled * math.exp(max(-max(mid_ret_bps, 0.0), -50.0))
            sell_flow_without_price_down = sell_notional_scaled * math.exp(max(-max(-mid_ret_bps, 0.0), -50.0))
            absorption_ask = buy_notional_scaled / max(0.25, max(mid_ret_bps, 0.0))
            absorption_bid = sell_notional_scaled / max(0.25, max(-mid_ret_bps, 0.0))
            absorption_values = [
                buy_flow_without_price_up,
                sell_flow_without_price_down,
                absorption_bid,
                absorption_ask,
            ]
            for i, value in enumerate(absorption_values):
                if not math.isfinite(float(value)):
                    raise ValueError(f"Non-finite absorption stat idx={i} value={value!r} at ts_ms={ts_ms} window={ms}")
            feat_list.extend(absorption_values)

        for ms in FLOW_WINDOWS_MS:
            feat_list.append(return_std_bps[ms])
        for ms in REGIME_WINDOWS_MS:
            dist = self._regime_distribution(ms)
            feat_list.extend([
                regime_volume[ms],
                regime_realized[ms],
                regime_flow_snapshot[ms],
                dist["down_up_vol_ratio"],
                dist["max_abs_return_bps"],
            ])
        for ms in SPREAD_DEPTH_REGIME_WINDOWS_MS:
            spread_win = self._spread_bps_regime_states[ms]
            depth_state = self._depth_5bps_total_regime_states[ms]
            imb_win = self._depth_5bps_imbalance_regime_states[ms]
            spread_mean, spread_std = spread_win.mean_std()
            spread_z = 0.0 if spread_std <= 1e-9 else (spread_bps - spread_mean) / max(spread_std, 1e-9)
            spread_slope = spread_win.slope()
            spread_above = spread_win.frac_above(1.0)

            depth_mean, depth_std = depth_state.mean_std()
            depth_z = 0.0 if depth_std <= 1e-9 else (depth_5bps_total - depth_mean) / max(depth_std, 1e-9)
            imb_mean = imb_win.mean()
            imb_slope = imb_win.slope()

            feat_list.extend([
                spread_z,
                spread_slope,
                spread_above,
                depth_z,
                imb_mean,
                imb_slope,
            ])
        depth_5bps_total_base = depth_5bps_total

        for ms in FAST_WINDOWS_MS:
            ofi_pressure = ofi_pressure_by_ms[ms]
            feat_list.extend([
                ofi_pressure,
                self._safe_div(ofi_pressure, max(depth_5bps_total_base, 1e-9), 0.0),
                self._safe_div(ofi_pressure, max(self._realized_vol_for_pressure(ms), 1e-9), 0.0),
            ])

        names = self.feature_names()
        if len(feat_list) != len(names):
            raise ValueError(
                f"Feature vector/name length mismatch: len(feat_list)={len(feat_list)} "
                f"len(feature_names)={len(names)}"
            )
        feat = np.asarray(feat_list, dtype=np.float64)
        if len(names) != 245:
            raise ValueError(f"Expected pruned raw core dim 245, got {len(names)}")
        if self.strict_feature_validation:
            if not np.all(np.isfinite(feat)):
                bad_idx = np.flatnonzero(~np.isfinite(feat))
                details = [
                    f"{int(i)}:{names[int(i)]}={feat[int(i)]!r}"
                    for i in bad_idx[:20]
                ]
                raise FloatingPointError("Non-finite feature values: " + ", ".join(details))
        feat_out = self._transform_features(feat, ob_dt_ms)
        if feat_out.shape != (245,):
            raise ValueError(f"Expected transformed feature shape (245,), got {feat_out.shape}")
        self._append_price_history(ts_ms, mid, micro)
        self.prev_bid1_price = bid1
        self.prev_ask1_price = ask1
        self.prev_mid_price_for_age = mid
        self.prev_spread_for_age = spread
        self.last_ts = ts_ms
        self._last_ob_feature_ts = int(ts_ms)
        self._last_any_event_ts = int(ts_ms)

        self.ob_feature_build_count += 1

        return FeatureEventResult(
            ts_ms=int(ts_ms),
            features=feat_out,
            dt_ms=float(any_event_dt_ms),
            is_decision=True,
            raw_mid=float(mid),
            event_type="ob",
        )

    def _levels_to_book_dict(
        self,
        levels: Sequence[Tuple[float, float]],
    ) -> Dict[float, float]:
        out: Dict[float, float] = {}
        for price_raw, size_raw in levels:
            try:
                price = float(price_raw)
                size = float(size_raw)
            except (TypeError, ValueError):
                continue
            if not math.isfinite(price) or not math.isfinite(size):
                continue
            if price <= 0.0 or size <= 0.0:
                continue
            out[price] = size
        return out

    def _update_book_from_ob(
        self,
        tp_code: int,
        bids: Sequence[Tuple[float, float]],
        asks: Sequence[Tuple[float, float]],
    ) -> None:
        if int(tp_code) == 1:
            # Snapshot replaces the full in-memory book. Do not truncate to feature_depth here.
            # Truncation happens only in _sorted_ladders().
            self.bids = self._levels_to_book_dict(bids)
            self.asks = self._levels_to_book_dict(asks)
            self._sorted_ladders()
            return

        rebuild_bid = self._apply_side_updates(self.bids, self.bid_lvls, bids, is_bid=True)
        rebuild_ask = self._apply_side_updates(self.asks, self.ask_lvls, asks, is_bid=False)
        self._book_dirty = bool(rebuild_bid or rebuild_ask)

    def _interpret_tick_direction(self, tick_dir: Any) -> Tuple[int, int]:
        tick_sign = 0
        is_zero_tick = 0

        if isinstance(tick_dir, (int, float)):
            if tick_dir > 0:
                tick_sign = 1
            elif tick_dir < 0:
                tick_sign = -1
            else:
                tick_sign = 0
                is_zero_tick = 1
        elif isinstance(tick_dir, str):
            norm = tick_dir.strip().lower()
            cleaned = norm.replace("-", "").replace("_", "").replace(" ", "")
            if 'plus' in cleaned or cleaned in {"plustick", "uptick", "up", "buy", "bid"}:
                tick_sign = 1
            elif 'minus' in cleaned or cleaned in {"minustick", "downtick", "down", "sell", "ask"}:
                tick_sign = -1
            elif cleaned in {"zerotick", "flat", "unchanged", "0"}:
                tick_sign = 0
            if 'zero' in cleaned or cleaned in {"zerotick", "flat", "unchanged", "0"}:
                is_zero_tick = 1

        return int(tick_sign), int(is_zero_tick)

    def _update_trade_windows(self, ts_ms: int, trade_evt: Any, dt_ms: float):
        if isinstance(trade_evt, tuple):
            price, size, side_code, tick_dir_code, is_rpi = trade_evt
            side = 'buy' if int(side_code) > 0 else 'sell' if int(side_code) < 0 else 'unknown'
            side_sign = 1.0 if int(side_code) > 0 else -1.0 if int(side_code) < 0 else 0.0
            price = float(price)
            size = float(size)
            tick_sign = float(int(tick_dir_code))
            is_zero_tick = 1.0 if int(tick_dir_code) == 0 else 0.0
            is_rpi = int(is_rpi)
        else:
            side = str(trade_evt['side']).lower()  # 'buy'|'sell'
            side_sign = 1.0 if side == "buy" else -1.0 if side == "sell" else 0.0
            price = float(trade_evt['price'])
            size = float(trade_evt['size'])

            tick_dir = trade_evt.get("tickDirection")
            tick_sign = float(int(self.last_tick_sign))
            is_zero_tick = float(int(self.last_is_zero_tick))
            is_rpi = int(self.last_is_rpi)

            rpi_raw = trade_evt.get("RPI")
            if rpi_raw is None:
                rpi_raw = trade_evt.get("rpi")

            if rpi_raw is not None:
                if isinstance(rpi_raw, str):
                    rpi_norm = rpi_raw.strip().lower()
                    if rpi_norm in {"1", "true", "t", "yes", "y"}:
                        is_rpi = 1
                    elif rpi_norm in {"0", "false", "f", "no", "n", ""}:
                        is_rpi = 0
                    else:
                        try:
                            is_rpi = 1 if float(rpi_norm) != 0.0 else 0
                        except ValueError:
                            pass
                else:
                    try:
                        is_rpi = 1 if float(rpi_raw) != 0.0 else 0
                    except (TypeError, ValueError):
                        pass

            if tick_dir is not None:
                td_sign, td_zero = self._interpret_tick_direction(tick_dir)
                tick_sign, is_zero_tick = float(td_sign), float(td_zero)

        if self.last_trade_price is not None:
            if price > self.last_trade_price:
                tick_sign, is_zero_tick = 1.0, 0.0
            elif price < self.last_trade_price:
                tick_sign, is_zero_tick = -1.0, 0.0
            else:
                if tick_sign == 0 and is_zero_tick == 0:
                    tick_sign = float(self.last_tick_sign if self.last_tick_sign != 0 else 0)
                if is_zero_tick == 0:
                    is_zero_tick = 1.0
        else:
            if tick_sign == 0 and is_zero_tick == 0:
                tick_sign, is_zero_tick = 0.0, 0.0

        self.last_tick_sign = int(tick_sign)
        self.last_is_zero_tick = int(is_zero_tick)
        self.last_trade_price = float(price)
        self.last_is_rpi = int(is_rpi)

        notional_usd = float(price) * float(size)
        self.last_trade_side_sign = float(side_sign)
        self.last_trade_notional_usd = float(notional_usd)
        if side_sign > 0:
            self.last_buy_trade_ts = int(ts_ms)
            self.consecutive_buy_trade_count += 1
            self.consecutive_sell_trade_count = 0
            trade_sign = 1
        elif side_sign < 0:
            self.last_sell_trade_ts = int(ts_ms)
            self.consecutive_sell_trade_count += 1
            self.consecutive_buy_trade_count = 0
            trade_sign = -1
        else:
            self.consecutive_buy_trade_count = 0
            self.consecutive_sell_trade_count = 0
            trade_sign = 0

        entry = (ts_ms, price, size, notional_usd, side, side_sign, tick_sign, is_zero_tick)
        for window, deq in self._trade_window_deques.items():
            deq.append(entry)
            self._update_trade_window_state_with_insert(window, entry)

        if side_sign != 0.0:
            self.cvd_notional += side_sign * notional_usd
        for cvd_state in self.cvd_window_states.values():
            cvd_state.add(int(ts_ms), float(self.cvd_notional))
        if self.last_cvd_update_ts is None:
            for ms in FLOW_WINDOWS_MS:
                self._cvd_ema[ms] = float(self.cvd_notional)
                self._cvd_ema_initialized[ms] = True
        else:
            cvd_dt_ms = max(1, int(ts_ms - self.last_cvd_update_ts))
            for ms in FLOW_WINDOWS_MS:
                if not self._cvd_ema_initialized[ms]:
                    self._cvd_ema[ms] = float(self.cvd_notional)
                    self._cvd_ema_initialized[ms] = True
                    continue
                alpha = 1.0 - math.exp(-math.log(2.0) * max(cvd_dt_ms, 1) / float(ms))
                self._cvd_ema[ms] = (1.0 - alpha) * self._cvd_ema[ms] + alpha * float(self.cvd_notional)
        self.last_cvd_update_ts = int(ts_ms)



        dt_trade_ms = (
            max(1.0, float(ts_ms - self.last_trade_ts))
            if self.last_trade_ts is not None
            else max(1.0, float(dt_ms))
        )

        # Update volume-regime (vol/sec) EWMAs using trade-arrival timing
        vol_rate = size / (dt_trade_ms / 1000.0)  # base per second
        for hl in self.regime_windows_ms:
            self.volume_ewma[hl] = self._ewma_update(self.volume_ewma[hl], vol_rate, dt_trade_ms, hl)

        self.last_trade_ts = ts_ms

    def _add_return(self, ts_ms: int, mid: float, ob_dt_ms: float):
        if mid <= 0.0:
            return 0.0
    
        if self.last_mid_for_ret is None:
            self.last_mid_for_ret = mid
            return 0.0
    
        r = (1e4 * math.log(mid / self.last_mid_for_ret)) if self.last_mid_for_ret > 0 else 0.0
        self.last_mid_for_ret = mid

        for ms, stats in self.return_histories.items():
            stats.add(ts_ms, r)

        r2 = r * r
        for hl in self.regime_windows_ms:
            self.rv_ewma[hl] = self._ewma_update(self.rv_ewma[hl], r2, ob_dt_ms, hl)

        for ms, state in self.regime_return_states.items():
            self._regime_return_add(state, int(ts_ms), float(r))
            self._regime_return_prune(state, int(ts_ms))
            self.realized_vol[ms] = self._regime_realized_vol(state)
        return r

    def _regime_return_add(self, state: RollingReturnDistributionState, ts_ms: int, r: float) -> None:
        v = float(r)
        av = abs(v)
        state.seq += 1
        seq = state.seq
        if state.deq:
            state.bipower += abs(state.deq[-1][1]) * av
        state.deq.append((int(ts_ms), v, av, seq))
        state.n += 1
        state.sum1 += v
        state.sum2 += v * v
        state.sum3 += v * v * v
        state.sum4 += v * v * v * v
        if v > 0.0:
            state.up_sumsq += v * v
        elif v < 0.0:
            state.down_sumsq += v * v
        while state.max_abs_q and state.max_abs_q[-1][2] <= av:
            state.max_abs_q.pop()
        state.max_abs_q.append((seq, int(ts_ms), av))

    def _regime_return_prune(self, state: RollingReturnDistributionState, now_ms: int) -> None:
        cutoff = int(now_ms) - int(state.window_ms)
        while state.deq and (int(now_ms) - state.deq[0][0] > state.window_ms):
            old_ts, old_r, old_abs, old_seq = state.deq.popleft()
            state.n -= 1
            state.sum1 -= old_r
            state.sum2 -= old_r * old_r
            state.sum3 -= old_r * old_r * old_r
            state.sum4 -= old_r * old_r * old_r * old_r
            if old_r > 0.0:
                state.up_sumsq -= old_r * old_r
            elif old_r < 0.0:
                state.down_sumsq -= old_r * old_r
            if state.deq:
                state.bipower -= old_abs * abs(state.deq[0][1])
            if state.max_abs_q and state.max_abs_q[0][0] == old_seq:
                state.max_abs_q.popleft()
        while state.max_abs_q and state.max_abs_q[0][1] < cutoff:
            state.max_abs_q.popleft()
        for attr in ("sum1", "sum2", "sum3", "sum4", "up_sumsq", "down_sumsq", "bipower"):
            val = getattr(state, attr)
            if abs(val) < 1e-12:
                setattr(state, attr, 0.0)

    def _regime_realized_vol(self, state: RollingReturnDistributionState) -> float:
        return math.sqrt(max(state.sum2, 0.0))

    def _regime_distribution(self, ms: int) -> Dict[str, float]:
        state = self.regime_return_states[ms]
        if state.n <= 0:
            return self._return_distribution_stats([])
        eps = 1e-9
        n = float(state.n)
        mean = state.sum1 / n
        m2 = max(0.0, state.sum2 / n - mean * mean)
        std = math.sqrt(max(m2, 0.0))
        m3 = state.sum3 / n - 3.0 * mean * (state.sum2 / n) + 2.0 * mean * mean * mean
        m4 = state.sum4 / n - 4.0 * mean * (state.sum3 / n) + 6.0 * (mean * mean) * (state.sum2 / n) - 3.0 * mean ** 4
        skew = 0.0
        kurt = 0.0
        if state.n >= 3 and std > eps:
            skew = m3 / (std ** 3)
        if state.n >= 4 and std > eps:
            kurt = m4 / (std ** 4)
        up = math.sqrt(max(state.up_sumsq, 0.0))
        down = math.sqrt(max(state.down_sumsq, 0.0))
        realized_var = max(state.sum2, 0.0)
        bipower = max(state.bipower, 0.0)
        return {
            "realized_up_vol_bps": float(up),
            "realized_down_vol_bps": float(down),
            "down_up_vol_ratio": self._safe_div(down, max(up, eps), 0.0),
            "bipower_variation": float(bipower),
            "jump_variation": float(max(realized_var - bipower, 0.0)),
            "max_abs_return_bps": float(state.max_abs_q[0][2]) if state.max_abs_q else 0.0,
            "return_skew": float(skew) if math.isfinite(skew) else 0.0,
            "return_kurtosis": float(kurt) if math.isfinite(kurt) else 0.0,
        }

    def _realized_vol_for_pressure(self, ms: int) -> float:
        """Return a finite realized-vol proxy for pressure features.

        FAST_WINDOWS_MS is intentionally shorter than REGIME_WINDOWS_MS in the 1s
        maker contract. Therefore self.realized_vol may not contain every requested
        short window, especially 200ms.

        Priority:
          1. Use self.realized_vol[ms] when available.
          2. Use return_histories[ms].mean_var() when available.
          3. Fall back to the shortest available realized_vol value.
          4. Return 0.0 only if no volatility state exists yet.
        """
        ms_i = int(ms)

        if ms_i in self.realized_vol:
            val = float(self.realized_vol[ms_i])
            return val if math.isfinite(val) and val >= 0.0 else 0.0

        stats = self.return_histories.get(ms_i)
        if stats is not None:
            _mean, var = stats.mean_var()
            val = math.sqrt(max(float(var), 0.0))
            return val if math.isfinite(val) and val >= 0.0 else 0.0

        if self.realized_vol:
            fallback_ms = min(int(k) for k in self.realized_vol.keys())
            val = float(self.realized_vol.get(fallback_ms, 0.0))
            return val if math.isfinite(val) and val >= 0.0 else 0.0

        return 0.0

    def _transform_features(self, raw: np.ndarray, dt_ms: float) -> np.ndarray:
        names = self.feature_names()
        if self._feature_transform_engine is None:
            self._feature_transform_engine = FeatureTransformEngine(names, enable_diagnostics=True)
        return self._feature_transform_engine.apply(raw, dt_ms)

    def transform_diagnostics_summary(self) -> Dict[str, Any]:
        if self._feature_transform_engine is None:
            return {"version": "feature_transform_diag_v1", "rows_seen": 0}
        return self._feature_transform_engine.diagnostics_summary()

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------
    def on_fast_event(self, e: Any) -> FeatureEventResult:
        """Fast ingest path for compact tuples.

        Returns FeatureEventResult with named fields: ts_ms, features, dt_ms,
        is_decision, raw_mid, and event_type. Trade rows update trade/flow
        state and event-density state only. They do not build features and do
        not update OB-clock feature state. OB rows are the only decision rows.
        """
        if not isinstance(e, tuple) or len(e) < 4 or not isinstance(e[0], str):
            raise ValueError(f"Expected compact ingest tuple, got: {e!r}")
        etype = e[0].lower()
        ts_ms = int(e[1])
        if etype == 'ob':
            payload = (int(e[3]), e[4], e[5])
        elif etype == 'trade':
            payload = e[3:8]
        else:
            raise ValueError(f"Unsupported compact event type: {etype!r}")
        return self._dispatch_parsed_event(etype, ts_ms, payload)

    def on_event(self, e: Any) -> FeatureEventResult:
        """Slow path returning FeatureEventResult for generic event shapes."""
        etype, ts_ms, payload = self._parse_event(e)

        if etype == 'ob':
            if isinstance(payload, tuple):
                compact_payload = (int(payload[0]), payload[1], payload[2])
            else:
                data = payload.get('data', payload)
                tp_raw = payload.get("type")
                if tp_raw is None and isinstance(data, dict):
                    tp_raw = data.get("type")
                if tp_raw is None:
                    tp_raw = payload.get("DataType")

                tp_norm = str(tp_raw or "").strip().lower()
                if tp_norm == "snapshot":
                    tp_code = 1
                elif tp_norm == "delta":
                    tp_code = 2
                else:
                    raise ValueError(f"Missing/unknown OB type in generic on_event payload: {tp_raw!r}")

                compact_payload = (
                    tp_code,
                    tuple((float(p), float(q)) for p, q in data.get("b", [])),
                    tuple((float(p), float(q)) for p, q in data.get("a", [])),
                )
            payload = compact_payload
        elif etype == 'trade' and not isinstance(payload, tuple):
            rpi_raw = payload.get('RPI')
            if rpi_raw is None:
                rpi_raw = payload.get('rpi')
            if isinstance(rpi_raw, str):
                rpi_norm = rpi_raw.strip().lower()
                is_rpi = 1 if rpi_norm in {"1", "true", "t", "yes", "y", "on"} else 0
            else:
                try:
                    is_rpi = 1 if float(rpi_raw) != 0.0 else 0
                except (TypeError, ValueError):
                    is_rpi = 0
            payload = (
                float(payload['price']),
                float(payload['size']),
                1 if str(payload['side']).lower() == 'buy' else -1 if str(payload['side']).lower() == 'sell' else 0,
                self._interpret_tick_direction(payload.get('tickDirection'))[0],
                is_rpi,
            )
        return self._dispatch_parsed_event(etype, ts_ms, payload)


class LabelBuilder:
    def __init__(self, delta_ms: int = 0, horizons_ms: Optional[List[int]] = None):
        self.delta = int(delta_ms)
        self.horizons = sorted(horizons_ms if horizons_ms is not None else HORIZONS_MS)
        assert len(self.horizons) > 0, "At least one horizon required"
        self.max_h = int(self.horizons[-1])

        # Decisions waiting to reach t_delta (entry time): items are t_delta timestamps
        self.wait_delta: Deque[int] = deque()
        # Decisions with entry price recorded, waiting to reach final maturity:
        # entries are (t_ready, t_delta, mid_entry)
        self.wait_mature: Deque[Tuple[int, int, float]] = deque()

        # Maintain recent midprice history as parallel arrays (timestamp, mid)
        self.price_ts: List[int] = []
        self.price_mid: List[float] = []
        self.price_start_idx = 0
        self.history_span = self.max_h + self.delta + 1000

        self.last_ts = -10**15
        self.last_mid: Optional[float] = None

    # ---- decision API ----
    def on_decision(self, t_now_ms: int):
        t_delta = t_now_ms + self.delta
        self.wait_delta.append(t_delta)

    def on_event(self, t_ms: int, mid_current: float):
        out = []
        t = int(t_ms)
        m = float(mid_current)

        # Record the latest price observation
        self._record_price(t, m)

        # Move items whose t_delta has passed into wait_mature, recording the midprice at t_delta
        while self.wait_delta and self.wait_delta[0] <= t:
            t_delta = self.wait_delta.popleft()
            mid0 = self._price_at(t_delta)
            t_ready = t_delta + self.max_h
            self.wait_mature.append((t_ready, t_delta, mid0))

        # Mature any items whose horizon has passed; compute labels using fixed-horizon returns
        while self.wait_mature and self.wait_mature[0][0] <= t:
            _, t_delta, mid0 = self.wait_mature.popleft()

            e = 1e-12
            mid0_safe = max(e, mid0)

            returns = []
            for horizon in self.horizons:
                mid_T = self._price_at(t_delta + int(horizon))
                mid_T_safe = max(e, mid_T)
                y_ret = 1e4 * math.log(mid_T_safe / mid0_safe)
                returns.append(y_ret)

            out.append(np.array(returns, dtype=np.float32))

        self.last_ts = t
        self.last_mid = m
        return out

    # ---- price history helpers ----
    def _push_price(self, ts: int, mid: float):
        if self.price_ts:
            last_ts = self.price_ts[-1]
            assert ts >= last_ts, "price timestamps must be non-decreasing"
            if ts == last_ts:
                self.price_mid[-1] = float(mid)
                return

        self.price_ts.append(ts)
        self.price_mid.append(float(mid))

    def _record_price(self, t: int, m: float):
        self._push_price(t, m)
        cutoff = t - self.history_span
        new_start = bisect_left(self.price_ts, cutoff, lo=self.price_start_idx)
        if new_start >= len(self.price_ts):
            new_start = max(0, len(self.price_ts) - 1)
        self.price_start_idx = new_start

        if self.price_start_idx > 0 and (
            self.price_start_idx >= 4096
            or self.price_start_idx >= len(self.price_ts) // 2
        ):
            self.price_ts = self.price_ts[self.price_start_idx:]
            self.price_mid = self.price_mid[self.price_start_idx:]
            self.price_start_idx = 0

    def _price_at(self, t_query: int) -> float:
        if not self.price_ts:
            return self.last_mid if self.last_mid is not None else 0.0

        start = self.price_start_idx
        if t_query < self.price_ts[start]:
            return self.price_mid[start]

        idx = bisect_right(self.price_ts, t_query, lo=start) - 1
        if idx < start:
            return self.price_mid[start]
        return self.price_mid[idx]


def _validate_flat_dataset_meta(meta: Dict[str, Any], source: str, *, require_storage_format: bool = True) -> None:
    if require_storage_format:
        storage_format = meta.get("storage_format")
        if storage_format != "flat_decision_rows_v1":
            raise ValueError(
                f"{source} has unsupported storage_format={storage_format!r}; expected 'flat_decision_rows_v1'."
            )

    label_dim = meta.get("label_dim")
    try:
        label_dim_int = int(label_dim)
    except (TypeError, ValueError):
        raise ValueError(f"{source} has invalid label_dim={label_dim!r}; expected {NUM_HORIZONS}.")
    if label_dim_int != int(NUM_HORIZONS):
        raise ValueError(f"{source} has label_dim={label_dim_int}; expected {NUM_HORIZONS}.")

    for field in ("feature_schema", "feature_transform", "feature_transform_policy", "feature_transform_spec_hash", "feature_transform_warmup_rows", "feature_dim_core", "feature_dim_total", "feature_names_hash", "aux_dim", "aux_names", "aux_transform", "lookback"):
        if field not in meta:
            raise ValueError(f"{source} missing required field '{field}'.")

    if meta.get("feature_schema") != FEATURE_SCHEMA:
        raise ValueError(
            f"{source} has feature_schema={meta.get('feature_schema')!r}; expected {FEATURE_SCHEMA!r}."
        )
    if meta.get("feature_transform") != FEATURE_TRANSFORM:
        raise ValueError(
            f"{source} has feature_transform={meta.get('feature_transform')!r}; expected {FEATURE_TRANSFORM!r}."
        )
    if meta.get("feature_transform_policy") != FEATURE_TRANSFORM_POLICY:
        raise ValueError(
            f"{source} has feature_transform_policy={meta.get('feature_transform_policy')!r}; expected {FEATURE_TRANSFORM_POLICY!r}."
        )
    if meta.get("aux_transform") != AUX_TRANSFORM:
        raise ValueError(
            f"Dataset aux_transform mismatch: got {meta.get('aux_transform')!r}, "
            f"expected {AUX_TRANSFORM!r}"
        )
    if int(meta.get("feature_transform_warmup_rows", -1)) != int(FEATURE_TRANSFORM_WARMUP_ROWS):
        raise ValueError(
            f"{source} has feature_transform_warmup_rows={meta.get('feature_transform_warmup_rows')!r}; expected {FEATURE_TRANSFORM_WARMUP_ROWS}."
        )
    if not str(meta.get("feature_transform_spec_hash", "")):
        raise ValueError(f"{source} has empty feature_transform_spec_hash.")
    if meta.get("feature_names"):
        expected_spec_hash = feature_transform_spec_hash(list(meta.get("feature_names", [])))
        if str(meta.get("feature_transform_spec_hash")) != expected_spec_hash:
            raise ValueError(
                f"{source} has feature_transform_spec_hash={meta.get('feature_transform_spec_hash')!r}; "
                f"expected {expected_spec_hash!r}."
            )

    for field in ("feature_dim_core", "feature_dim_total", "aux_dim", "lookback"):
        try:
            value = int(meta[field])
        except (TypeError, ValueError):
            raise ValueError(f"{source} has non-integer {field}={meta[field]!r}.")
        if value <= 0:
            raise ValueError(f"{source} has non-positive {field}={value}.")

    if int(meta["feature_dim_total"]) != int(meta["feature_dim_core"]) + int(meta["aux_dim"]):
        raise ValueError(
            f"{source} has feature_dim_total={meta['feature_dim_total']!r}, "
            f"feature_dim_core={meta['feature_dim_core']!r}, aux_dim={meta['aux_dim']!r}; "
            "expected feature_dim_total == feature_dim_core + aux_dim."
        )
    if int(meta["aux_dim"]) != int(AUX_DIM):
        raise ValueError(f"{source} has aux_dim={meta['aux_dim']!r}; expected {AUX_DIM}.")
    if list(meta.get("aux_names", [])) != list(FEATURE_AUX_TAIL):
        raise ValueError(
            f"{source} has aux_names={meta.get('aux_names')!r}; expected {list(FEATURE_AUX_TAIL)!r}."
        )
    if not str(meta.get("feature_names_hash", "")):
        raise ValueError(f"{source} missing non-empty feature_names_hash.")


class WeekFeatureStore:
    def __init__(self, week_dir: Path, week_meta: Dict[str, Any], lookback: int):
        self.week_dir = week_dir
        self.week_meta = week_meta
        self.lookback = int(lookback)
        self.feature_dim_total = int(week_meta["feature_dim_total"])
        self.feature_dim_core = int(week_meta["feature_dim_core"])
        self.aux_dim = int(week_meta["aux_dim"])
        self.feature_chunks = sorted(list(week_meta.get("feature_chunks", [])), key=lambda x: int(x["chunk"]))
        self.row_starts = np.array([int(ch["row_start"]) for ch in self.feature_chunks], dtype=np.int64)
        self.row_ends = np.array([int(ch["row_end"]) for ch in self.feature_chunks], dtype=np.int64)
        # Fully load the chunk into RAM by omitting mmap_mode (or calling .copy())
        # This eliminates the massive disk I/O bottleneck during shuffled Dataloader reads
        self.features_mm = [np.load(self.week_dir / ch["files"]["features"])[:] for ch in self.feature_chunks]

    def contiguous_features(self) -> np.ndarray:
        if self.row_starts.size == 0 or self.row_ends.size == 0:
            raise ValueError(f"No feature chunks found for week store {self.week_dir}")
        if int(self.row_starts[0]) != 0:
            raise ValueError(
                f"Feature chunks are not contiguous for {self.week_dir}: row_starts[0]={int(self.row_starts[0])}, expected 0"
            )
        for i in range(1, int(self.row_starts.size)):
            if int(self.row_starts[i]) != int(self.row_ends[i - 1]):
                raise ValueError(
                    f"Feature chunks are not contiguous for {self.week_dir}: "
                    f"row_starts[{i}]={int(self.row_starts[i])} != row_ends[{i-1}]={int(self.row_ends[i - 1])}"
                )
        out = np.concatenate(self.features_mm, axis=0).astype(np.float32, copy=False)
        expected_shape = (int(self.row_ends[-1]), int(self.feature_dim_total))
        if out.shape != expected_shape:
            raise ValueError(
                f"Contiguous feature matrix shape mismatch for {self.week_dir}: got {out.shape}, expected {expected_shape}"
            )
        return np.ascontiguousarray(out, dtype=np.float32)

    def _locate_chunk(self, row_idx: int) -> int:
        i = int(np.searchsorted(self.row_ends, int(row_idx), side="right"))
        if i >= len(self.row_ends) or row_idx < int(self.row_starts[i]) or row_idx >= int(self.row_ends[i]):
            raise IndexError(f"row_idx={row_idx} out of range for week store {self.week_dir}")
        return i

    def _read_row(self, row_idx: int) -> np.ndarray:
        ci = self._locate_chunk(int(row_idx))
        off = int(row_idx - int(self.row_starts[ci]))
        return np.asarray(self.features_mm[ci][off], dtype=np.float32)

    def read_window(self, row_idx: int, lookback: int) -> np.ndarray:
        L = int(lookback)
        end = int(row_idx)
        start = max(0, end - L + 1)
        n_real = end - start + 1
        out = np.empty((L, self.feature_dim_total), dtype=np.float32)

        pos = L - n_real
        cur = start
        while cur <= end:
            ci = self._locate_chunk(cur)
            chunk_start = int(self.row_starts[ci])
            chunk_end = int(self.row_ends[ci])
            take_end = min(end + 1, chunk_end)
            local_l = cur - chunk_start
            local_r = take_end - chunk_start
            n = local_r - local_l
            out[pos:pos+n] = np.asarray(self.features_mm[ci][local_l:local_r], dtype=np.float32)
            pos += n
            cur = take_end

        if n_real < L:
            earliest = out[L - n_real]
            out[: L - n_real, : self.feature_dim_core] = earliest[: self.feature_dim_core]
            out[: L - n_real, self.feature_dim_core :] = 0.0
        return out


class HFTFlatDataset(Dataset):
    def __init__(self, dataset_root: str, weeks: List[str], decision_ts_start: Optional[int] = None, decision_ts_end: Optional[int] = None):
        self.dataset_root = Path(dataset_root)
        self.meta = json.loads((self.dataset_root / "meta.json").read_text())
        _validate_flat_dataset_meta(self.meta, f"dataset metadata {self.dataset_root / 'meta.json'}")
        self.lookback = int(self.meta["lookback"])
        self.feature_dim_total = int(self.meta["feature_dim_total"])
        self.weeks = list(weeks)

        self.week_keys = list(self.weeks)
        self.week_to_id = {wk: i for i, wk in enumerate(self.week_keys)}
        self.stores: List[WeekFeatureStore] = []

        week_meta_paths = self.meta.get("weeks_meta", {})
        week_ids_parts: List[np.ndarray] = []
        row_idx_parts: List[np.ndarray] = []
        y_parts: List[np.ndarray] = []

        for wk in self.week_keys:
            rel = week_meta_paths.get(wk)
            if not rel:
                raise KeyError(f"Week {wk!r} missing from meta['weeks_meta']")
            week_meta_path = self.dataset_root / rel
            week_meta = json.loads(week_meta_path.read_text())
            _validate_flat_dataset_meta(week_meta, f"week metadata {week_meta_path}", require_storage_format=False)
            if int(week_meta["lookback"]) != int(self.lookback):
                raise ValueError(
                    f"week metadata {week_meta_path} has lookback={int(week_meta['lookback'])}, "
                    f"but dataset metadata requires lookback={self.lookback}."
                )
            if int(week_meta["feature_dim_total"]) != int(self.feature_dim_total):
                raise ValueError(
                    f"week metadata {week_meta_path} has feature_dim_total={int(week_meta['feature_dim_total'])}, "
                    f"but dataset metadata requires feature_dim_total={self.feature_dim_total}."
                )
            if int(week_meta["aux_dim"]) != int(self.meta["aux_dim"]):
                raise ValueError(
                    f"week metadata {week_meta_path} has aux_dim={int(week_meta['aux_dim'])}, "
                    f"but dataset metadata requires aux_dim={int(self.meta['aux_dim'])}."
                )
            store = WeekFeatureStore(week_meta_path.parent, week_meta, self.lookback)
            self.stores.append(store)

            for chunk in week_meta.get("label_chunks", []):
                files = chunk.get("files", {})
                row_idx_arr = np.load(week_meta_path.parent / files["row_idx"], mmap_mode="r")
                label_ts_arr = np.load(week_meta_path.parent / files["label_ts"], mmap_mode="r")
                y_arr = np.load(week_meta_path.parent / files["y"], mmap_mode="r")

                mask = np.ones(label_ts_arr.shape[0], dtype=bool)
                if decision_ts_start is not None:
                    mask &= (label_ts_arr >= int(decision_ts_start))
                if decision_ts_end is not None:
                    mask &= (label_ts_arr < int(decision_ts_end))
                row_idx_i64 = np.asarray(row_idx_arr, dtype=np.int64)
                full_history_mask = row_idx_i64 >= (self.lookback - 1)
                mask &= full_history_mask
                if not np.any(mask):
                    continue
                idx = np.nonzero(mask)[0]
                week_ids_parts.append(np.full((int(idx.shape[0]),), self.week_to_id[wk], dtype=np.int64))
                row_idx_parts.append(row_idx_i64[idx])
                y_parts.append(np.asarray(y_arr[idx], dtype=np.float32))
        self.week_ids = np.concatenate(week_ids_parts, axis=0).astype(np.int64, copy=False) if week_ids_parts else np.empty((0,), dtype=np.int64)
        self.row_idx = np.concatenate(row_idx_parts, axis=0).astype(np.int64, copy=False) if row_idx_parts else np.empty((0,), dtype=np.int64)
        self.y = np.concatenate(y_parts, axis=0).astype(np.float32, copy=False) if y_parts else np.empty((0, NUM_HORIZONS), dtype=np.float32)
        if self.y.shape[0] == 0:
            raise ValueError(
                "No rows remain in HFTFlatDataset after decision_ts/full-history filtering. "
                f"weeks={self.weeks}, decision_ts_start={decision_ts_start}, decision_ts_end={decision_ts_end}, lookback={self.lookback}"
            )

    def __len__(self) -> int:
        return int(self.y.shape[0])

    def __getitem__(self, idx: int):
        wk_id = int(self.week_ids[idx])
        row_idx = int(self.row_idx[idx])
        x_seq = self.stores[wk_id].read_window(row_idx, self.lookback)
        y_i = self.y[idx]
        return torch.from_numpy(x_seq), torch.from_numpy(y_i)


def build_dataset_from_split(dataset_root: str, split_cfg: Dict[str, Any]) -> HFTFlatDataset:
    meta = json.loads((Path(dataset_root) / "meta.json").read_text())
    _validate_flat_dataset_meta(meta, f"dataset metadata {Path(dataset_root) / 'meta.json'}")
    weeks = split_cfg.get("weeks")
    if weeks is None:
        wk = split_cfg.get("week")
        weeks = [wk] if wk else []
    if not weeks:
        raise ValueError("Split config must include 'week' or 'weeks'.")

    start = split_cfg.get("start")
    end = split_cfg.get("end")
    if start is None or end is None:
        dr = split_cfg.get("decision_ts_range") or {}
        start = dr.get("start")
        end = dr.get("end")
    return HFTFlatDataset(
        dataset_root=dataset_root,
        weeks=list(weeks),
        decision_ts_start=None if start is None else int(start),
        decision_ts_end=None if end is None else int(end),
    )


# --------------------  Utils ---------------------
def get_primary_metric_mode(metric_name: Optional[str] = None) -> str:
    metric = metric_name or PRIMARY_METRIC
    if (
        metric.startswith("dir_auc_kept")
        or metric.startswith("dir_auc_q50plus")
        or metric.startswith("edge_spearman_q50plus")
    ):
        return "max"
    raise ValueError(f"Unsupported primary metric '{metric}'")

def derive_dir_mag_predictions(pred: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    required = ("dir_logits", "mag_up_sqrt", "mag_down_sqrt")
    for key in required:
        if key not in pred:
            raise KeyError(f"Model output missing required key: {key}")
    dir_logits = pred["dir_logits"]
    mag_up_sqrt = pred["mag_up_sqrt"]
    mag_down_sqrt = pred["mag_down_sqrt"]
    if dir_logits.shape != mag_up_sqrt.shape or dir_logits.shape != mag_down_sqrt.shape:
        raise ValueError(
            "Invalid model output shapes: "
            f"dir_logits={tuple(dir_logits.shape)} "
            f"mag_up_sqrt={tuple(mag_up_sqrt.shape)} "
            f"mag_down_sqrt={tuple(mag_down_sqrt.shape)}"
        )
    p_up = torch.sigmoid(dir_logits)
    mag_up_bps = mag_up_sqrt.square()
    mag_down_bps = mag_down_sqrt.square()
    edge_bps = p_up * mag_up_bps - (1.0 - p_up) * mag_down_bps
    mag_pred_sqrt = p_up * mag_up_sqrt + (1.0 - p_up) * mag_down_sqrt
    return {
        "p_up": p_up,
        "mag_up_bps": mag_up_bps,
        "mag_down_bps": mag_down_bps,
        "edge_bps": edge_bps,
        "mag_pred_sqrt": mag_pred_sqrt,
    }

def derive_mag_pred_sqrt_for_mag_loss(pred: Dict[str, torch.Tensor]) -> torch.Tensor:
    p_up_detached = torch.sigmoid(pred["dir_logits"]).detach()
    return p_up_detached * pred["mag_up_sqrt"] + (1.0 - p_up_detached) * pred["mag_down_sqrt"]

def compute_primary_metric(metric_payload: Dict[str, Any]) -> Tuple[float, str]:
    if PRIMARY_METRIC_HORIZON_MS not in HORIZONS_MS:
        raise ValueError(
            f"PRIMARY_METRIC_HORIZON_MS={PRIMARY_METRIC_HORIZON_MS} not in HORIZONS_MS={HORIZONS_MS}"
        )

    idx = HORIZONS_MS.index(PRIMARY_METRIC_HORIZON_MS)

    if PRIMARY_METRIC.startswith("dir_auc_kept"):
        vals = metric_payload.get("dir_auc_kept", [])
        guard_vals = metric_payload.get("dir_bal_acc_kept", [])
    elif PRIMARY_METRIC.startswith("dir_auc_q50plus"):
        vals = metric_payload.get("dir_auc_q50plus", [])
        guard_vals = metric_payload.get("dir_bal_acc_q50plus", [])
    elif PRIMARY_METRIC.startswith("edge_spearman_q50plus"):
        vals = metric_payload.get("edge_spearman_q50plus", [])
        guard_vals = metric_payload.get("dir_bal_acc_q50plus", [])
    else:
        raise ValueError(f"Unsupported PRIMARY_METRIC={PRIMARY_METRIC!r}")

    if idx >= len(vals) or idx >= len(guard_vals):
        return float("nan"), PRIMARY_METRIC

    value = float(vals[idx])
    guard_value = float(guard_vals[idx])

    if not math.isfinite(value):
        return float("nan"), PRIMARY_METRIC
    if not math.isfinite(guard_value) or guard_value < PRIMARY_DIR_BAL_ACC_GUARD:
        return float("nan"), PRIMARY_METRIC

    return value, PRIMARY_METRIC

def is_metric_improved(value: float, best: float, mode: str) -> bool:
    if mode == "min":
        return value < best
    if mode == "max":
        return value > best
    raise ValueError(f"Unsupported mode '{mode}'")

# --------------------  Training loop  ---------------------


def build_datasets_from_meta_splits(dataset_root: str) -> Dict[str, HFTFlatDataset]:
    root = Path(dataset_root)
    meta = json.loads((root / "meta.json").read_text())
    splits = meta.get("splits", {})
    out: Dict[str, HFTFlatDataset] = {}
    for section in ("cmssl", "rl", "eval"):
        sec = splits.get(section, {})
        if not isinstance(sec, dict):
            continue
        for name, cfg in sec.items():
            if not isinstance(cfg, dict):
                continue
            out[f"{section}.{name}"] = build_dataset_from_split(dataset_root, cfg)
    return out
