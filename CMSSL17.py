import os, math, copy, json, csv, zipfile, io, gzip, contextlib, time
from pathlib import Path
from collections import deque
from bisect import bisect_left, bisect_right
from decimal import Decimal, ROUND_HALF_EVEN, InvalidOperation
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Deque, Any, List, Dict, Tuple, Generator, Optional, Iterable, Union, Sequence
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import math
from einops import rearrange, repeat
import torch._functorch.config as ft_config
from sklearn.decomposition import PCA

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
LOOKBACK        = 600        # canonical event-time token lookback
WINDOW_MS       = 60_000     # canonical rolling window span (60s)
PAD_DT_FOR_LEFT = 0.0
BATCH_SIZE      = 512
DMODEL          = 512
MAMBA_LAYERS    = 2
CONV_KERNELS    = [3,3,5,5,7,7]
DFF_CONV        = 2 * DMODEL

# Prediction horizons (in milliseconds)
HORIZONS_MS     = [7_500, 15_000, 30_000]
NUM_HORIZONS    = len(HORIZONS_MS)
HORIZON_WEIGHTS = [0.25, 0.5, 1.0]

LOW_ABS_TRIM_FRACTION = 0.05
HIGH_ABS_TRIM_FRACTION = 0.02
TARGET_TRANSFORM = "signed_sqrt_raw_bps"
TARGET_TASK = "horizon_specific_signed_raw_bps_targets"
FEATURE_SCHEMA = "cmssl17_30s_taker_features_v3"
AUX_SCHEMA = "cmssl17_aux_ob_decision_density_v2"
CHECKPOINT_SCHEMA = "cmssl17-signed-raw-v3"
EPOCHS          = 200
LR              = 4e-4
CLIP_GRAD       = 10000
PATIENCE        = 15
# Primary metric config (used for checkpointing + early stopping)
PRIMARY_METRIC = "spearman_kept_q50plus_30000ms"
PRIMARY_METRIC_HORIZON_MS = 30_000
SINGLE_WEEK_PATIENCE = 3
# Number of auxiliary channels appended after the base feature vector
# These correspond to:
# [log_dt_decision_ms, log_events_1000ms, log_events_3000ms, log_events_7500ms,
#  log_events_15000ms, log_events_30000ms, log_events_60000ms]
AUX_DIM        = 7

PRICE_WINDOWS_MS = (
    1_000, 3_000, 7_500, 15_000, 30_000, 60_000, 120_000, 300_000,
)
FAST_WINDOWS_MS = (1_000, 3_000, 7_500, 15_000, 30_000)
FLOW_WINDOWS_MS = (1_000, 3_000, 7_500, 15_000, 30_000, 60_000)
REGIME_WINDOWS_MS = (3_000, 7_500, 15_000, 30_000, 60_000, 120_000, 300_000)
EVENT_DENSITY_WINDOWS_MS = (1_000, 3_000, 7_500, 15_000, 30_000, 60_000)
EMA_HALF_LIVES_MS = (7_500, 15_000, 30_000, 60_000, 120_000)
MACD_TRIPLETS_MS = (
    (7_500, 15_000, 10_000),
    (15_000, 30_000, 20_000),
    (30_000, 60_000, 40_000),
    (60_000, 120_000, 90_000),
    (120_000, 300_000, 180_000),
)
VPIN_BUCKET_SECS = (7.5, 15.0, 30.0)
BOOK_DEPTH_FEATURE_LEVELS = (1, 2, 3, 5, 7, 10, 15, 20, 30, 50, 100)
NORMALIZED_OFI_LEVELS = (1, 3, 5, 10, 20, 50)
BPS_DEPTH_BANDS = (0.5, 1.0, 2.0, 3.0, 5.0, 7.5, 10.0, 15.0, 25.0, 50.0)
BOOK_SHAPE_BANDS = (1.0, 2.0, 5.0, 10.0)
SLIPPAGE_NOTIONAL_USD = (10_000.0, 25_000.0, 50_000.0, 100_000.0, 250_000.0)
LARGE_TRADE_NOTIONAL_USD = (50_000.0, 100_000.0, 250_000.0, 500_000.0)
MAX_BOOK_FEATURE_LEVEL = max(BOOK_DEPTH_FEATURE_LEVELS)

NUM_HEADS       = 8
# Loss mixing (fixed lambdas), with EMA normalization per loss
EMA_DECAY       = 0.99


# ---------------------------  Building blocks  ----------------------------
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
        hid_dim = 8
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
        self.offset_predictor = OffsetPredictor(in_feats, patch_size, stride)
        self.box_coder = BoxCoder(self.patch_count, stride, patch_size, self.seq_len, in_feats)
    def get_sampling_location(self, X):
        pred_offset = self.offset_predictor(X)
        sampling_locations, bound = self.box_coder(pred_offset)
        return sampling_locations, bound
    def forward(self, X, return_bound=False):
        img = X.unsqueeze(1)
        B = img.shape[0]
        sampling_locations, bound = self.get_sampling_location(X)
        sampling_locations = sampling_locations.view(B, self.patch_count*self.in_feats, self.patch_size, 2)
        sampling_locations = (sampling_locations - 0.5) * 2
        output = F.grid_sample(img, sampling_locations, align_corners=True)
        output = output.view(B, self.patch_count, self.in_feats, self.patch_size)
        output = output.permute(0, 2, 1, 3).contiguous()
        return output # (B, C, patch_count, patch_size)

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

class ConvTimeNetFeatureExtractor(nn.Module):
    def __init__(self, in_feats, seq_len, d_model, dw_ks, n_layers, d_ff=256, dropout=0.1, act='gelu', 
                 enable_res_param=True, norm='batch', re_param=True, re_param_kernel=3, patch_size=2, stride=1):
        super(ConvTimeNetFeatureExtractor, self).__init__()
        self.depatch = DepatchSampling(in_feats=in_feats, seq_len=seq_len, patch_size=patch_size, stride=stride)
        self.patch_count = (seq_len - patch_size) // stride + 1
        self.patch_size = patch_size
        self.d_model_internal = max(1, d_model // in_feats)
        self.d_ff_internal = max(2 * self.d_model_internal, 4)
        assert self.d_model_internal >= 1, "d_model_internal must be >= 1"
        assert self.d_ff_internal >= 2 * self.d_model_internal, "d_ff_internal must be >= 2 * d_model_internal"
        self.output_linear = nn.Linear(patch_size, self.d_model_internal)
        self.encoder = ConvEncoder(d_model=self.d_model_internal, d_ff=self.d_ff_internal, kernel_size=dw_ks, dropout=dropout, activation=act,
                                   n_layers=n_layers, enable_res_param=enable_res_param, norm=norm, re_param=re_param, small_ks=re_param_kernel)
        self.final_proj = nn.Linear(self.d_model_internal * in_feats, d_model)
    def forward(self, x):
        out_patch = self.depatch(x).contiguous()  # [B, feats, patch_count, patch_size]
        out = self.output_linear(out_patch).contiguous()  # [B, feats, patch_count, d_model_internal]
    
        B = out.shape[0]
    
        u = out.reshape(B * out.shape[1], out.shape[2], self.d_model_internal)
        u = u.permute(0, 2, 1).contiguous()  # [B * feats, d_model_internal, patch_count]
    
        out = self.encoder(u)
        out = out.permute(0, 2, 1).contiguous()  # [B * feats, patch_count, d_model_internal]
    
        out = out.reshape(B, out.shape[0] // B, out.shape[1], self.d_model_internal)
        out = out.permute(0, 2, 3, 1).contiguous()  # [B, patch_count, d_model_internal, feats]
    
        out = out.reshape(B, self.depatch.patch_count, self.d_model_internal * out.shape[3]).contiguous()
        out = self.final_proj(out).contiguous()  # [B, patch_count, d_model]
        return out

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

class Mamba(nn.Module):
    """Bidirectional (forward/backward) Mamba stacks with gated pooling."""

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
        self.pool = GatedPooling(2 * args.d_model)

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

        pooled = self.pool(h)
        return pooled, h, h_fwd

# -------------  SAMBA -------------
class SAMBA(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        assert args.d_model == DMODEL, f"Expected args.d_model ({args.d_model}) == DMODEL ({DMODEL})"
        assert args.d_model % NUM_HEADS == 0, "args.d_model must be divisible by NUM_HEADS"
        assert args.headdim == (args.d_model // NUM_HEADS), "args.headdim must match d_model // NUM_HEADS"
        # (3) Switch BatchNorm -> LayerNorm in ConvTimeNet
        self.depatch_proj_encoder = ConvTimeNetFeatureExtractor(
            in_feats=args.vocab_size, seq_len=args.seq_in, d_model=args.d_model, 
            dw_ks=[3,3,5,5,7,7], n_layers=6, d_ff=DFF_CONV, dropout=0.1, act='gelu',
            enable_res_param=False, norm='layer', re_param=True, re_param_kernel=3, 
            patch_size=2, stride=1
        )
        # Mamba backbone (forward/backward fusion) + pooling
        self.mamba = Mamba(args, ff_hid=4*DMODEL)

        # Heads
        fused_dim = args.d_model * 2
        head_hidden_dim = fused_dim * 2
        self.return_head = nn.Sequential(
            nn.Linear(fused_dim, head_hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(head_hidden_dim, NUM_HORIZONS)
        )

    def forward(self, x):
        x_permuted = x.permute(0, 2, 1).contiguous()
        h_tokens = self.depatch_proj_encoder(x_permuted).contiguous()        # [B, L, D] (ConvTimeNet projection applied)

        pooled, _, _ = self.mamba(h_tokens, embedded=True)
        pred_return = self.return_head(pooled)

        return pred_return

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
    def __init__(self, window_ms: int = 60_000):
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


# -------------------------  Feature engine  -------------------------
class FeatureEngine:
    """30s taker feature contract engine (OB decision-time tokens, PCA-required pipeline)."""

    def __init__(self, depth: int = MAX_BOOK_FEATURE_LEVEL, z_hl_ms: int = 30_000, vpin_target_bucket_secs: float = 30.0):
        self.depth = int(depth)
        if self.depth < MAX_BOOK_FEATURE_LEVEL:
            raise ValueError(f"FeatureEngine depth={self.depth} must be >= {MAX_BOOK_FEATURE_LEVEL}")

        self.bids: Dict[float, float] = {}
        self.asks: Dict[float, float] = {}
        self.bid_lvls: List[Tuple[float, float]] = []
        self.ask_lvls: List[Tuple[float, float]] = []
        self.last_ts: Optional[int] = None
        self.last_trade_ts: Optional[int] = None

        self.last_bid_price_change_ts: Optional[int] = None
        self.last_ask_price_change_ts: Optional[int] = None
        self.last_mid_change_ts: Optional[int] = None
        self.last_spread_widen_ts: Optional[int] = None
        self.last_spread_tighten_ts: Optional[int] = None
        self.current_bid_lifetime_start_ts: Optional[int] = None
        self.current_ask_lifetime_start_ts: Optional[int] = None

        self.last_bid1: Optional[float] = None
        self.last_ask1: Optional[float] = None
        self.last_mid: Optional[float] = None
        self.last_spread_bps: Optional[float] = None

        self.price_history_keep_ms = max(PRICE_WINDOWS_MS) + 5_000
        self._price_ts: Deque[int] = deque()
        self._mid_history: Deque[float] = deque()
        self._micro_history: Deque[float] = deque()

        self._event_density_deques: Dict[int, Deque[int]] = {ms: deque() for ms in EVENT_DENSITY_WINDOWS_MS}

        self.trade_windows: Tuple[int, ...] = FLOW_WINDOWS_MS
        self._trade_window_deques: Dict[int, Deque[Tuple[int, float, float, float, str, int, int, int]]] = {ms: deque() for ms in self.trade_windows}
        self._quote_window_deques: Dict[int, Deque[int]] = {ms: deque() for ms in self.trade_windows}
        self._trade_stats: Dict[int, Dict[str, float]] = {ms: self._empty_trade_stats() for ms in self.trade_windows}

        self.last_trade_price: Optional[float] = None
        self.last_tick_sign_internal: int = 0

        self._spread_hist: Dict[int, Deque[Tuple[int, float]]] = {ms: deque() for ms in REGIME_WINDOWS_MS}
        self._depth5_total_hist: Dict[int, Deque[Tuple[int, float]]] = {ms: deque() for ms in REGIME_WINDOWS_MS}
        self._depth5_imb_hist: Dict[int, Deque[Tuple[int, float]]] = {ms: deque() for ms in REGIME_WINDOWS_MS}
        self._depth10_total_hist: Dict[int, Deque[Tuple[int, float]]] = {ms: deque() for ms in REGIME_WINDOWS_MS}
        self._depth10_imb_hist: Dict[int, Deque[Tuple[int, float]]] = {ms: deque() for ms in REGIME_WINDOWS_MS}

        self.return_histories: Dict[int, Deque[Tuple[int, float]]] = {ms: deque() for ms in FLOW_WINDOWS_MS}
        self._regime_return_deques: Dict[int, Deque[Tuple[int, float]]] = {ms: deque() for ms in REGIME_WINDOWS_MS}
        self.last_mid_for_ret: Optional[float] = None
        self.realized_vol: Dict[int, float] = {ms: 0.0 for ms in REGIME_WINDOWS_MS}
        self.rv_ewma: Dict[int, float] = {ms: 0.0 for ms in REGIME_WINDOWS_MS}
        self.volume_ewma: Dict[int, float] = {ms: 0.0 for ms in REGIME_WINDOWS_MS}
        self.flow_regime: Dict[int, float] = {ms: 0.0 for ms in REGIME_WINDOWS_MS}

        self.cvd_notional: float = 0.0
        self._cvd_history: Deque[Tuple[int, float]] = deque()
        self._cvd_ema: Dict[int, float] = {ms: 0.0 for ms in FLOW_WINDOWS_MS}

        self.last_large_buy_ts: Optional[int] = None
        self.last_large_sell_ts: Optional[int] = None

        self.vpin_bucket_secs: Tuple[float, ...] = VPIN_BUCKET_SECS
        self.vpin_state: Dict[float, Dict[str, Any]] = {
            secs: {"Vb": None, "cum_buy": 0.0, "cum_sell": 0.0, "cum": 0.0, "phi": deque(maxlen=50)}
            for secs in self.vpin_bucket_secs
        }

        self.ema_half_lives_ms = EMA_HALF_LIVES_MS
        self.ema_indicator_names = (
            "spread_bps", "gap_a_bps", "gap_b_bps", "micro_premia", "micro_minus_mid_bps",
            "depth_imbalance_within_1.0bps", "depth_imbalance_within_2.0bps", "depth_imbalance_within_5.0bps", "depth_imbalance_within_10.0bps",
            "notional_imbalance_within_1.0bps", "notional_imbalance_within_2.0bps", "notional_imbalance_within_5.0bps", "notional_imbalance_within_10.0bps",
            "obi_l1", "obi_l3", "obi_l5", "obi_l10",
            "ofi_l1_over_depth_l1", "ofi_l3_over_depth_l3", "ofi_l5_over_depth_l5", "ofi_l10_over_depth_l10",
            "signed_notional_flow_usd_30000ms", "trade_imbalance_notional_30000ms", "vwap_vs_mid_bps_30000ms",
        )
        self.ema_states: Dict[int, Dict[str, Optional[float]]] = {
            hl: {name: None for name in self.ema_indicator_names} for hl in self.ema_half_lives_ms
        }

        self.macd_state: Dict[int, Dict[str, Optional[float]]] = {
            idx: {"fast": None, "slow": None, "signal": None} for idx in range(len(MACD_TRIPLETS_MS))
        }

        self._feature_names_cache: Optional[List[str]] = None
        self._feat_dim: Optional[int] = None
        self.z_mean: Optional[np.ndarray] = None
        self.z_m2: Optional[np.ndarray] = None
        self.z_hl_vec: Optional[List[Optional[int]]] = None

        self.timer_parse_dispatch_s = 0.0
        self.timer_book_update_s = 0.0
        self.timer_trade_update_s = 0.0
        self.timer_feature_build_s = 0.0

    def feature_schema(self) -> str:
        return FEATURE_SCHEMA

    def aux_schema(self) -> str:
        return AUX_SCHEMA

    def core_feature_dim(self) -> int:
        if self._feature_names_cache is None:
            raise ValueError("Core feature dimension unknown before first event")
        return len(self._feature_names_cache)

    def feature_dim(self) -> int:
        return self.core_feature_dim() + AUX_DIM

    def feature_names(self) -> List[str]:
        if self._feature_names_cache is not None:
            return list(self._feature_names_cache)
        names: List[str] = []
        names += [
            "time_hour_sin", "time_hour_cos", "time_minute_sin", "time_minute_cos", "time_dow_sin", "time_dow_cos",
            "session_is_weekend", "session_is_asia", "session_is_europe", "session_is_us", "session_is_europe_us_overlap",
        ]
        for w in PRICE_WINDOWS_MS:
            names += [
                f"mid_ret_bps_{w}ms", f"micro_ret_bps_{w}ms", f"mid_slope_bps_per_sec_{w}ms", f"mid_trend_r2_{w}ms",
                f"mid_position_in_range_{w}ms", f"mid_dist_to_high_bps_{w}ms", f"mid_dist_to_low_bps_{w}ms", f"mid_range_bps_{w}ms",
                f"mid_breakout_up_{w}ms", f"mid_breakout_down_{w}ms", f"sign_persistence_{w}ms", f"up_return_fraction_{w}ms", f"return_autocorr_lag1_{w}ms",
            ]
        names += [
            "spread_bps", "gap_a_bps", "gap_b_bps", "bsz1", "asz1", "micro_premia", "micro_minus_mid_bps", "micro_minus_mid_over_spread",
            "time_since_trade_ms", "time_since_bid_price_change_ms", "time_since_ask_price_change_ms", "time_since_mid_change_ms",
            "time_since_spread_widen_ms", "time_since_spread_tighten_ms", "best_bid_lifetime_ms", "best_ask_lifetime_ms", "mid_price_staleness_ms",
        ]
        for lvl in BOOK_DEPTH_FEATURE_LEVELS: names += [f"cum_bid_l{lvl}", f"cum_ask_l{lvl}"]
        for lvl in BOOK_DEPTH_FEATURE_LEVELS: names.append(f"obi_l{lvl}")
        for lvl in BOOK_DEPTH_FEATURE_LEVELS: names.append(f"ofi_l{lvl}")
        for lvl in NORMALIZED_OFI_LEVELS:
            names += [f"ofi_l{lvl}_over_depth_l{lvl}", f"ofi_l{lvl}_over_spread_bps", f"ofi_l{lvl}_over_depth_5bps"]
        for band in BPS_DEPTH_BANDS:
            b = str(band).replace('.', 'p')
            names += [
                f"bid_depth_within_{b}bps", f"ask_depth_within_{b}bps", f"bid_notional_within_{b}bps", f"ask_notional_within_{b}bps",
                f"depth_imbalance_within_{b}bps", f"notional_imbalance_within_{b}bps",
            ]
        for band in BOOK_SHAPE_BANDS:
            b = str(band).replace('.', 'p')
            names += [
                f"max_bid_size_within_{b}bps", f"max_ask_size_within_{b}bps", f"max_bid_notional_within_{b}bps", f"max_ask_notional_within_{b}bps",
                f"dist_to_max_bid_wall_bps_within_{b}bps", f"dist_to_max_ask_wall_bps_within_{b}bps", f"bid_depth_hhi_within_{b}bps", f"ask_depth_hhi_within_{b}bps",
                f"bid_top1_share_within_{b}bps", f"ask_top1_share_within_{b}bps",
            ]
        names += ["book_slope_bid_top5", "book_slope_ask_top5", "book_slope_bid_5bps", "book_slope_ask_5bps", "book_convexity_bid_10bps", "book_convexity_ask_10bps"]
        for notion in SLIPPAGE_NOTIONAL_USD:
            n = f"{int(notion)}usd"
            names += [f"slippage_bps_to_buy_{n}", f"slippage_bps_to_sell_{n}", f"depth_needed_bps_to_buy_{n}", f"depth_needed_bps_to_sell_{n}", f"filled_fraction_to_buy_{n}", f"filled_fraction_to_sell_{n}"]
        for ms in FAST_WINDOWS_MS:
            names += [
                f"spread_delta_bps_{ms}ms", f"spread_change_count_{ms}ms", f"bid_price_change_count_{ms}ms", f"ask_price_change_count_{ms}ms",
                f"bid_price_change_rate_{ms}ms", f"ask_price_change_rate_{ms}ms", f"bid_l1_depletion_{ms}ms", f"ask_l1_depletion_{ms}ms",
                f"bid_l1_depletion_over_depth_{ms}ms", f"ask_l1_depletion_over_depth_{ms}ms",
            ]
            for level in (1, 2):
                names += [
                    f"bid_l{level}_add_rate_{ms}ms", f"bid_l{level}_rem_rate_{ms}ms", f"ask_l{level}_add_rate_{ms}ms", f"ask_l{level}_rem_rate_{ms}ms",
                    f"bid_l{level}_add_rate_over_depth_{ms}ms", f"bid_l{level}_rem_rate_over_depth_{ms}ms", f"ask_l{level}_add_rate_over_depth_{ms}ms", f"ask_l{level}_rem_rate_over_depth_{ms}ms",
                ]
        # condensed: remaining names are generated dynamically in dispatch to avoid manual drift
        # guarantee deterministic order by storing once built from first emission
        self._feature_names_cache = names
        return list(self._feature_names_cache)

    @staticmethod
    def _safe_div(num: float, den: float, default: float = 0.0) -> float:
        if not (math.isfinite(num) and math.isfinite(den)) or abs(den) <= 1e-12:
            return float(default)
        return float(num / den)

    @staticmethod
    def _bps_return(current: float, past: float) -> float:
        if current <= 0.0 or past <= 0.0 or not (math.isfinite(current) and math.isfinite(past)):
            return 0.0
        return float(1e4 * math.log(current / past))

    def _alpha(self, dt_ms: float, hl_ms: float) -> float:
        return float(1.0 - math.pow(0.5, max(1.0, dt_ms) / max(1.0, hl_ms)))

    def _append_tuple_with_guard(self, deq: Deque[Tuple[int, Any]], entry: Tuple[int, Any], ts_ms: int, window_ms: int, is_ob_event: bool):
        deq.append(entry)
        while deq and (ts_ms - int(deq[0][0]) > window_ms):
            deq.popleft()

    def _append_price_history(self, ts_ms: int, mid: float, micro: float) -> None:
        self._price_ts.append(ts_ms)
        self._mid_history.append(mid)
        self._micro_history.append(micro)
        self._prune_price_history(ts_ms)

    def _prune_price_history(self, ts_ms: int) -> None:
        while self._price_ts and ts_ms - self._price_ts[0] > self.price_history_keep_ms:
            self._price_ts.popleft(); self._mid_history.popleft(); self._micro_history.popleft()

    def _price_asof(self, series_deque: Deque[float], t_query: int) -> float:
        if not self._price_ts:
            return 0.0
        ts_list = list(self._price_ts)
        idx = bisect_right(ts_list, t_query) - 1
        if idx < 0:
            return float(series_deque[0])
        return float(list(series_deque)[idx])

    def _window_price_points(self, now_ms: int, window_ms: int, series: str = "mid") -> List[Tuple[int, float]]:
        vals = self._mid_history if series == "mid" else self._micro_history
        out = []
        for ts, v in zip(self._price_ts, vals):
            if now_ms - ts <= window_ms:
                out.append((ts, float(v)))
        return out

    def _rolling_stats(self, values: List[float]) -> Tuple[float, float, float, float]:
        if not values:
            return 0.0, 0.0, 0.0, 0.0
        arr = np.asarray(values, dtype=np.float64)
        return float(arr.mean()), float(arr.std()), float(arr.min()), float(arr.max())

    def _rolling_slope_r2(self, ts_values: List[int], y_values: List[float]) -> Tuple[float, float]:
        if len(ts_values) < 3:
            return 0.0, 0.0
        x = (np.asarray(ts_values, dtype=np.float64) - float(ts_values[0])) / 1000.0
        y = np.asarray(y_values, dtype=np.float64)
        A = np.vstack([x, np.ones_like(x)]).T
        slope, intercept = np.linalg.lstsq(A, y, rcond=None)[0]
        yhat = slope * x + intercept
        ss_res = float(np.sum((y - yhat) ** 2))
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        r2 = 0.0 if ss_tot <= 1e-12 else max(0.0, 1.0 - ss_res / ss_tot)
        return float(slope), float(r2)

    def _range_position(self, values: List[float], current: float) -> Tuple[float, float, float, float, float, float]:
        if not values:
            return 0.5, 0.0, 0.0, 0.0, 0.0, 0.0
        lo, hi = min(values), max(values)
        if hi - lo <= 1e-12:
            pos = 0.5
        else:
            pos = (current - lo) / (hi - lo)
        dist_hi = self._bps_return(hi, current)
        dist_lo = self._bps_return(current, lo)
        rng = self._bps_return(hi, lo) if hi > 0 and lo > 0 else 0.0
        prev = values[:-1]
        up = 1.0 if len(prev) >= 3 and current >= max(prev) else 0.0
        dn = 1.0 if len(prev) >= 3 and current <= min(prev) else 0.0
        return float(pos), float(dist_hi), float(dist_lo), float(rng), up, dn

    def _depth_within_bps(self, levels: List[Tuple[float, float]], mid: float, band_bps: float, is_bid: bool) -> Dict[str, float]:
        total_size = total_notional = max_size = max_notional = 0.0
        dist_to_max = 0.0
        parts = []
        for p, s in levels:
            if mid <= 0 or s <= 0:
                continue
            dist = 1e4 * ((mid - p) / mid if is_bid else (p - mid) / mid)
            if dist <= band_bps + 1e-12:
                total_size += s
                n = p * s
                total_notional += n
                parts.append(s)
                if s > max_size:
                    max_size = s; max_notional = n; dist_to_max = dist
        if total_size <= 0:
            return {"size": 0.0, "notional": 0.0, "max_size": 0.0, "max_notional": 0.0, "dist_to_max_bps": 0.0, "hhi": 0.0, "top1_share": 0.0}
        shares = np.asarray(parts, dtype=np.float64) / total_size
        return {
            "size": total_size, "notional": total_notional, "max_size": max_size, "max_notional": max_notional,
            "dist_to_max_bps": dist_to_max, "hhi": float(np.sum(shares ** 2)), "top1_share": max_size / total_size,
        }

    def _slippage_for_notional(self, levels: List[Tuple[float, float]], mid: float, notional_usd: float, is_buy: bool) -> Dict[str, float]:
        if mid <= 0 or notional_usd <= 0:
            return {"slippage_bps": 0.0, "depth_needed_bps": 0.0, "filled_fraction": 0.0}
        rem, spent, qty = float(notional_usd), 0.0, 0.0
        last_dist = 0.0
        for p, s in levels[:100]:
            lvl_not = p * s
            take_not = min(rem, lvl_not)
            if take_not <= 0:
                continue
            take_qty = take_not / max(p, 1e-12)
            spent += take_not
            qty += take_qty
            rem -= take_not
            last_dist = 1e4 * abs(p - mid) / max(mid, 1e-12)
            if rem <= 1e-9:
                break
        fill = spent / max(notional_usd, 1e-12)
        if fill < 1.0 - 1e-9:
            return {"slippage_bps": 10_000.0, "depth_needed_bps": 10_000.0, "filled_fraction": float(fill)}
        vwap = spent / max(qty, 1e-12)
        if is_buy:
            cost = max(vwap / mid - 1.0, 0.0)
        else:
            cost = max(mid / max(vwap, 1e-12) - 1.0, 0.0)
        return {"slippage_bps": float(1e4 * cost), "depth_needed_bps": float(last_dist), "filled_fraction": float(fill)}

    def _compute_session_features(self, ts_ms: int) -> Dict[str, float]:
        dt = datetime.fromtimestamp(ts_ms / 1000.0, tz=timezone.utc)
        h, m, dow = dt.hour, dt.minute, dt.weekday()
        hour_phase = 2.0 * math.pi * (h / 24.0)
        minute_phase = 2.0 * math.pi * (m / 60.0)
        dow_phase = 2.0 * math.pi * (dow / 7.0)
        asia = 1.0 if 0 <= h < 8 else 0.0
        eur = 1.0 if 7 <= h < 16 else 0.0
        us = 1.0 if 13 <= h < 22 else 0.0
        overlap = 1.0 if 13 <= h < 16 else 0.0
        wknd = 1.0 if dow >= 5 else 0.0
        return {
            "time_hour_sin": math.sin(hour_phase), "time_hour_cos": math.cos(hour_phase),
            "time_minute_sin": math.sin(minute_phase), "time_minute_cos": math.cos(minute_phase),
            "time_dow_sin": math.sin(dow_phase), "time_dow_cos": math.cos(dow_phase),
            "session_is_weekend": wknd, "session_is_asia": asia, "session_is_europe": eur, "session_is_us": us, "session_is_europe_us_overlap": overlap,
        }

    def _feature_z_half_life_ms(self, feature_name: str) -> Optional[int]:
        if feature_name.startswith("time_") or feature_name.startswith("session_"):
            return None
        n = feature_name
        if any(k in n for k in ("spread_delta", "ofi_", "obi_", "pressure", "replen", "change_count", "depletion")):
            return 30_000
        if any(k in n for k in ("trade", "flow", "activity", "large", "vpin")):
            return 60_000
        if any(k in n for k in ("price", "trend", "range", "macd", "return", "vol", "regime", "absorption", "impact", "slippage", "depth")):
            return 120_000
        return 120_000

    def _zscore(self, x: np.ndarray, dt_ms: float) -> np.ndarray:
        assert x.ndim == 1, "x must be 1D"
        if not np.all(np.isfinite(x)):
            raise ValueError("Non-finite input to _zscore")
        x64 = x.astype(np.float64, copy=False)
        if self.z_mean is None:
            self.z_mean = x64.copy(); self.z_m2 = np.square(x64); self._feat_dim = int(x.shape[0])
            names = self.feature_names()
            if len(names) != self._feat_dim:
                self._feature_names_cache = names + [f"extra_feature_{i}" for i in range(self._feat_dim - len(names))]
                names = self.feature_names()
            self.z_hl_vec = [self._feature_z_half_life_ms(n) for n in names]
            assert len(self.z_hl_vec) == len(x64)
            return np.zeros_like(x64, dtype=np.float32)
        if len(self.z_hl_vec or []) != len(x64):
            raise ValueError("z_hl_vec length mismatch")
        out = np.zeros_like(x64, dtype=np.float64)
        eps = 1e-9
        for i, v in enumerate(x64):
            hl = self.z_hl_vec[i]
            if hl is None:
                out[i] = v
                self.z_mean[i] = v
                self.z_m2[i] = v * v
                continue
            a = self._alpha(dt_ms, float(hl))
            om = 1.0 - a
            self.z_mean[i] = om * self.z_mean[i] + a * v
            self.z_m2[i] = om * self.z_m2[i] + a * (v * v)
            var = max(self.z_m2[i] - self.z_mean[i] * self.z_mean[i], eps)
            out[i] = (v - self.z_mean[i]) / math.sqrt(var)
        if not np.all(np.isfinite(out)):
            raise ValueError("Non-finite output from _zscore")
        return out.astype(np.float32)

    def _parse_event(self, e: Any) -> Tuple[str, int, Any]:
        if isinstance(e, tuple) and len(e) >= 4 and isinstance(e[0], str):
            etype = e[0].lower(); ts = int(e[1])
            if etype == "ob":
                return etype, ts, (int(e[3]), e[4], e[5])
            return etype, ts, e[3:]
        raise ValueError(f"Unsupported event: {e!r}")

    def _sorted_ladders(self):
        self.bid_lvls = sorted(self.bids.items(), key=lambda x: x[0], reverse=True)[: self.depth]
        self.ask_lvls = sorted(self.asks.items(), key=lambda x: x[0])[: self.depth]

    def _update_book_from_ob(self, tp_code: int, bids: Sequence[Tuple[float, float]], asks: Sequence[Tuple[float, float]]) -> None:
        if int(tp_code) == 1:
            self.bids = {float(p): float(q) for p, q in bids[: self.depth] if float(q) > 0}
            self.asks = {float(p): float(q) for p, q in asks[: self.depth] if float(q) > 0}
        else:
            for p, q in bids:
                p, q = float(p), float(q)
                if q <= 0: self.bids.pop(p, None)
                else: self.bids[p] = q
            for p, q in asks:
                p, q = float(p), float(q)
                if q <= 0: self.asks.pop(p, None)
                else: self.asks[p] = q
        self._sorted_ladders()

    def _empty_trade_stats(self) -> Dict[str, float]:
        return {
            "buy_cnt":0.0,"sell_cnt":0.0,"buy_vol_base":0.0,"sell_vol_base":0.0,"buy_notional_usd":0.0,"sell_notional_usd":0.0,
            "signed_notional_usd":0.0,"signed_trade_count":0.0,"plus_tick_count":0.0,"minus_tick_count":0.0,"zero_tick_count":0.0,
            "pxv_sum":0.0,"vol_sum":0.0,"buy_max_notional_q":0.0,"sell_max_notional_q":0.0,
        }

    def _update_trade_windows(self, ts_ms: int, payload: Any, dt_ms: float) -> None:
        price, size, side_code = float(payload[0]), float(payload[1]), int(payload[2])
        side_sign = 1 if side_code > 0 else -1 if side_code < 0 else 0
        side = "buy" if side_sign > 0 else "sell" if side_sign < 0 else "unknown"
        tick_sign = int(payload[3]) if len(payload) > 3 else 0
        is_zero_tick = 1 if tick_sign == 0 else 0
        if self.last_trade_price is not None:
            if price > self.last_trade_price: tick_sign, is_zero_tick = 1, 0
            elif price < self.last_trade_price: tick_sign, is_zero_tick = -1, 0
            else: is_zero_tick = 1
        self.last_trade_price = price
        notional = price * size
        entry = (ts_ms, price, size, notional, side, side_sign, tick_sign, is_zero_tick)
        for ms, dq in self._trade_window_deques.items():
            dq.append(entry)
            while dq and ts_ms - dq[0][0] > ms:
                dq.popleft()
        for ms in self.trade_windows:
            dq = self._trade_window_deques[ms]
            st = self._empty_trade_stats()
            for _, px, sz, nt, s, ss, tsign, zt in dq:
                st["pxv_sum"] += px * sz; st["vol_sum"] += sz
                if tsign > 0: st["plus_tick_count"] += 1
                elif tsign < 0: st["minus_tick_count"] += 1
                else: st["zero_tick_count"] += 1
                if ss > 0:
                    st["buy_cnt"] += 1; st["buy_vol_base"] += sz; st["buy_notional_usd"] += nt; st["buy_max_notional_q"] = max(st["buy_max_notional_q"], nt)
                    st["signed_notional_usd"] += nt; st["signed_trade_count"] += 1
                elif ss < 0:
                    st["sell_cnt"] += 1; st["sell_vol_base"] += sz; st["sell_notional_usd"] += nt; st["sell_max_notional_q"] = max(st["sell_max_notional_q"], nt)
                    st["signed_notional_usd"] -= nt; st["signed_trade_count"] -= 1
            self._trade_stats[ms] = st
        if side_sign != 0:
            self.cvd_notional += side_sign * notional
            self._cvd_history.append((ts_ms, self.cvd_notional))
            while self._cvd_history and ts_ms - self._cvd_history[0][0] > (max(FLOW_WINDOWS_MS) + 5000):
                self._cvd_history.popleft()
        if notional >= 100_000:
            if side_sign > 0: self.last_large_buy_ts = ts_ms
            elif side_sign < 0: self.last_large_sell_ts = ts_ms
        dt_trade_ms = max(1.0, float(ts_ms - self.last_trade_ts)) if self.last_trade_ts is not None else max(1.0, dt_ms)
        vol_rate = size / (dt_trade_ms / 1000.0)
        for hl in REGIME_WINDOWS_MS:
            a = self._alpha(dt_trade_ms, float(hl))
            self.volume_ewma[hl] = (1 - a) * self.volume_ewma[hl] + a * vol_rate
        self.last_trade_ts = ts_ms

    def _add_return(self, ts_ms: int, mid: float, is_ob_event: bool):
        if mid <= 0.0:
            return 0.0
        if self.last_mid_for_ret is None:
            self.last_mid_for_ret = mid
            return 0.0
        if not is_ob_event:
            return 0.0
        r = 1e4 * math.log(mid / self.last_mid_for_ret)
        self.last_mid_for_ret = mid
        for ms, deq in self.return_histories.items():
            self._append_tuple_with_guard(deq, (ts_ms, r), ts_ms, ms, is_ob_event)
        for ms, deq in self._regime_return_deques.items():
            self._append_tuple_with_guard(deq, (ts_ms, r), ts_ms, ms, is_ob_event)
            self.realized_vol[ms] = math.sqrt(sum(val * val for _, val in deq))
        dt_ms = 1.0 if self.last_ts is None else max(1.0, ts_ms - self.last_ts)
        r2 = r * r
        for hl in REGIME_WINDOWS_MS:
            a = self._alpha(dt_ms, float(hl))
            self.rv_ewma[hl] = (1 - a) * self.rv_ewma[hl] + a * r2
        return r

    def _event_density(self, window_ms: int) -> float:
        deq = self._event_density_deques.get(int(window_ms))
        return float(len(deq)) if deq is not None else 0.0

    def event_density_1000ms(self) -> float: return self._event_density(1_000)
    def event_density_3000ms(self) -> float: return self._event_density(3_000)
    def event_density_7500ms(self) -> float: return self._event_density(7_500)
    def event_density_15000ms(self) -> float: return self._event_density(15_000)
    def event_density_30000ms(self) -> float: return self._event_density(30_000)
    def event_density_60000ms(self) -> float: return self._event_density(60_000)

    def _dispatch_parsed_event(self, etype: str, ts_ms: int, payload: Any):
        t0 = time.perf_counter()
        is_trade = etype == "trade"
        prev_ts = self.last_ts
        dt_ms = 0.0 if prev_ts is None else max(0.0, float(ts_ms - prev_ts))

        if is_trade:
            self._update_trade_windows(ts_ms, payload, dt_ms)
        else:
            tp, bids, asks = payload
            self._update_book_from_ob(tp, bids, asks)
            for ms, deq in self._quote_window_deques.items():
                deq.append(ts_ms)
                while deq and ts_ms - deq[0] > ms: deq.popleft()

        for w, deq in self._event_density_deques.items():
            deq.append(ts_ms)
            while deq and ts_ms - deq[0] > w: deq.popleft()

        if not self.bid_lvls or not self.ask_lvls:
            z = np.zeros((len(self.feature_names()),), dtype=np.float32)
            self.last_ts = ts_ms
            return ts_ms, z, 0.0, is_trade, dt_ms

        bid1, bsz1 = self.bid_lvls[0]
        ask1, asz1 = self.ask_lvls[0]
        if bid1 <= 0 or ask1 <= 0 or ask1 < bid1:
            raise ValueError(f"Invalid top-of-book: bid1={bid1}, ask1={ask1}")

        mid = 0.5 * (bid1 + ask1)
        spread = max(ask1 - bid1, 1e-12)
        spread_bps = 1e4 * spread / max(mid, 1e-12)
        micro = (ask1 * bsz1 + bid1 * asz1) / max(bsz1 + asz1, 1e-12)

        if self.last_bid1 is None or bid1 != self.last_bid1:
            self.last_bid_price_change_ts = ts_ms
            self.current_bid_lifetime_start_ts = ts_ms
        if self.last_ask1 is None or ask1 != self.last_ask1:
            self.last_ask_price_change_ts = ts_ms
            self.current_ask_lifetime_start_ts = ts_ms
        if self.last_mid is None or mid != self.last_mid:
            self.last_mid_change_ts = ts_ms
        if self.last_spread_bps is not None:
            if spread_bps > self.last_spread_bps: self.last_spread_widen_ts = ts_ms
            elif spread_bps < self.last_spread_bps: self.last_spread_tighten_ts = ts_ms
        self.last_bid1, self.last_ask1, self.last_mid, self.last_spread_bps = bid1, ask1, mid, spread_bps

        self._append_price_history(ts_ms, mid, micro)
        self._add_return(ts_ms, mid, is_ob_event=(etype == "ob"))

        gap_a_bps = 0.0 if len(self.ask_lvls) < 2 else 1e4 * (self.ask_lvls[1][0] - ask1) / max(mid, 1e-12)
        gap_b_bps = 0.0 if len(self.bid_lvls) < 2 else 1e4 * (bid1 - self.bid_lvls[1][0]) / max(mid, 1e-12)
        micro_premia = (micro - mid) / max(spread, 1e-9)
        micro_minus_mid_bps = 1e4 * (micro / max(mid, 1e-12) - 1.0)
        micro_minus_mid_over_spread = (micro - mid) / max(spread, 1e-9)

        cum_bid = {lvl: float(sum(s for _, s in self.bid_lvls[:lvl])) for lvl in BOOK_DEPTH_FEATURE_LEVELS}
        cum_ask = {lvl: float(sum(s for _, s in self.ask_lvls[:lvl])) for lvl in BOOK_DEPTH_FEATURE_LEVELS}
        obi = {lvl: self._safe_div(cum_bid[lvl] - cum_ask[lvl], cum_bid[lvl] + cum_ask[lvl]) for lvl in BOOK_DEPTH_FEATURE_LEVELS}
        ofi = {lvl: (cum_bid[lvl] - cum_ask[lvl]) for lvl in BOOK_DEPTH_FEATURE_LEVELS}

        depth_bands = {}
        for band in BPS_DEPTH_BANDS:
            bd = self._depth_within_bps(self.bid_lvls, mid, float(band), True)
            ad = self._depth_within_bps(self.ask_lvls, mid, float(band), False)
            depth_bands[band] = (bd, ad)

        slip = {}
        for n in SLIPPAGE_NOTIONAL_USD:
            slip[(n, "buy")] = self._slippage_for_notional(self.ask_lvls, mid, n, True)
            slip[(n, "sell")] = self._slippage_for_notional(self.bid_lvls, mid, n, False)

        session = self._compute_session_features(ts_ms)

        feat_map: Dict[str, float] = {}
        feat_map.update(session)

        mid_ret_map = {}
        for w in PRICE_WINDOWS_MS:
            past_mid = self._price_asof(self._mid_history, ts_ms - w)
            past_micro = self._price_asof(self._micro_history, ts_ms - w)
            mid_ret = self._bps_return(mid, past_mid if past_mid > 0 else mid)
            micro_ret = self._bps_return(micro, past_micro if past_micro > 0 else micro)
            mid_ret_map[w] = mid_ret
            pts = self._window_price_points(ts_ms, w, "mid")
            tsv = [t for t, _ in pts]; yv = [v for _, v in pts]
            slope_raw, r2 = self._rolling_slope_r2(tsv, yv)
            slope_bps = 1e4 * self._safe_div(slope_raw, max(mid, 1e-9), 0.0)
            pos, d_hi, d_lo, rng, br_up, br_dn = self._range_position(yv, mid)
            rets = [self._bps_return(yv[i], yv[i-1]) for i in range(1, len(yv))] if len(yv) >= 2 else []
            if rets:
                sgn = np.sign(rets)
                sign_p = abs(float(np.sum(sgn))) / len(rets)
                up_frac = float(np.mean(np.asarray(rets) > 0.0))
                if len(rets) >= 3 and np.std(rets[:-1]) > 1e-12 and np.std(rets[1:]) > 1e-12:
                    ac = float(np.corrcoef(rets[:-1], rets[1:])[0, 1])
                else:
                    ac = 0.0
            else:
                sign_p = up_frac = ac = 0.0
            feat_map[f"mid_ret_bps_{w}ms"] = mid_ret
            feat_map[f"micro_ret_bps_{w}ms"] = micro_ret
            feat_map[f"mid_slope_bps_per_sec_{w}ms"] = slope_bps
            feat_map[f"mid_trend_r2_{w}ms"] = r2
            feat_map[f"mid_position_in_range_{w}ms"] = pos
            feat_map[f"mid_dist_to_high_bps_{w}ms"] = d_hi
            feat_map[f"mid_dist_to_low_bps_{w}ms"] = d_lo
            feat_map[f"mid_range_bps_{w}ms"] = rng
            feat_map[f"mid_breakout_up_{w}ms"] = br_up
            feat_map[f"mid_breakout_down_{w}ms"] = br_dn
            feat_map[f"sign_persistence_{w}ms"] = sign_p
            feat_map[f"up_return_fraction_{w}ms"] = up_frac
            feat_map[f"return_autocorr_lag1_{w}ms"] = ac

        feat_map.update({
            "spread_bps": spread_bps, "gap_a_bps": gap_a_bps, "gap_b_bps": gap_b_bps, "bsz1": bsz1, "asz1": asz1,
            "micro_premia": micro_premia, "micro_minus_mid_bps": micro_minus_mid_bps, "micro_minus_mid_over_spread": micro_minus_mid_over_spread,
            "time_since_trade_ms": float(ts_ms - self.last_trade_ts) if self.last_trade_ts is not None else 0.0,
            "time_since_bid_price_change_ms": float(ts_ms - self.last_bid_price_change_ts) if self.last_bid_price_change_ts is not None else 0.0,
            "time_since_ask_price_change_ms": float(ts_ms - self.last_ask_price_change_ts) if self.last_ask_price_change_ts is not None else 0.0,
            "time_since_mid_change_ms": float(ts_ms - self.last_mid_change_ts) if self.last_mid_change_ts is not None else 0.0,
            "time_since_spread_widen_ms": float(ts_ms - self.last_spread_widen_ts) if self.last_spread_widen_ts is not None else 0.0,
            "time_since_spread_tighten_ms": float(ts_ms - self.last_spread_tighten_ts) if self.last_spread_tighten_ts is not None else 0.0,
            "best_bid_lifetime_ms": float(ts_ms - self.current_bid_lifetime_start_ts) if self.current_bid_lifetime_start_ts is not None else 0.0,
            "best_ask_lifetime_ms": float(ts_ms - self.current_ask_lifetime_start_ts) if self.current_ask_lifetime_start_ts is not None else 0.0,
            "mid_price_staleness_ms": float(ts_ms - self.last_mid_change_ts) if self.last_mid_change_ts is not None else 0.0,
        })

        for lvl in BOOK_DEPTH_FEATURE_LEVELS:
            feat_map[f"cum_bid_l{lvl}"] = cum_bid[lvl]
            feat_map[f"cum_ask_l{lvl}"] = cum_ask[lvl]
            feat_map[f"obi_l{lvl}"] = obi[lvl]
            feat_map[f"ofi_l{lvl}"] = ofi[lvl]
        bd5 = depth_bands[5.0][0]["size"]; ad5 = depth_bands[5.0][1]["size"]
        for lvl in NORMALIZED_OFI_LEVELS:
            feat_map[f"ofi_l{lvl}_over_depth_l{lvl}"] = self._safe_div(ofi[lvl], cum_bid[lvl] + cum_ask[lvl])
            feat_map[f"ofi_l{lvl}_over_spread_bps"] = self._safe_div(ofi[lvl], max(spread_bps, 0.1))
            feat_map[f"ofi_l{lvl}_over_depth_5bps"] = self._safe_div(ofi[lvl], bd5 + ad5)

        for band in BPS_DEPTH_BANDS:
            bd, ad = depth_bands[band]
            b = str(band).replace('.', 'p')
            feat_map[f"bid_depth_within_{b}bps"] = bd["size"]
            feat_map[f"ask_depth_within_{b}bps"] = ad["size"]
            feat_map[f"bid_notional_within_{b}bps"] = bd["notional"]
            feat_map[f"ask_notional_within_{b}bps"] = ad["notional"]
            feat_map[f"depth_imbalance_within_{b}bps"] = self._safe_div(bd["size"] - ad["size"], bd["size"] + ad["size"])
            feat_map[f"notional_imbalance_within_{b}bps"] = self._safe_div(bd["notional"] - ad["notional"], bd["notional"] + ad["notional"])
        for band in BOOK_SHAPE_BANDS:
            bd, ad = depth_bands[band]
            b = str(band).replace('.', 'p')
            feat_map[f"max_bid_size_within_{b}bps"] = bd["max_size"]
            feat_map[f"max_ask_size_within_{b}bps"] = ad["max_size"]
            feat_map[f"max_bid_notional_within_{b}bps"] = bd["max_notional"]
            feat_map[f"max_ask_notional_within_{b}bps"] = ad["max_notional"]
            feat_map[f"dist_to_max_bid_wall_bps_within_{b}bps"] = bd["dist_to_max_bps"]
            feat_map[f"dist_to_max_ask_wall_bps_within_{b}bps"] = ad["dist_to_max_bps"]
            feat_map[f"bid_depth_hhi_within_{b}bps"] = bd["hhi"]
            feat_map[f"ask_depth_hhi_within_{b}bps"] = ad["hhi"]
            feat_map[f"bid_top1_share_within_{b}bps"] = bd["top1_share"]
            feat_map[f"ask_top1_share_within_{b}bps"] = ad["top1_share"]

        feat_map["book_slope_bid_top5"] = self._rolling_slope_r2(list(range(min(5, len(self.bid_lvls)))), [p for p, _ in self.bid_lvls[:5]])[0]
        feat_map["book_slope_ask_top5"] = self._rolling_slope_r2(list(range(min(5, len(self.ask_lvls)))), [p for p, _ in self.ask_lvls[:5]])[0]
        feat_map["book_slope_bid_5bps"] = 0.0
        feat_map["book_slope_ask_5bps"] = 0.0
        feat_map["book_convexity_bid_10bps"] = 0.0
        feat_map["book_convexity_ask_10bps"] = 0.0

        for n in SLIPPAGE_NOTIONAL_USD:
            k = f"{int(n)}usd"
            sb, ss = slip[(n, "buy")], slip[(n, "sell")]
            feat_map[f"slippage_bps_to_buy_{k}"] = sb["slippage_bps"]
            feat_map[f"slippage_bps_to_sell_{k}"] = ss["slippage_bps"]
            feat_map[f"depth_needed_bps_to_buy_{k}"] = sb["depth_needed_bps"]
            feat_map[f"depth_needed_bps_to_sell_{k}"] = ss["depth_needed_bps"]
            feat_map[f"filled_fraction_to_buy_{k}"] = sb["filled_fraction"]
            feat_map[f"filled_fraction_to_sell_{k}"] = ss["filled_fraction"]

        if self._feature_names_cache is None:
            self._feature_names_cache = sorted(feat_map.keys(), key=lambda x: list(feat_map.keys()).index(x))
        names = self.feature_names()
        # fill any missing from contract subset with zeros
        vec = []
        for n in names:
            v = float(feat_map.get(n, 0.0))
            if not math.isfinite(v):
                raise ValueError(f"Non-finite feature {n}={v}")
            vec.append(v)
        assert len(vec) == len(names)
        feat = np.asarray(vec, dtype=np.float64)
        feat_z = self._zscore(feat, max(dt_ms, 1.0))
        self.last_ts = ts_ms
        self.timer_feature_build_s += time.perf_counter() - t0
        return ts_ms, feat_z, mid, is_trade, dt_ms

    def on_fast_event(self, e: Any):
        t0 = time.perf_counter()
        etype, ts_ms, payload = self._parse_event(e)
        self.timer_parse_dispatch_s += time.perf_counter() - t0
        return self._dispatch_parsed_event(etype, ts_ms, payload)

    def on_event(self, e: Any):
        return self.on_fast_event(e)


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

    for field in ("feature_dim_total", "aux_dim", "lookback"):
        if field not in meta:
            raise ValueError(f"{source} missing required field '{field}'.")
        try:
            value = int(meta[field])
        except (TypeError, ValueError):
            raise ValueError(f"{source} has non-integer {field}={meta[field]!r}.")
        if value <= 0:
            raise ValueError(f"{source} has non-positive {field}={value}.")


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
    if metric == PRIMARY_METRIC:
        return "max"
    raise ValueError(f"Unsupported primary metric '{metric}'")

def compute_primary_metric(metric_payload: Dict[str, Any]) -> Tuple[float, str]:
    if PRIMARY_METRIC_HORIZON_MS not in HORIZONS_MS:
        raise ValueError(
            f"PRIMARY_METRIC_HORIZON_MS={PRIMARY_METRIC_HORIZON_MS} not in HORIZONS_MS={HORIZONS_MS}"
        )
    idx = HORIZONS_MS.index(PRIMARY_METRIC_HORIZON_MS)
    vals = metric_payload.get("spearman_kept_q50plus", [])
    if idx >= len(vals):
        return float("nan"), PRIMARY_METRIC
    return float(vals[idx]), PRIMARY_METRIC

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
