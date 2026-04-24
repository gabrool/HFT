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
CHECKPOINT_SCHEMA = "cmssl17-signed-raw-v2"
EPOCHS          = 200
LR              = 4e-4
CLIP_GRAD       = 10000
PATIENCE        = 15
# Primary metric config (used for checkpointing + early stopping)
PRIMARY_METRIC = "spearman_kept_q50plus_30000ms"
PRIMARY_METRIC_HORIZON_MS = 30_000
SINGLE_WEEK_PATIENCE = 3
# Number of auxiliary channels appended after the base feature vector
# These correspond to [log_dt_ms, is_trade, log_events_100ms, log_events_500ms,
#  log_events_1000ms, log_events_3000ms, log_events_7500ms]
AUX_DIM        = 7


FAST_WINDOWS_MS = (500, 1_000, 3_000, 7_500, 15_000)
FLOW_WINDOWS_MS = (1_000, 3_000, 7_500, 15_000, 30_000)
REGIME_WINDOWS_MS = (3_000, 7_500, 15_000, 30_000, 60_000)
EVENT_DENSITY_WINDOWS_MS = (100, 500, 1_000, 3_000, 7_500)
EMA_HALF_LIVES_MS = (1_000, 3_000, 7_500, 15_000, 30_000)
MACD_TRIPLETS_MS = (
    (1_000, 3_000, 1_500),
    (3_000, 7_500, 5_000),
    (7_500, 15_000, 10_000),
    (15_000, 30_000, 20_000),
    (30_000, 60_000, 40_000),
)
VPIN_BUCKET_SECS = (1.0, 3.0, 7.5, 15.0, 30.0)
BOOK_DEPTH_FEATURE_LEVELS = (1, 2, 3, 5, 7, 10, 15, 20, 30, 50, 100)
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
            self.DW_infer = nn.Conv1d(d_model, d_model, kernel_size, stride=1, padding='same', groups=d_model)
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

    def _get_merged_param(self):
        left_pad = (self.large_ks - self.small_ks) // 2
        right_pad = (self.large_ks - self.small_ks) - left_pad
        module_output = copy.deepcopy(self.DW_conv_large)
        module_output.weight = torch.nn.Parameter(module_output.weight + F.pad(self.DW_conv_small.weight, (left_pad, right_pad), value=0))
        module_output.bias = torch.nn.Parameter(module_output.bias + self.DW_conv_small.bias)
        self.DW_infer = module_output

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
        z_hl_ms: int = 500,
        vpin_target_bucket_secs: float = 2.0,
    ):
        self.depth = int(depth)
        if self.depth < MAX_BOOK_FEATURE_LEVEL:
            raise ValueError(
                f"FeatureEngine depth={self.depth} is too small for BOOK_DEPTH_FEATURE_LEVELS={BOOK_DEPTH_FEATURE_LEVELS}. "
                f"Need depth >= {MAX_BOOK_FEATURE_LEVEL}."
            )
        self.z_hl_ms = int(z_hl_ms)
        self.vpin_target_bucket_secs = float(vpin_target_bucket_secs)
        self.vpin_bucket_secs: Tuple[float, ...] = VPIN_BUCKET_SECS

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
        self.prev_cum_bid_by_level: Dict[int, float] = {lvl: 0.0 for lvl in BOOK_DEPTH_FEATURE_LEVELS}
        self.prev_cum_ask_by_level: Dict[int, float] = {lvl: 0.0 for lvl in BOOK_DEPTH_FEATURE_LEVELS}

        # Guard against sub-interval jitter when targeting ~100ms OB cadence
        self.ob_jitter_guard_ms: int = 100

        # ---------- Time bookkeeping ----------
        self.last_ts: Optional[int] = None
        self._last_event_ts: Optional[int] = None
        self.last_trade_ts: Optional[int] = None
        self.last_bid1_update_ts: Optional[int] = None
        self.last_ask1_update_ts: Optional[int] = None

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
        self._regime_return_deques: Dict[int, RollingWindowStats] = {
            ms: RollingWindowStats(ms) for ms in self.regime_windows_ms
        }
        self.last_mid_for_ret: Optional[float] = None

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
        self._bid1_change_deques: Dict[int, Deque[int]] = {ms: deque() for ms in self.bestlvl_windows}
        self._ask1_change_deques: Dict[int, Deque[int]] = {ms: deque() for ms in self.bestlvl_windows}
        self.last_bid1 = None; self.last_ask1 = None
        self.sz_delta_deques: Dict[int, Deque[Tuple[int, float, float]]] = {
            ms: deque() for ms in self.bestlvl_windows
        }
        self.sz_delta_sums: Dict[int, Dict[str, float]] = {
            ms: {"bid": 0.0, "ask": 0.0} for ms in self.bestlvl_windows
        }

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
        # (ts, price, size, side, tick_sign, is_zero_tick)
        self.trade_windows: Tuple[int, ...] = FLOW_WINDOWS_MS
        self._trade_window_deques: Dict[int, Deque[Tuple[int, float, float, str, int, int]]] = {
            ms: deque() for ms in self.trade_windows
        }
        self.trade_window_state: Dict[int, Dict[str, Any]] = {
            window: self._new_trade_window_state()
            for window in self.trade_windows
        }

        # Tick-direction & RPI tracking
        self.last_tick_sign: int = 0
        self.last_is_zero_tick: int = 0
        self.last_trade_price: Optional[float] = None
        self.last_is_rpi: int = 0

        # ---------- Quote windows ----------
        self._quote_window_deques: Dict[int, Deque[int]] = {
            ms: deque() for ms in FLOW_WINDOWS_MS
        }

        # ---------- Event density windows ----------
        self._event_density_deques: Dict[int, Deque[int]] = {
            ms: deque() for ms in EVENT_DENSITY_WINDOWS_MS
        }

        # ---------- VPIN state ----------
        self.vpin_state: Dict[float, Dict[str, Any]] = {
            secs: {"Vb": None, "cum_buy": 0.0, "cum_sell": 0.0, "cum": 0.0, "phi": deque(maxlen=50)}
            for secs in self.vpin_bucket_secs
        }

        # ---------- Decayed pressure (EWMA of OFI L1) ----------
        self.pressure_by_window: Dict[int, float] = {ms: 0.0 for ms in FAST_WINDOWS_MS}

        # ---------- RSI/MACD/CCI state ----------
        self.rsi_state: Dict[int, Dict[str, float]] = {
            hl: {"gain": 0.0, "loss": 0.0} for hl in EMA_HALF_LIVES_MS
        }
        self.macd_state: Dict[int, Dict[str, Optional[float]]] = {
            idx: {"fast": None, "slow": None, "signal": None}
            for idx in range(len(MACD_TRIPLETS_MS))
        }
        self.cci_state: Dict[int, Dict[str, Optional[float]]] = {
            hl: {"mean": None, "mad": None} for hl in EMA_HALF_LIVES_MS
        }

        # ---------- Fast EMA state for microstructure signals ----------
        self.ema_half_lives_ms = EMA_HALF_LIVES_MS
        self.ema_indicator_names = (
            "bid1",
            "ask1",
            "mid",
            "spread_bps",
            "micro",
            "smart",
            "gap_a_bps",
            "gap_b_bps",
            "cum_bid1",
            "cum_ask1",
            "cum_bid3",
            "cum_ask3",
            "slope_a",
            "slope_b",
            "obi_l1",
            "obi_l3",
            "obi_l5",
            "ofi_l1",
            "ofi_l3",
            "ofi_l5",
            "micro_premia",
            "smart_premia",
        )
        self.ema_states: Dict[int, Dict[str, Optional[float]]] = {
            hl: {name: None for name in self.ema_indicator_names}
            for hl in self.ema_half_lives_ms
        }

        # ---------- Rolling z-score state (per-feature EWMA mean/var) ----------
        self.z_mean: Optional[np.ndarray] = None
        self.z_m2: Optional[np.ndarray] = None  # EWMA of x^2 (for var = m2 - mean^2)
        self._feat_dim: Optional[int] = None

        # ---------- ingest timing ----------
        self.timer_parse_dispatch_s: float = 0.0
        self.timer_book_update_s: float = 0.0
        self.timer_trade_update_s: float = 0.0
        self.timer_feature_build_s: float = 0.0

    def feature_dim(self) -> int:
        """Return feature dimension including Filiary channels."""
        if self._feat_dim is None:
            raise ValueError("Feature dimension unknown before first event")
        return self._feat_dim + AUX_DIM

    def _new_trade_window_state(self) -> Dict[str, Any]:
        return {
            "buy_cnt": 0,
            "sell_cnt": 0,
            "buy_vol": 0.0,
            "sell_vol": 0.0,
            "signed_px_sum": 0.0,
            "signed_cnt": 0.0,
            "pxv_sum": 0.0,
            "vol_sum": 0.0,
            "buy_max_q": deque(),
            "sell_max_q": deque(),
        }

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

    def _update_indicator_emas(self, signals: Dict[str, float], dt_ms: float) -> None:
        for hl in self.ema_half_lives_ms:
            state = self.ema_states[hl]
            for name in self.ema_indicator_names:
                value = signals[name]
                prev = state[name]
                if prev is None:
                    state[name] = value
                else:
                    state[name] = self._ewma_update(prev, value, dt_ms, hl)

    def _update_trade_window_state_with_insert(
        self,
        window_ms: int,
        entry: Tuple[int, float, float, str, int, int],
    ) -> None:
        ts_ms, price, size, side, *_ = entry
        state = self.trade_window_state[window_ms]

        state["pxv_sum"] += price * size
        state["vol_sum"] += size

        if side == "buy":
            state["buy_cnt"] += 1
            state["buy_vol"] += size
            state["signed_px_sum"] += price
            state["signed_cnt"] += 1.0
            q = state["buy_max_q"]
            while q and q[-1][1] <= size:
                q.pop()
            q.append((ts_ms, size))
        else:
            state["sell_cnt"] += 1
            state["sell_vol"] += size
            state["signed_px_sum"] -= price
            state["signed_cnt"] -= 1.0
            q = state["sell_max_q"]
            while q and q[-1][1] <= size:
                q.pop()
            q.append((ts_ms, size))

    def _update_trade_window_state_with_expire(
        self,
        window_ms: int,
        entry: Tuple[int, float, float, str, int, int],
    ) -> None:
        ts_ms, price, size, side, *_ = entry
        state = self.trade_window_state[window_ms]

        state["pxv_sum"] -= price * size
        state["vol_sum"] -= size

        if side == "buy":
            state["buy_cnt"] -= 1
            state["buy_vol"] -= size
            state["signed_px_sum"] -= price
            state["signed_cnt"] -= 1.0
            q = state["buy_max_q"]
            if q and q[0][0] == ts_ms and abs(q[0][1] - size) <= 1e-12:
                q.popleft()
        else:
            state["sell_cnt"] -= 1
            state["sell_vol"] -= size
            state["signed_px_sum"] += price
            state["signed_cnt"] += 1.0
            q = state["sell_max_q"]
            if q and q[0][0] == ts_ms and abs(q[0][1] - size) <= 1e-12:
                q.popleft()

        state["buy_cnt"] = max(0, state["buy_cnt"])
        state["sell_cnt"] = max(0, state["sell_cnt"])
        state["buy_vol"] = max(0.0, state["buy_vol"])
        state["sell_vol"] = max(0.0, state["sell_vol"])
        state["vol_sum"] = max(0.0, state["vol_sum"])
        if state["buy_cnt"] == 0 and state["sell_cnt"] == 0:
            state["signed_px_sum"] = 0.0
            state["signed_cnt"] = 0.0

    def _prune_trade_window(self, now_ms: int, window_ms: int) -> None:
        deq = self._trade_window_deques[window_ms]
        while deq and (now_ms - deq[0][0] > window_ms):
            expired = deq.popleft()
            self._update_trade_window_state_with_expire(window_ms, expired)

    def _compute_trade_window_stats(self, window_ms: int, mid: float) -> Dict[str, float]:
        state = self.trade_window_state[window_ms]
        buy_vol = float(state["buy_vol"])
        sell_vol = float(state["sell_vol"])
        buy_cnt = float(state["buy_cnt"])
        sell_cnt = float(state["sell_cnt"])
        buy_max = float(state["buy_max_q"][0][1]) if state["buy_max_q"] else 0.0
        sell_max = float(state["sell_max_q"][0][1]) if state["sell_max_q"] else 0.0

        buy_mean = buy_vol / buy_cnt if buy_cnt > 0 else 0.0
        sell_mean = sell_vol / sell_cnt if sell_cnt > 0 else 0.0
        net_flow = buy_vol - sell_vol
        total_vol = buy_vol + sell_vol
        denom = max(total_vol, 1e-12)
        imbalance = net_flow / denom
        toxicity = abs(net_flow) / denom

        trade_through = 0.0
        if mid > 0.0:
            trade_through = (float(state["signed_px_sum"]) / mid) - float(state["signed_cnt"])

        return {
            "buy_vol": buy_vol,
            "sell_vol": sell_vol,
            "buy_cnt": buy_cnt,
            "sell_cnt": sell_cnt,
            "buy_mean": buy_mean,
            "sell_mean": sell_mean,
            "buy_max": buy_max,
            "sell_max": sell_max,
            "net_flow": net_flow,
            "imbalance": imbalance,
            "toxicity": toxicity,
            "trade_through": trade_through,
        }

    def _lin_slope(self, xs: List[float], ys: List[float], eps: float = 1e-12) -> float:
        n = len(xs)
        if n < 2:
            return 0.0
        mx = sum(xs) / n
        my = sum(ys) / n
        num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
        den = sum((x - mx) * (x - mx) for x in xs) + eps
        return num / den

    def _sorted_ladders(self):
        # slow-path full rebuild, retained for snapshot resets / invalidated cached ladders
        self.bid_lvls = sorted(self.bids.items(), key=lambda x: x[0], reverse=True)[: self.depth]
        self.ask_lvls = sorted(self.asks.items(), key=lambda x: x[0], reverse=False)[: self.depth]
        self._book_dirty = False

    def _insert_level(self, levels: List[Tuple[float, float]], price: float, size: float, is_bid: bool) -> bool:
        insert_at = 0
        while insert_at < len(levels):
            px_i = levels[insert_at][0]
            if px_i == price:
                levels[insert_at] = (price, size)
                return insert_at < self.depth
            if (price > px_i) if is_bid else (price < px_i):
                break
            insert_at += 1
        levels.insert(insert_at, (price, size))
        if len(levels) > self.depth:
            levels.pop()
        return insert_at < self.depth

    def _remove_level(self, levels: List[Tuple[float, float]], price: float) -> bool:
        for idx, (px_i, _sz_i) in enumerate(levels):
            if px_i == price:
                levels.pop(idx)
                return idx < self.depth
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
        boundary_price = levels[-1][0] if len(levels) >= self.depth else None
        deleting = size <= 0.0

        if deleting and prev_size is None:
            return False
        if deleting and best_price is not None and price == best_price:
            return True
        if deleting:
            removed = self._remove_level(levels, price)
            if removed and len(book) >= self.depth:
                return True
            return False

        book[price] = size
        if was_tracked:
            self._insert_level(levels, price, size, is_bid)
            return False
        if len(levels) < self.depth:
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
            price = float(price_raw)
            size = float(size_raw)
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

    def _cum_depth(self, lvls: List[Tuple[float, float]], n: int) -> float:
        return float(sum(s for _, s in lvls[: n]))

    def _levels_to_xy(self, levels: List[Tuple[float, float]], mid: float, is_bid: bool, K: int) -> Tuple[List[float], List[float]]:
        xs, ys = [], []
        cum = 0.0
        for p, s in levels[:K]:
            if s <= 0.0:
                continue
            cum += s
            xs.append(cum)
            ys.append((mid - p) if is_bid else (p - mid))
        return xs, ys

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

    def event_density_500ms(self) -> float:
        return self.event_density(500)

    def event_density_1000ms(self) -> float:
        return self.event_density(1_000)

    def event_density_3000ms(self) -> float:
        return self.event_density(3_000)

    def event_density_7500ms(self) -> float:
        return self.event_density(7_500)

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
    def _parse_event(self, e: Any) -> Tuple[str, int, dict]:
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
    ) -> Tuple[int, np.ndarray, float, bool, float]:
        dt_ms = 1.0 if self._last_event_ts is None else max(1.0, ts_ms - self._last_event_ts)
        prev_bid_l1 = self.prev_bsz
        prev_ask_l1 = self.prev_asz
        prev_bid_l2 = self.prev_bsz2
        prev_ask_l2 = self.prev_asz2

        self._prune_replen_windows(ts_ms)
        for window in self._trade_window_deques:
            self._prune_trade_window(ts_ms, window)

        is_trade = (etype == 'trade')
        for w, deq in self._event_density_deques.items():
            self._append_ts_with_guard(deq, ts_ms, w, is_ob_event=(etype == 'ob'))

        if etype == 'ob':
            tp_code, bids, asks = payload
            t0 = time.perf_counter()
            self._update_book_from_ob(tp_code, bids, asks)
            self.timer_book_update_s += time.perf_counter() - t0
            for window, deq in self._quote_window_deques.items():
                self._append_ts_with_guard(deq, ts_ms, window, is_ob_event=True)
        else:
            t0 = time.perf_counter()
            self._update_trade_windows(ts_ms, payload, dt_ms)
            self.timer_trade_update_s += time.perf_counter() - t0

        t0 = time.perf_counter()
        self._ensure_book_ladders()
        bid1, ask1, bsz1, asz1 = self._book_best()
        mid = 0.5 * (bid1 + ask1) if (bid1 > 0 and ask1 > 0) else 0.0

        if (bsz1 + asz1) > 0:
            micro = (ask1 * bsz1 + bid1 * asz1) / (bsz1 + asz1)
            wb = 1.0 / max(bsz1, 1e-12)
            wa = 1.0 / max(asz1, 1e-12)
            smart = (bid1 * wb + ask1 * wa) / (wb + wa)
        else:
            micro = smart = mid

        spread = max(0.0, ask1 - bid1)
        spread_bps = 1e4 * spread / max(mid, 1e-12)

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

        cum_bid_by_level = {lvl: self._cum_depth(self.bid_lvls, lvl) for lvl in BOOK_DEPTH_FEATURE_LEVELS}
        cum_ask_by_level = {lvl: self._cum_depth(self.ask_lvls, lvl) for lvl in BOOK_DEPTH_FEATURE_LEVELS}
        cum_bid1 = cum_bid_by_level[1]
        cum_ask1 = cum_ask_by_level[1]
        cum_bid3 = cum_bid_by_level[3]
        cum_ask3 = cum_ask_by_level[3]
        cum_bid5 = cum_bid_by_level[5]
        cum_ask5 = cum_ask_by_level[5]
        cum_bid10 = cum_bid_by_level[10]
        cum_ask10 = cum_ask_by_level[10]

        if etype == 'ob':
            self._append_ob_snapshot(ts_ms, bid1, ask1, bsz1, asz1, spread, cum_bid3, cum_ask3, cum_bid5, cum_ask5)

        obi_by_level = {
            lvl: (cum_bid_by_level[lvl] - cum_ask_by_level[lvl]) / max(cum_bid_by_level[lvl] + cum_ask_by_level[lvl], 1e-12)
            for lvl in BOOK_DEPTH_FEATURE_LEVELS
        }

        ofi_by_level: Dict[int, float] = {}
        ofi_by_level[1] = (bsz1 - prev_bid_l1) - (asz1 - prev_ask_l1)
        for lvl in BOOK_DEPTH_FEATURE_LEVELS:
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
        for lvl in BOOK_DEPTH_FEATURE_LEVELS:
            self.prev_cum_bid_by_level[lvl] = cum_bid_by_level[lvl]
            self.prev_cum_ask_by_level[lvl] = cum_ask_by_level[lvl]

        obi_l1 = obi_by_level[1]
        obi_l3 = obi_by_level[3]
        obi_l5 = obi_by_level[5]

        micro_premia = (micro - mid) / max(spread, 1e-9)
        smart_premia = (smart - mid) / max(spread, 1e-9)

        xb, yb = self._levels_to_xy(self.bid_lvls, mid, True, 5)
        xa, ya = self._levels_to_xy(self.ask_lvls, mid, False, 5)
        slope_b = self._lin_slope(xb, yb)
        slope_a = self._lin_slope(xa, ya)

        indicator_values = {
            "bid1": bid1, "ask1": ask1, "mid": mid, "spread_bps": spread_bps,
            "micro": micro, "smart": smart, "gap_a_bps": gap_a_bps, "gap_b_bps": gap_b_bps,
            "cum_bid1": cum_bid1, "cum_ask1": cum_ask1, "cum_bid3": cum_bid3, "cum_ask3": cum_ask3,
            "slope_a": slope_a, "slope_b": slope_b, "obi_l1": obi_l1, "obi_l3": obi_l3, "obi_l5": obi_l5,
            "ofi_l1": ofi_l1, "ofi_l3": ofi_l3, "ofi_l5": ofi_l5, "micro_premia": micro_premia, "smart_premia": smart_premia,
        }
        self._update_indicator_emas(indicator_values, dt_ms)

        for ms in FAST_WINDOWS_MS:
            self.pressure_by_window[ms] = self._ewma_update(self.pressure_by_window[ms], ofi_l1, dt_ms, ms)

        trade_stats = {ms: self._compute_trade_window_stats(ms, mid) for ms in self.trade_windows}
        quote_counts = {ms: len(self._quote_window_deques[ms]) for ms in self.trade_windows}
        vwap_per_ms = {
            ms: (self.trade_window_state[ms]["pxv_sum"] / self.trade_window_state[ms]["vol_sum"] if self.trade_window_state[ms]["vol_sum"] > 1e-12 else mid)
            for ms in self.trade_windows
        }
        vwap_vs_mid_bps = {ms: (1e4 * ((vwap_per_ms[ms] / max(mid, 1e-12)) - 1.0)) if mid > 0 else 0.0 for ms in self.trade_windows}
        vwap_vs_micro_bps = {ms: (1e4 * ((vwap_per_ms[ms] / max(micro, 1e-12)) - 1.0)) if micro > 0 else 0.0 for ms in self.trade_windows}

        for ms in self._spread_change_deques:
            if self.last_spread is None or spread != self.last_spread:
                self._append_ts_with_guard(self._spread_change_deques[ms], ts_ms, ms, is_ob_event=True)
            else:
                self._prune_ts_deque(self._spread_change_deques[ms], ts_ms, ms)
        if self.last_spread is None or spread != self.last_spread:
            self.last_spread = spread
            self.last_spread_ts = ts_ms

        bid_level_changed = (self.last_bid1 is None or bid1 != self.last_bid1 or bsz1 != prev_bid_l1)
        ask_level_changed = (self.last_ask1 is None or ask1 != self.last_ask1 or asz1 != prev_ask_l1)
        for ms, dq in self._bid1_change_deques.items():
            self._append_ts_with_guard(dq, ts_ms, ms, is_ob_event=True) if bid_level_changed else self._prune_ts_deque(dq, ts_ms, ms)
        for ms, dq in self._ask1_change_deques.items():
            self._append_ts_with_guard(dq, ts_ms, ms, is_ob_event=True) if ask_level_changed else self._prune_ts_deque(dq, ts_ms, ms)
        if bid_level_changed:
            self.last_bid1_update_ts = ts_ms
        if ask_level_changed:
            self.last_ask1_update_ts = ts_ms
        self.last_bid1, self.last_ask1 = bid1, ask1

        for ms, deq in self.sz_delta_deques.items():
            bid_dep = min(bsz1 - prev_bid_l1, 0.0)
            ask_dep = min(asz1 - prev_ask_l1, 0.0)
            deq.append((ts_ms, bid_dep, ask_dep))
            sums = self.sz_delta_sums[ms]
            sums["bid"] += bid_dep
            sums["ask"] += ask_dep
            while deq and (ts_ms - deq[0][0] > ms):
                _, old_bid, old_ask = deq.popleft()
                sums["bid"] -= old_bid
                sums["ask"] -= old_ask
        neg_depletion = {
            ms: (self.sz_delta_sums[ms]["bid"], self.sz_delta_sums[ms]["ask"])
            for ms in self.sz_delta_deques
        }

        dt_since_trade = float(ts_ms - self.last_trade_ts) if self.last_trade_ts is not None else 0.0
        dt_since_bid1_update = float(ts_ms - self.last_bid1_update_ts) if self.last_bid1_update_ts is not None else 0.0
        dt_since_ask1_update = float(ts_ms - self.last_ask1_update_ts) if self.last_ask1_update_ts is not None else 0.0

        self._add_return(ts_ms, mid, is_ob_event=(etype == 'ob'))
        return_var = {ms: stats.mean_var()[1] for ms, stats in self.return_histories.items()}
        return_std = {ms: math.sqrt(var) for ms, var in return_var.items()}
        vr_adjacent = {}
        for prev_ms, cur_ms in zip(self.return_windows_ms[:-1], self.return_windows_ms[1:]):
            var_prev = return_var[prev_ms]
            var_cur = return_var[cur_ms]
            vr_adjacent[(cur_ms, prev_ms)] = (var_cur / max((cur_ms / prev_ms) * var_prev, 1e-12)) if var_prev > 0 else 0.0

        regime_vol_ewma = {ms: math.sqrt(max(self.rv_ewma[ms], 1e-18)) for ms in self.regime_windows_ms}
        regime_realized = {ms: self.realized_vol[ms] for ms in self.regime_windows_ms}
        regime_volume = {ms: self.volume_ewma[ms] for ms in self.regime_windows_ms}
        for ms in self.regime_windows_ms:
            nearest = min(self.trade_windows, key=lambda w: abs(w - ms))
            self.flow_regime[ms] = trade_stats[nearest]["imbalance"]
        regime_flow_snapshot = {ms: self.flow_regime[ms] for ms in self.regime_windows_ms}

        rsi_vals = []
        for hl in EMA_HALF_LIVES_MS:
            ema_micro = self.ema_states[hl]["micro"] if self.ema_states[hl]["micro"] is not None else micro
            delta = micro - ema_micro
            state = self.rsi_state[hl]
            state["gain"] = self._ewma_update(state["gain"], max(delta, 0.0), dt_ms, hl)
            state["loss"] = self._ewma_update(state["loss"], max(-delta, 0.0), dt_ms, hl)
            rs = state["gain"] / max(state["loss"], 1e-12)
            rsi_vals.append(100.0 - 100.0 / (1.0 + rs))

        def ema_ms(prev: Optional[float], x: float, hl_ms: float) -> float:
            a = 1.0 - math.exp(-dt_ms / max(1.0, hl_ms))
            return x if prev is None else ((1.0 - a) * prev + a * x)

        macd_features = []
        for idx, (fast_ms, slow_ms, sig_ms) in enumerate(MACD_TRIPLETS_MS):
            st = self.macd_state[idx]
            st["fast"] = ema_ms(st["fast"], micro, float(fast_ms))
            st["slow"] = ema_ms(st["slow"], micro, float(slow_ms))
            raw = float(st["fast"] - st["slow"])
            st["signal"] = ema_ms(st["signal"], raw, float(sig_ms))
            sig = float(st["signal"])
            macd_features.extend([raw, sig, raw - sig])

        cci_features = []
        for hl in EMA_HALF_LIVES_MS:
            st = self.cci_state[hl]
            st["mean"] = ema_ms(st["mean"], micro, float(hl))
            mad = abs(micro - float(st["mean"]))
            st["mad"] = ema_ms(st["mad"], mad, float(hl))
            cci_features.append(0.015 * ((micro - float(st["mean"])) / max(float(st["mad"]), 1e-12)))

        vpin_features = []
        for secs in self.vpin_bucket_secs:
            phi = self.vpin_state[secs]["phi"]
            vpin_features.append((sum(phi) / len(phi)) if phi else 0.0)

        replen_rates = self._replenishment_rates()

        # Canonical order:
        # 1) instantaneous; 2) fast-window OB/spread/churn/depletion/replen; 3) flow-window trade/quote/VWAP;
        # 4) flow-window return std/VR; 5) regime-window summaries; 6) pressure; 7) EMA bank; 8) EMA residuals;
        # 9) RSI; 10) MACD; 11) CCI; 12) VPIN.
        feat_list = [
            bid1, ask1, mid, micro, smart, spread_bps, gap_a_bps, gap_b_bps,
            bsz1, asz1,
        ]
        for lvl in BOOK_DEPTH_FEATURE_LEVELS:
            feat_list.extend([cum_bid_by_level[lvl], cum_ask_by_level[lvl]])
        for lvl in BOOK_DEPTH_FEATURE_LEVELS:
            feat_list.append(obi_by_level[lvl])
        for lvl in BOOK_DEPTH_FEATURE_LEVELS:
            feat_list.append(ofi_by_level[lvl])
        feat_list.extend([
            slope_a, slope_b,
            micro_premia, smart_premia,
            dt_since_trade, dt_since_bid1_update, dt_since_ask1_update,
            float(self.last_tick_sign), float(self.last_is_zero_tick), float(self.last_is_rpi),
        ])

        for ms in FAST_WINDOWS_MS:
            feat_list.extend([
                spread_delta_bps.get(ms, 0.0),
                float(len(self._spread_change_deques[ms])),
                float(len(self._bid1_change_deques[ms])),
                float(len(self._ask1_change_deques[ms])),
                neg_depletion[ms][0], neg_depletion[ms][1],
            ])
            rates = replen_rates[ms]
            for level in (1, 2):
                feat_list.extend([rates[("bid", level, "add")], rates[("bid", level, "rem")], rates[("ask", level, "add")], rates[("ask", level, "rem")]])

        for ms in FLOW_WINDOWS_MS:
            stats = trade_stats[ms]
            feat_list.extend([
                stats["buy_vol"], stats["sell_vol"], stats["buy_cnt"], stats["sell_cnt"],
                stats["buy_mean"], stats["sell_mean"], stats["buy_max"], stats["sell_max"],
                stats["net_flow"], stats["imbalance"], stats["toxicity"], stats["trade_through"],
                float(quote_counts[ms]), vwap_vs_mid_bps[ms], vwap_vs_micro_bps[ms],
            ])

        for ms in FLOW_WINDOWS_MS:
            feat_list.append(return_std[ms])
        for prev_ms, cur_ms in zip(FLOW_WINDOWS_MS[:-1], FLOW_WINDOWS_MS[1:]):
            feat_list.append(vr_adjacent[(cur_ms, prev_ms)])

        for ms in REGIME_WINDOWS_MS:
            feat_list.extend([regime_volume[ms], regime_realized[ms], regime_vol_ewma[ms], regime_flow_snapshot[ms]])

        for ms in FAST_WINDOWS_MS:
            feat_list.append(self.pressure_by_window[ms])

        for hl in self.ema_half_lives_ms:
            state = self.ema_states[hl]
            for name in self.ema_indicator_names:
                feat_list.append(state[name] if state[name] is not None else indicator_values[name])
        for hl in self.ema_half_lives_ms:
            state = self.ema_states[hl]
            for name in self.ema_indicator_names:
                ema_val = state[name] if state[name] is not None else indicator_values[name]
                feat_list.append(indicator_values[name] - ema_val)

        feat_list.extend(rsi_vals)
        feat_list.extend(macd_features)
        feat_list.extend(cci_features)
        feat_list.extend(vpin_features)

        feat = np.array(feat_list, dtype=np.float64)
        feat_z = self._zscore(feat, dt_ms)
        self.last_ts = ts_ms
        self._last_event_ts = ts_ms

        self.timer_feature_build_s += time.perf_counter() - t0
        return ts_ms, feat_z, mid, is_trade, dt_ms

    def _update_book_from_ob(
        self,
        tp_code: int,
        bids: Sequence[Tuple[float, float]],
        asks: Sequence[Tuple[float, float]],
    ) -> None:
        if int(tp_code) == 1:
            self.bids = {float(p): float(q) for p, q in bids[: self.depth]}
            self.asks = {float(p): float(q) for p, q in asks[: self.depth]}
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
            price = float(price)
            size = float(size)
            tick_sign = int(tick_dir_code)
            is_zero_tick = 1 if int(tick_dir_code) == 0 else 0
            is_rpi = int(is_rpi)
        else:
            side = str(trade_evt['side']).lower()  # 'buy'|'sell'
            price = float(trade_evt['price'])
            size = float(trade_evt['size'])

            tick_dir = trade_evt.get("tickDirection")
            tick_sign = int(self.last_tick_sign)
            is_zero_tick = int(self.last_is_zero_tick)
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
                tick_sign, is_zero_tick = self._interpret_tick_direction(tick_dir)

        if self.last_trade_price is not None:
            if price > self.last_trade_price:
                tick_sign, is_zero_tick = 1, 0
            elif price < self.last_trade_price:
                tick_sign, is_zero_tick = -1, 0
            else:
                if tick_sign == 0 and is_zero_tick == 0:
                    tick_sign = self.last_tick_sign if self.last_tick_sign != 0 else 0
                if is_zero_tick == 0:
                    is_zero_tick = 1
        else:
            if tick_sign == 0 and is_zero_tick == 0:
                tick_sign, is_zero_tick = 0, 0

        self.last_tick_sign = tick_sign
        self.last_is_zero_tick = is_zero_tick
        self.last_trade_price = price
        self.last_is_rpi = is_rpi

        entry = (ts_ms, price, size, side, tick_sign, is_zero_tick)
        for window, deq in self._trade_window_deques.items():
            deq.append(entry)
            self._update_trade_window_state_with_insert(window, entry)

        dt_trade_ms = (
            max(1.0, float(ts_ms - self.last_trade_ts))
            if self.last_trade_ts is not None
            else max(1.0, float(dt_ms))
        )

        # Update volume-regime (vol/sec) EWMAs using trade-arrival timing
        vol_rate = size / (dt_trade_ms / 1000.0)  # base per second
        for hl in self.regime_windows_ms:
            self.volume_ewma[hl] = self._ewma_update(self.volume_ewma[hl], vol_rate, dt_trade_ms, hl)

        # VPIN bucket sizing and accumulation per configured bucket scale
        v_per_sec = max(self.volume_ewma[3_000], 1e-9)
        for secs, st in self.vpin_state.items():
            Vb = max(v_per_sec * float(secs), 1e-9)
            st["Vb"] = Vb if st["Vb"] is None else (0.9 * st["Vb"] + 0.1 * Vb)
            if side == 'buy':
                st["cum_buy"] += size
            else:
                st["cum_sell"] += size
            st["cum"] += size

            while st["cum"] >= (st["Vb"] or 1e9):
                if st["Vb"] is None:
                    break
                total = max(st["cum"], 1e-12)
                scale = st["Vb"] / total
                buy_bucket = st["cum_buy"] * scale
                sell_bucket = st["cum_sell"] * scale
                phi = abs(buy_bucket - sell_bucket) / max(st["Vb"], 1e-12)
                st["phi"].append(phi)
                st["cum_buy"] -= buy_bucket
                st["cum_sell"] -= sell_bucket
                st["cum"] -= st["Vb"]

        self.last_trade_ts = ts_ms

    def _add_return(self, ts_ms: int, mid: float, is_ob_event: bool):
        if mid <= 0.0:
            return 0.0
    
        if self.last_mid_for_ret is None:
            self.last_mid_for_ret = mid
            return 0.0
    
        # Do not let trade-only rows inject zero mid returns into OB-return statistics.
        if not is_ob_event:
            return 0.0
            
        r = (1e4 * math.log(mid / self.last_mid_for_ret)) if self.last_mid_for_ret > 0 else 0.0
        self.last_mid_for_ret = mid

        for stats in self.return_histories.values():
            stats.add(ts_ms, r)

        dt_ms = 1.0 if self._last_event_ts is None else max(1.0, ts_ms - self._last_event_ts)
        r2 = r * r
        for hl in self.regime_windows_ms:
            self.rv_ewma[hl] = self._ewma_update(self.rv_ewma[hl], r2, dt_ms, hl)

        for ms, stats in self._regime_return_deques.items():
            stats.add(ts_ms, r)
            self.realized_vol[ms] = math.sqrt(max(0.0, stats.sumsq))
        return r

    def _zscore(self, x: np.ndarray, dt_ms: float) -> np.ndarray:
        """Per-feature EWMA mean/var rolling z-score.
    
        Keep running mean / second moment in float64 for numerical stability.
        Emit float32 z-scores to preserve the dataset/model contract.
        """
        eps = 1e-9
        x64 = x.astype(np.float64, copy=False)

        if self._feat_dim is None:
            self._feat_dim = int(x.shape[0])
            self.z_mean = x64.copy()
            self.z_m2 = np.square(x64, dtype=np.float64)
            return np.zeros_like(x, dtype=np.float32)

        hl = self._alpha_half_life_ms(self.z_hl_ms)
        alpha = float(1.0 - math.pow(0.5, max(1.0, dt_ms) / float(hl)))
        one_minus_alpha = 1.0 - alpha

        # Update EWMA mean and second moment in float64
        self.z_mean = one_minus_alpha * self.z_mean + alpha * x64
        self.z_m2 = one_minus_alpha * self.z_m2 + alpha * np.square(x64, dtype=np.float64)

        var = np.maximum(self.z_m2 - self.z_mean * self.z_mean, eps)
        z = (x64 - self.z_mean) / np.sqrt(var)

        return z.astype(np.float32, copy=False)

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------
    def on_fast_event(self, e: Any) -> Tuple[int, np.ndarray, float, bool, float]:
        """Fast ingest path for compact tuples emitted by offline_ingest.py."""
        t0 = time.perf_counter()
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
        self.timer_parse_dispatch_s += time.perf_counter() - t0
        return self._dispatch_parsed_event(etype, ts_ms, payload)

    def on_event(self, e: Any) -> Tuple[int, np.ndarray, float, bool, float]:
        """Slow compatibility path for callers that still pass generic event shapes."""
        t0 = time.perf_counter()
        etype, ts_ms, payload = self._parse_event(e)
        self.timer_parse_dispatch_s += time.perf_counter() - t0

        if etype == 'ob':
            if isinstance(payload, tuple):
                compact_payload = (int(payload[0]), payload[1], payload[2])
            else:
                data = payload.get('data', payload)
                compact_payload = (
                    1 if str(payload.get('type') or data.get('type') or payload.get('DataType') or 'delta').strip().lower() == 'snapshot' else 2,
                    tuple((float(p), float(q)) for p, q in data.get('b', [])),
                    tuple((float(p), float(q)) for p, q in data.get('a', [])),
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

    def timer_totals(self) -> Dict[str, float]:
        return {
            'parse_dispatch_s': float(self.timer_parse_dispatch_s),
            'order_book_update_s': float(self.timer_book_update_s),
            'trade_update_s': float(self.timer_trade_update_s),
            'feature_build_s': float(self.timer_feature_build_s),
        }

    def print_timer_totals(self, prefix: str = '[timers]') -> None:
        totals = self.timer_totals()
        print(
            f"{prefix} parse_dispatch={totals['parse_dispatch_s']:.6f}s "
            f"order_book_update={totals['order_book_update_s']:.6f}s "
            f"trade_update={totals['trade_update_s']:.6f}s "
            f"feature_build={totals['feature_build_s']:.6f}s",
            flush=True,
        )


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
        self.week_ids: List[int] = []
        self.row_idx: List[int] = []
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
                if not np.any(mask):
                    continue
                idx = np.nonzero(mask)[0]
                self.week_ids.extend([self.week_to_id[wk]] * int(idx.shape[0]))
                self.row_idx.extend(np.asarray(row_idx_arr[idx], dtype=np.int64).tolist())
                y_parts.append(np.asarray(y_arr[idx], dtype=np.float32))

        self.week_ids = np.asarray(self.week_ids, dtype=np.int16)
        self.row_idx = np.asarray(self.row_idx, dtype=np.int64)
        self.y = np.concatenate(y_parts, axis=0).astype(np.float32, copy=False) if y_parts else np.empty((0, NUM_HORIZONS), dtype=np.float32)

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
