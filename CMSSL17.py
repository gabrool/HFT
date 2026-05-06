import os, math, copy, json, csv, zipfile, io, gzip, contextlib, time, heapq
from pathlib import Path
from collections import deque
from bisect import bisect_left, bisect_right, insort
from decimal import Decimal, ROUND_HALF_EVEN, InvalidOperation
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from datetime import datetime, timezone
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
BATCH_SIZE      = 128
DMODEL          = 1024
MAMBA_LAYERS    = 2
CONV_KERNELS    = [3,3,5,5,7,7]
DFF_CONV        = 2 * DMODEL
DEPATCH_OFFSET_MODE = "learnable"

MODEL_ARCH_SCHEMA = "ctn_hybrid_ci4_gate_proj_mixed2_ci8_v1"
CTN_CI_KERNELS = [3, 3, 7, 7]
CTN_MIXED_KERNELS = [15, 15]
CTN_CI_INTERNAL_DIM = 8
CTN_CI_FF_MULT = 8
CTN_CI_DFF = CTN_CI_INTERNAL_DIM * CTN_CI_FF_MULT
CTN_POST_GATE_HIDDEN = 2 * DMODEL
CTN_MIXED_DIM = DMODEL
CTN_MIXED_DFF = 2 * DMODEL
CTN_PATCH_SIZE = 2
CTN_PATCH_STRIDE = 1

# Prediction horizons (in milliseconds)
HORIZONS_MS     = [7_500, 15_000, 30_000]
NUM_HORIZONS    = len(HORIZONS_MS)
HORIZON_WEIGHTS = [0.25, 0.5, 1.0]

LOW_ABS_TRIM_FRACTION = 0.05
HIGH_ABS_TRIM_FRACTION = 0.02
TARGET_TRANSFORM = "raw_signed_bps_to_direction_and_conditional_abs_sqrt_bps"
TARGET_TASK = "direction_and_conditional_magnitude_raw_bps_targets"
FEATURE_SCHEMA = "cmssl17_30s_taker_stage4_v6_fast_trade_obnorm"
AUX_SCHEMA = "cmssl17_aux_ob_decision_density_v3_no_1s"
CHECKPOINT_SCHEMA = "cmssl17-dir-mag-v1-stage4-v5-aux6-pca506"
EPOCHS          = 200
LR              = 4e-4
CLIP_GRAD       = 10000
PATIENCE        = 15
# Primary metric config (used for checkpointing + early stopping)
PRIMARY_METRIC = "edge_spearman_q50plus_30000ms"
PRIMARY_METRIC_HORIZON_MS = 30_000
PRIMARY_DIR_BAL_ACC_GUARD = 0.505
MODEL_OUTPUT_SCHEMA = "dir_logits_mag_up_down_sqrt_v1"
MAG_SQRT_EPS = 1e-6
DIR_LOSS_WEIGHT = 1.00
MAG_LOSS_WEIGHT = 0.75
MAG_CORR_LOSS_WEIGHT = 0.05
SINGLE_WEEK_PATIENCE = 1
# Number of auxiliary channels appended after the PCA/core feature vector.
# These correspond to:
# [log_dt_decision_ms, log_events_3000ms, log_events_7500ms,
#  log_events_15000ms, log_events_30000ms, log_events_60000ms]
AUX_DIM        = 6
FEATURE_AUX_TAIL = (
    "log_dt_decision_ms",
    "log_events_3000ms",
    "log_events_7500ms",
    "log_events_15000ms",
    "log_events_30000ms",
    "log_events_60000ms",
)


PRICE_WINDOWS_MS = (
    1_000,
    3_000,
    7_500,
    15_000,
    30_000,
    60_000,
)
NORMALIZED_OFI_LEVELS = (
    1,
    3,
    5,
    10,
    20,
    50,
)
BPS_DEPTH_BANDS = (
    0.5,
    1.0,
    2.0,
    3.0,
    5.0,
    7.5,
    10.0,
    15.0,
    25.0,
    50.0,
)
BOOK_SHAPE_BANDS = (
    1.0,
    2.0,
    5.0,
    10.0,
)
SLIPPAGE_NOTIONAL_USD = (
    10_000.0,
    25_000.0,
    50_000.0,
    100_000.0,
    250_000.0,
)
FAST_WINDOWS_MS = (1_000, 3_000, 7_500, 15_000, 30_000)
INTERACTION_WINDOWS_MS = (
    7_500,
    15_000,
    30_000,
)
FLOW_WINDOWS_MS = (
    1_000,
    3_000,
    7_500,
    15_000,
    30_000,
    60_000,
)
LARGE_TRADE_NOTIONAL_USD = (
    50_000.0,
    100_000.0,
    250_000.0,
    500_000.0,
)
LARGE_TRADE_CLUSTER_THRESHOLD_USD = 100_000.0
LARGE_TRADE_CLUSTER_GAP_MS = 1_000
LARGE_TRADE_CLOCK_THRESHOLD_USD = 100_000.0
ROLLING_OFI_LEVELS = (1, 3, 5, 10, 20)
ROLLING_OFI_WINDOWS_MS = (7_500, 15_000, 30_000, 60_000)
ROLLING_OBI_LEVELS = (3, 5, 10, 20)
ROLLING_OBI_WINDOWS_MS = (15_000, 30_000, 60_000)
DEEP_MICRO_LEVELS = (3, 5, 10, 20)
TRADE_BURST_WINDOWS_MS = (7_500, 15_000, 30_000)
LARGE_TRADE_CONTINUATION_WINDOWS_MS = (7_500, 15_000)
REGIME_WINDOWS_MS = (
    3_000,
    7_500,
    15_000,
    30_000,
    60_000,
    120_000,
)
EVENT_DENSITY_WINDOWS_MS = (1_000, 3_000, 7_500, 15_000, 30_000, 60_000)
EMA_HALF_LIVES_MS = (7_500, 15_000, 30_000, 60_000, 120_000)
MACD_TRIPLETS_MS = (
    (7_500, 15_000, 10_000),
    (15_000, 30_000, 20_000),
    (30_000, 60_000, 40_000),
)
VPIN_BUCKET_SECS = (7.5, 15.0, 30.0)
SPREAD_DEPTH_REGIME_WINDOWS_MS = (
    7_500,
    15_000,
    30_000,
    60_000,
    120_000,
)
BOOK_DEPTH_FEATURE_LEVELS = (1, 2, 3, 5, 7, 10, 15, 20, 30, 50, 100)
MAX_BOOK_FEATURE_LEVEL = max(BOOK_DEPTH_FEATURE_LEVELS)

NUM_HEADS       = 16
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
        return {
            "offset_dx_mean": float(dx.mean().cpu()),
            "offset_dx_std": float(dx.std(unbiased=False).cpu()),
            "offset_dx_abs_p95": float(torch.quantile(dx_abs, 0.95).cpu()),
            "offset_dx_abs_max": float(dx_abs.max().cpu()),
            "offset_ds_raw_mean": float(ds_raw.mean().cpu()),
            "offset_ds_raw_std": float(ds_raw.std(unbiased=False).cpu()),
            "offset_ds_raw_abs_p95": float(torch.quantile(ds_abs, 0.95).cpu()),
            "span_samples_mean": float(span_flat.mean().cpu()),
            "span_samples_p05": float(torch.quantile(span_flat, 0.05).cpu()),
            "span_samples_p50": float(torch.quantile(span_flat, 0.50).cpu()),
            "span_samples_p95": float(torch.quantile(span_flat, 0.95).cpu()),
            "bound_left_clip_frac": float((bf[..., 0] <= 1e-6).float().mean().cpu()),
            "bound_right_clip_frac": float((bf[..., 1] >= 1.0 - 1e-6).float().mean().cpu()),
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

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        gate, _, _ = self._compute_gate(z)
        multiplier = gate / self.init_keep_prob
        return z * multiplier

    @torch.no_grad()
    def gate_diagnostics(self, z: torch.Tensor) -> dict:
        gate, dyn, prior = self._compute_gate(z)
        gf = gate.detach().float()
        q = torch.quantile(gf.reshape(-1), torch.tensor([0.01, 0.05, 0.50, 0.95, 0.99], device=gf.device))
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
            "gate_frac_lt_0p2": float((gf < 0.2).float().mean().cpu()),
            "gate_frac_lt_0p5": float((gf < 0.5).float().mean().cpu()),
            "gate_frac_gt_0p95": float((gf > 0.95).float().mean().cpu()),
            "dyn_mean": float(dyn.detach().float().mean().cpu()),
            "dyn_std": float(dyn.detach().float().std(unbiased=False).cpu()),
            "prior_mean": float(prior.detach().float().mean().cpu()),
            "prior_std": float(prior.detach().float().std(unbiased=False).cpu()),
            "top_gate_feature_idx_8": top,
            "bottom_gate_feature_idx_8": bottom,
        }

    @torch.no_grad()
    def gate_stats(self, z: torch.Tensor) -> dict:
        diag = self.gate_diagnostics(z)
        return {"mean": diag["gate_mean"], "std": diag["gate_std"], "min": diag["gate_min"], "max": diag["gate_max"]}


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
            "abs_max": float("nan"),
            "zero_frac_abs_lt_1e_minus_6": float("nan"),
            "finite_bad_frac": 1.0,
        }
    else:
        abs_vals = vals.abs()
        out = {
            "shape": list(td.shape),
            "dtype": str(td.dtype),
            "mean": float(vals.mean().cpu()),
            "std": float(vals.std(unbiased=False).cpu()),
            "rms": float(torch.sqrt((vals * vals).mean()).cpu()),
            "abs_p95": float(torch.quantile(abs_vals.reshape(-1), 0.95).cpu()),
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
        assert CTN_MIXED_DIM == d_model
        assert len(CTN_CI_KERNELS) == 4
        assert len(CTN_MIXED_KERNELS) == 2

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
        self.feature_gate = FeatureReliabilityGate(
            in_feats=in_feats,
            d_internal=CTN_CI_INTERNAL_DIM,
            dropout=0.05,
            init_keep_prob=0.90,
        )
        self.post_gate_proj = nn.Sequential(
            nn.LayerNorm(self.final_in_dim),
            nn.Linear(self.final_in_dim, CTN_POST_GATE_HIDDEN),
            nn.GELU(),
            nn.Dropout(0.05),
            nn.Linear(CTN_POST_GATE_HIDDEN, d_model),
            nn.LayerNorm(d_model),
        )
        self.mixed_encoder = ConvEncoder(
            d_model=d_model,
            d_ff=CTN_MIXED_DFF,
            kernel_size=CTN_MIXED_KERNELS,
            dropout=dropout,
            activation=act,
            n_layers=len(CTN_MIXED_KERNELS),
            enable_res_param=True,
            norm=norm,
            re_param=re_param,
            small_ks=re_param_kernel,
        )
        self.output_norm = nn.LayerNorm(d_model)

        sample_ms = float(WINDOW_MS) / float(LOOKBACK)
        effective_rf_ms = int(round(46 * sample_ms))
        print(
            f"[ctn-config] arch={MODEL_ARCH_SCHEMA} patch_size={CTN_PATCH_SIZE} stride={CTN_PATCH_STRIDE} "
            f"patch_count={self.patch_count} effective_rf_samples=46 effective_rf_ms={effective_rf_ms}",
            flush=True,
        )
        ci_kernel_str = ",".join(str(k) for k in CTN_CI_KERNELS)
        mixed_kernel_str = ",".join(str(k) for k in CTN_MIXED_KERNELS)
        print(
            f"[ctn-config] ci_layers={len(CTN_CI_KERNELS)} ci_kernels=[{ci_kernel_str}] "
            f"ci_dim={CTN_CI_INTERNAL_DIM} ci_dff={CTN_CI_DFF} ci_res_param=1",
            flush=True,
        )
        print(
            f"[ctn-config] post_gate_in_dim={self.final_in_dim} post_gate_hidden={CTN_POST_GATE_HIDDEN} post_gate_out={d_model}",
            flush=True,
        )
        print(
            f"[ctn-config] mixed_layers={len(CTN_MIXED_KERNELS)} mixed_kernels=[{mixed_kernel_str}] "
            f"mixed_dim={CTN_MIXED_DIM} mixed_dff={CTN_MIXED_DFF} mixed_res_param=1",
            flush=True,
        )

    def forward(self, x):
        out_patch = self.depatch(x).contiguous()  # [B, F, L, patch_size]
        out = self.output_linear(out_patch).contiguous()  # [B, F, L, 8]

        B, F, L, C = out.shape
        assert C == CTN_CI_INTERNAL_DIM
        assert L == self.patch_count

        u = out.reshape(B * F, L, C).permute(0, 2, 1).contiguous()
        assert u.shape == (B * F, CTN_CI_INTERNAL_DIM, self.patch_count)

        out = self.ci_encoder(u)
        assert out.shape == (B * F, CTN_CI_INTERNAL_DIM, self.patch_count)

        out = out.permute(0, 2, 1).contiguous()
        out = out.reshape(B, F, self.patch_count, CTN_CI_INTERNAL_DIM)
        out = out.permute(0, 2, 1, 3).contiguous()  # [B, L, F, 8]

        out = self.feature_gate(out)

        out = out.reshape(B, self.patch_count, F * CTN_CI_INTERNAL_DIM).contiguous()
        out = self.post_gate_proj(out).contiguous()  # [B, L, 1024]

        out_t = out.transpose(1, 2).contiguous()     # [B, 1024, L]
        out_t = self.mixed_encoder(out_t)
        out = out_t.transpose(1, 2).contiguous()     # [B, L, 1024]

        out = self.output_norm(out)
        return out

    @torch.no_grad()
    def residual_scalar_diagnostics(self) -> dict:
        def collect(enc: nn.Module) -> torch.Tensor:
            vals = []
            for mod in enc.modules():
                if isinstance(mod, SublayerConnection) and getattr(mod, "enable", False) and hasattr(mod, "a"):
                    vals.append(mod.a.detach().float().reshape(1))
            if not vals:
                return torch.empty(0)
            return torch.cat(vals)

        ci = collect(self.ci_encoder)
        mixed = collect(self.mixed_encoder)
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
        out.update(stats("mixed", mixed))
        return out

    @torch.no_grad()
    def forward_with_diagnostics(self, x: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        diag = {"activations": {}}
        diag["activations"]["x_input"] = _activation_summary(x)
        out_patch = self.depatch(x).contiguous()
        diag["activations"]["depatch_out"] = _activation_summary(out_patch)
        out = self.output_linear(out_patch).contiguous()
        diag["activations"]["patch_embed"] = _activation_summary(out)

        B, F, L, C = out.shape
        assert C == CTN_CI_INTERNAL_DIM
        assert L == self.patch_count
        u = out.reshape(B * F, L, C).permute(0, 2, 1).contiguous()
        assert u.shape == (B * F, CTN_CI_INTERNAL_DIM, self.patch_count)
        ci_t = self.ci_encoder(u)
        assert ci_t.shape == (B * F, CTN_CI_INTERNAL_DIM, self.patch_count)
        diag["activations"]["ci_out"] = _activation_summary(ci_t)

        pre_gate = ci_t.permute(0, 2, 1).contiguous().reshape(B, F, self.patch_count, CTN_CI_INTERNAL_DIM)
        pre_gate = pre_gate.permute(0, 2, 1, 3).contiguous()
        diag["activations"]["pre_gate"] = _activation_summary(pre_gate)
        post_gate = self.feature_gate(pre_gate)
        diag["activations"]["post_gate"] = _activation_summary(post_gate)
        post_gate_flat = post_gate.reshape(B, self.patch_count, F * CTN_CI_INTERNAL_DIM).contiguous()
        diag["activations"]["post_gate_flat"] = _activation_summary(post_gate_flat)
        post_proj = self.post_gate_proj(post_gate_flat).contiguous()
        diag["activations"]["post_proj"] = _activation_summary(post_proj)
        post_mixed_t = self.mixed_encoder(post_proj.transpose(1, 2).contiguous())
        post_mixed = post_mixed_t.transpose(1, 2).contiguous()
        diag["activations"]["post_mixed"] = _activation_summary(post_mixed)
        out = self.output_norm(post_mixed)
        diag["activations"]["extractor_out"] = _activation_summary(out)

        eps = 1e-12
        diag["ratios"] = {
            "gate_over_ci_rms": diag["activations"]["post_gate"]["rms"] / (diag["activations"]["ci_out"]["rms"] + eps),
            "proj_over_flat_rms": diag["activations"]["post_proj"]["rms"] / (diag["activations"]["post_gate_flat"]["rms"] + eps),
            "mixed_over_proj_rms": diag["activations"]["post_mixed"]["rms"] / (diag["activations"]["post_proj"]["rms"] + eps),
        }
        diag["gate"] = self.feature_gate.gate_diagnostics(pre_gate)
        diag["depatch"] = self.depatch.diagnostics(x)
        diag["residual_scalars"] = self.residual_scalar_diagnostics()
        return out, diag

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
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        assert args.d_model == DMODEL, f"Expected args.d_model ({args.d_model}) == DMODEL ({DMODEL})"
        assert args.d_model % NUM_HEADS == 0, "args.d_model must be divisible by NUM_HEADS"
        assert args.headdim == (args.d_model // NUM_HEADS), "args.headdim must match d_model // NUM_HEADS"
        self.depatch_proj_encoder = ConvTimeNetFeatureExtractor(
            in_feats=args.vocab_size,
            seq_len=args.seq_in,
            d_model=args.d_model,
            dropout=0.1,
            act="gelu",
            norm="layer",
            re_param=True,
            re_param_kernel=3,
        )
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

    def forward(self, x):
        x_permuted = x.permute(0, 2, 1).contiguous()
        h_tokens = self.depatch_proj_encoder(x_permuted).contiguous()        # [B, L, D] (ConvTimeNet projection applied)
        h, _ = self.mamba(h_tokens, embedded=True)
        h_dir = self.dir_token_decoder(h)
        h_mag = self.mag_token_decoder(h)
        pooled_dir = self.dir_pool(h_dir)
        pooled_mag = self.mag_pool(h_mag)
        dir_logits = self.dir_head(pooled_dir)
        mag_up_sqrt = F.softplus(self.mag_up_head(pooled_mag)) + MAG_SQRT_EPS
        mag_down_sqrt = F.softplus(self.mag_down_head(pooled_mag)) + MAG_SQRT_EPS
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
        x_permuted = x.permute(0, 2, 1).contiguous()
        h_tokens, ext_diag = self.depatch_proj_encoder.forward_with_diagnostics(x_permuted)
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
        mag_up_sqrt = F.softplus(self.mag_up_head(pooled_mag)) + MAG_SQRT_EPS
        mag_down_sqrt = F.softplus(self.mag_down_head(pooled_mag)) + MAG_SQRT_EPS
        diag["activations"]["dir_logits"] = _activation_summary(dir_logits)
        diag["activations"]["mag_up_sqrt"] = _activation_summary(mag_up_sqrt)
        diag["activations"]["mag_down_sqrt"] = _activation_summary(mag_down_sqrt)
        probs = torch.sigmoid(dir_logits.float())
        diag["prediction"] = {
            "dir_logit_mean": float(dir_logits.float().mean().cpu()),
            "dir_logit_std": float(dir_logits.float().std(unbiased=False).cpu()),
            "dir_logit_abs_p95": float(torch.quantile(dir_logits.float().abs().reshape(-1), 0.95).cpu()),
            "dir_prob_mean": float(probs.mean().cpu()),
            "dir_prob_std": float(probs.std(unbiased=False).cpu()),
            "mag_up_mean": float(mag_up_sqrt.float().mean().cpu()),
            "mag_up_std": float(mag_up_sqrt.float().std(unbiased=False).cpu()),
            "mag_down_mean": float(mag_down_sqrt.float().mean().cpu()),
            "mag_down_std": float(mag_down_sqrt.float().std(unbiased=False).cpu()),
        }
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

        self.return_deq: Deque[Tuple[int, int, float]] = deque()
        self.return_n: int = 0
        self.return_sum: float = 0.0
        self.return_sumsq: float = 0.0
        self.return_pos_count: int = 0
        self.return_neg_count: int = 0
        self.return_sign_sum: float = 0.0

        self.pair_deq: Deque[Tuple[int, int, float, float]] = deque()
        self.pair_n: int = 0
        self.pair_sum_x: float = 0.0
        self.pair_sum_y: float = 0.0
        self.pair_sum_x2: float = 0.0
        self.pair_sum_y2: float = 0.0
        self.pair_sum_xy: float = 0.0

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

    def _append_return(self, left_ts: int, right_ts: int, ret_bps: float) -> None:
        r = float(ret_bps)
        self.return_deq.append((int(left_ts), int(right_ts), r))
        self.return_n += 1
        self.return_sum += r
        self.return_sumsq += r * r
        sign = 1.0 if r > 0.0 else (-1.0 if r < 0.0 else 0.0)
        self.return_sign_sum += sign
        if r > 0.0:
            self.return_pos_count += 1
        elif r < 0.0:
            self.return_neg_count += 1

    def _remove_return(self, left_ts: int, right_ts: int, ret_bps: float) -> None:
        _ = (left_ts, right_ts)
        r = float(ret_bps)
        self.return_n -= 1
        self.return_sum -= r
        self.return_sumsq -= r * r
        sign = 1.0 if r > 0.0 else (-1.0 if r < 0.0 else 0.0)
        self.return_sign_sum -= sign
        if r > 0.0:
            self.return_pos_count -= 1
        elif r < 0.0:
            self.return_neg_count -= 1
        if self.return_n <= 0:
            self.return_n = 0
            self.return_sum = 0.0
            self.return_sumsq = 0.0
            self.return_pos_count = 0
            self.return_neg_count = 0
            self.return_sign_sum = 0.0

    def _append_pair(self, first_return_left_ts: int, second_return_right_ts: int, r0: float, r1: float) -> None:
        x = float(r0)
        y = float(r1)
        self.pair_deq.append((int(first_return_left_ts), int(second_return_right_ts), x, y))
        self.pair_n += 1
        self.pair_sum_x += x
        self.pair_sum_y += y
        self.pair_sum_x2 += x * x
        self.pair_sum_y2 += y * y
        self.pair_sum_xy += x * y

    def _remove_pair(self, first_return_left_ts: int, second_return_right_ts: int, r0: float, r1: float) -> None:
        _ = (first_return_left_ts, second_return_right_ts)
        x = float(r0)
        y = float(r1)
        self.pair_n -= 1
        self.pair_sum_x -= x
        self.pair_sum_y -= y
        self.pair_sum_x2 -= x * x
        self.pair_sum_y2 -= y * y
        self.pair_sum_xy -= x * y
        if self.pair_n <= 0:
            self.pair_n = 0
            self.pair_sum_x = 0.0
            self.pair_sum_y = 0.0
            self.pair_sum_x2 = 0.0
            self.pair_sum_y2 = 0.0
            self.pair_sum_xy = 0.0

    def _rebuild_returns_and_pairs_from_prices(self) -> None:
        while self.return_deq:
            left_ts, right_ts, ret_bps = self.return_deq.popleft()
            self._remove_return(left_ts, right_ts, ret_bps)
        while self.pair_deq:
            left_ts, right_ts, r0, r1 = self.pair_deq.popleft()
            self._remove_pair(left_ts, right_ts, r0, r1)
        if len(self.deq) < 2:
            return
        for i in range(1, len(self.deq)):
            prev_ts, prev_p, _prev_logp, _prev_t = self.deq[i - 1]
            cur_ts, cur_p, _cur_logp, _cur_t = self.deq[i]
            ret_bps = 1e4 * math.log(cur_p / prev_p)
            self._append_return(prev_ts, cur_ts, ret_bps)
            if len(self.return_deq) >= 2:
                prev_return = self.return_deq[-2][2]
                cur_return = self.return_deq[-1][2]
                self._append_pair(self.return_deq[-2][0], self.return_deq[-1][1], prev_return, cur_return)

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

        if len(self.deq) >= 2:
            prev_ts, prev_p, _prev_logp, _prev_t = self.deq[-2]
            ret_bps = 1e4 * math.log(p / prev_p)
            self._append_return(prev_ts, ts, ret_bps)
        if len(self.return_deq) >= 2:
            prev_return = self.return_deq[-2][2]
            cur_return = self.return_deq[-1][2]
            self._append_pair(self.return_deq[-2][0], self.return_deq[-1][1], prev_return, cur_return)

    def prune(self, now_ms: int) -> None:
        cutoff = int(now_ms) - self.window_ms
        changed = False
        while self.deq and self.deq[0][0] < cutoff:
            _ts, p, logp, t_rel_sec = self.deq.popleft()
            self.n -= 1
            self.sum_t -= t_rel_sec
            self.sum_t2 -= t_rel_sec * t_rel_sec
            self.sum_logp -= logp
            self.sum_logp2 -= logp * logp
            self.sum_t_logp -= t_rel_sec * logp
            self._remove_sorted_price(p)
            changed = True
        if self.n <= 0:
            self.n = 0
            self.sum_t = 0.0
            self.sum_t2 = 0.0
            self.sum_logp = 0.0
            self.sum_logp2 = 0.0
            self.sum_t_logp = 0.0
        if changed:
            self._rebuild_returns_and_pairs_from_prices()

    def slope_r2(self) -> Tuple[float, float]:
        if self.n < 3:
            return 0.0, 0.0
        n = float(self.n)
        x_mean = self.sum_t / n
        y_mean = self.sum_logp / n
        x_var = self.sum_t2 / n - x_mean * x_mean
        y_var = self.sum_logp2 / n - y_mean * y_mean
        if x_var <= 1e-12:
            return 0.0, 0.0
        cov = self.sum_t_logp / n - x_mean * y_mean
        slope = 1e4 * cov / x_var
        if y_var <= 1e-12:
            return float(slope), 0.0
        r2 = (cov * cov) / max(x_var * y_var, 1e-30)
        r2 = max(0.0, min(1.0, r2))
        return float(slope), float(r2)

    def range_features(self, current: float) -> Tuple[float, float, float, float, float, float]:
        if self.n < 3:
            return 0.5, 0.0, 0.0, 0.0, 0.0, 0.0
        low = self.sorted_prices[0]
        high = self.sorted_prices[-1]
        cur = float(current)
        if high <= low:
            position = 0.5
        else:
            position = max(0.0, min(1.0, (cur - low) / (high - low)))
        dist_to_high_bps = 1e4 * math.log(high / cur) if high > 0.0 and cur > 0.0 else 0.0
        dist_to_low_bps = 1e4 * math.log(cur / low) if cur > 0.0 and low > 0.0 else 0.0
        rolling_range_bps = 1e4 * math.log(high / low) if high > 0.0 and low > 0.0 else 0.0
        breakout_up = 1.0 if cur >= high else 0.0
        breakout_down = 1.0 if cur <= low else 0.0
        return (
            float(position),
            float(dist_to_high_bps),
            float(dist_to_low_bps),
            float(rolling_range_bps),
            float(breakout_up),
            float(breakout_down),
        )

    def return_shape_features(self) -> Tuple[float, float, float]:
        if self.n < 3 or self.return_n < 2:
            return 0.0, 0.0, 0.0
        denom = float(self.return_n)
        sign_persistence = abs(self.return_sign_sum) / denom
        up_return_fraction = float(self.return_pos_count) / denom
        if self.return_n < 3 or self.pair_n < 2:
            autocorr = 0.0
        else:
            n = float(self.pair_n)
            mean_x = self.pair_sum_x / n
            mean_y = self.pair_sum_y / n
            var_x = self.pair_sum_x2 / n - mean_x * mean_x
            var_y = self.pair_sum_y2 / n - mean_y * mean_y
            if var_x <= 1e-24 or var_y <= 1e-24:
                autocorr = 0.0
            else:
                cov = self.pair_sum_xy / n - mean_x * mean_y
                autocorr = cov / math.sqrt(var_x * var_y)
        if not math.isfinite(autocorr):
            autocorr = 0.0
        return float(sign_persistence), float(up_return_fraction), float(autocorr)

    def features(self, current: float) -> Tuple[float, float, float, float, float, float, float, float, float, float, float]:
        slope, r2 = self.slope_r2()
        pos, d_high, d_low, rng, br_up, br_down = self.range_features(current)
        sign_persistence, up_frac, autocorr = self.return_shape_features()
        return slope, r2, pos, d_high, d_low, rng, br_up, br_down, sign_persistence, up_frac, autocorr


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
    threshold_counts: Dict[float, Dict[str, float]]
    max_heap: List[Tuple[float, int, float]]
    cluster_ts: Deque[int]
    cluster_count: int

    @classmethod
    def create(cls, window_ms: int) -> "LargeTradeWindowState":
        return cls(
            window_ms=int(window_ms),
            threshold_counts={
                float(t): {"buy_count": 0.0, "sell_count": 0.0, "buy_notional": 0.0, "sell_notional": 0.0}
                for t in LARGE_TRADE_NOTIONAL_USD
            },
            max_heap=[],
            cluster_ts=deque(),
            cluster_count=0,
        )


@dataclass
class TradeBurstWindowState:
    window_ms: int
    signs: Deque[Tuple[int, int, int]]
    runs: Deque[Tuple[int, int, int]]
    run_lengths: Dict[int, Tuple[int, int]]
    buy_heap: List[Tuple[int, int]]
    sell_heap: List[Tuple[int, int]]
    buy_count: int
    sell_count: int
    pair_n: int
    sum_x: float
    sum_y: float
    sum_x2: float
    sum_y2: float
    sum_xy: float
    next_run_id: int

    @classmethod
    def create(cls, window_ms: int) -> "TradeBurstWindowState":
        return cls(
            window_ms=int(window_ms),
            signs=deque(),
            runs=deque(),
            run_lengths={},
            buy_heap=[],
            sell_heap=[],
            buy_count=0,
            sell_count=0,
            pair_n=0,
            sum_x=0.0,
            sum_y=0.0,
            sum_x2=0.0,
            sum_y2=0.0,
            sum_xy=0.0,
            next_run_id=1,
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
        z_hl_ms: int = 30_000,
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
        self._last_any_event_ts: Optional[int] = None
        self._last_ob_feature_ts: Optional[int] = None
        self.last_trade_ts: Optional[int] = None
        self.last_bid1_update_ts: Optional[int] = None
        self.last_ask1_update_ts: Optional[int] = None
        self.last_bid_price_change_ts: Optional[int] = None
        self.last_ask_price_change_ts: Optional[int] = None
        self.last_mid_change_ts: Optional[int] = None
        self.last_spread_widen_ts: Optional[int] = None
        self.last_spread_tighten_ts: Optional[int] = None
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
        # (ts_ms, price, size, notional_usd, side, side_sign, tick_sign, is_zero_tick)
        self.trade_windows: Tuple[int, ...] = FLOW_WINDOWS_MS
        self._trade_window_deques: Dict[int, Deque[Tuple[int, float, float, float, str, float, float, float]]] = {
            ms: deque() for ms in self.trade_windows
        }
        self._trade_seq: int = 0
        self.trade_window_state: Dict[int, Dict[str, Any]] = {
            window: self._new_trade_window_state()
            for window in self.trade_windows
        }
        self.large_trade_states: Dict[int, LargeTradeWindowState] = {
            ms: LargeTradeWindowState.create(ms) for ms in self.trade_windows
        }
        self.cvd_notional = 0.0
        self._cvd_ema = {ms: 0.0 for ms in FLOW_WINDOWS_MS}
        self._cvd_ema_initialized = {ms: False for ms in FLOW_WINDOWS_MS}
        self.cvd_window_states: Dict[int, CVDWindowState] = {
            ms: CVDWindowState.create(ms, initial_cvd=0.0) for ms in self.trade_windows
        }
        self.last_cvd_update_ts: Optional[int] = None
        self.last_large_buy_ts: Optional[int] = None
        self.last_large_sell_ts: Optional[int] = None
        self.last_large_buy_mid: Optional[float] = None
        self.last_large_sell_mid: Optional[float] = None
        self.last_large_buy_notional_usd: float = 0.0
        self.last_large_sell_notional_usd: float = 0.0
        self.last_large_buy_ofi_l5_at_event: float = 0.0
        self.last_large_sell_ofi_l5_at_event: float = 0.0
        self.last_large_buy_trade_imbalance_at_event: float = 0.0
        self.last_large_sell_trade_imbalance_at_event: float = 0.0
        self.consecutive_buy_trade_count: int = 0
        self.consecutive_sell_trade_count: int = 0
        self.trade_burst_states: Dict[int, TradeBurstWindowState] = {
            ms: TradeBurstWindowState.create(ms) for ms in TRADE_BURST_WINDOWS_MS
        }
        self.trade_sign_history: Deque[Tuple[int, int, float]] = deque()
        self.last_ob_ofi_l5: float = 0.0
        self.last_ob_trade_imbalance_30000ms: float = 0.0

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
            (level, window): RollingScalarWindowState(
                window,
                track_sign=True,
            )
            for level in ROLLING_OBI_LEVELS
            for window in ROLLING_OBI_WINDOWS_MS
        }
        self.deep_micro_histories: Dict[int, Deque[Tuple[int, float]]] = {
            5: deque(),
            10: deque(),
        }

        # ---------- MACD state ----------
        self.macd_state = {
            (fast, slow, sig): {
                "fast": None,
                "slow": None,
                "signal": 0.0,
                "signal_initialized": False,
            }
            for (fast, slow, sig) in MACD_TRIPLETS_MS
        }

        # ---------- Fast EMA state for microstructure signals ----------
        self.ema_half_lives_ms = EMA_HALF_LIVES_MS
        self.ema_indicator_names = (
            "spread_bps",
            "gap_a_bps",
            "gap_b_bps",
            "micro_premia",
            "micro_minus_mid_bps",
            "depth_imbalance_within_1bps",
            "depth_imbalance_within_2bps",
            "depth_imbalance_within_5bps",
            "depth_imbalance_within_10bps",
            "notional_imbalance_within_1bps",
            "notional_imbalance_within_2bps",
            "notional_imbalance_within_5bps",
            "notional_imbalance_within_10bps",
            "obi_l1",
            "obi_l3",
            "obi_l5",
            "obi_l10",
            "ofi_l1_over_depth_l1",
            "ofi_l3_over_depth_l3",
            "ofi_l5_over_depth_l5",
            "ofi_l10_over_depth_l10",
            "signed_notional_flow_usd_30000ms",
            "trade_imbalance_notional_30000ms",
            "vwap_vs_mid_bps_30000ms",
        )
        self.ema_states: Dict[int, Dict[str, Optional[float]]] = {
            hl: {name: None for name in self.ema_indicator_names}
            for hl in self.ema_half_lives_ms
        }

        # ---------- Rolling z-score state (per-feature EWMA mean/var) ----------
        self.z_mean: Optional[np.ndarray] = None
        self.z_var: Optional[np.ndarray] = None
        self._feat_dim: Optional[int] = None
        self._feature_names_cache: Optional[List[str]] = None
        self._z_half_lives_ms: Optional[List[Optional[int]]] = None
        # Empty immutable-ish placeholder returned for trade events.
        # Trade rows are never consumed by offline_ingest, so this avoids per-trade allocation.
        self._trade_fast_path_empty_feature = np.empty((0,), dtype=np.float32)
        # Vectorized z-score metadata.
        self._z_half_lives_arr: Optional[np.ndarray] = None
        self._z_mask: Optional[np.ndarray] = None
        self._last_z_ts_ms: Optional[int] = None

        self.trade_fast_path_count: int = 0
        self.ob_feature_build_count: int = 0
        self.strict_feature_validation = os.environ.get("BYBIT_STRICT_FEATURE_VALIDATION", "0") == "1"

    def feature_names(self) -> List[str]:
        if self._feature_names_cache is not None:
            return list(self._feature_names_cache)
        names: List[str] = []
        names.extend([
            "time_hour_sin",
            "time_hour_cos",
            "time_dow_sin",
            "time_dow_cos",
            "session_is_weekend",
            "session_is_asia",
            "session_is_europe",
            "session_is_us",
            "session_is_europe_us_overlap",
        ])
        for w in PRICE_WINDOWS_MS:
            names.extend([
                f"mid_ret_bps_{w}ms",
                f"micro_ret_bps_{w}ms",
                f"mid_slope_bps_per_sec_{w}ms",
                f"mid_trend_r2_{w}ms",
                f"mid_position_in_range_{w}ms",
                f"mid_dist_to_high_bps_{w}ms",
                f"mid_dist_to_low_bps_{w}ms",
                f"mid_range_bps_{w}ms",
                f"mid_breakout_up_{w}ms",
                f"mid_breakout_down_{w}ms",
                f"sign_persistence_{w}ms",
                f"up_return_fraction_{w}ms",
                f"return_autocorr_lag1_{w}ms",
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
            "time_since_bid_price_change_ms",
            "time_since_ask_price_change_ms",
            "time_since_mid_change_ms",
            "time_since_spread_widen_ms",
            "time_since_spread_tighten_ms",
            "best_bid_lifetime_ms",
            "best_ask_lifetime_ms",
            "mid_price_staleness_ms",
        ])
        for lvl in BOOK_DEPTH_FEATURE_LEVELS:
            names.append(f"cum_bid_l{lvl}")
            names.append(f"cum_ask_l{lvl}")
        for lvl in BOOK_DEPTH_FEATURE_LEVELS:
            names.append(f"obi_l{lvl}")
        for lvl in BOOK_DEPTH_FEATURE_LEVELS:
            names.append(f"ofi_l{lvl}")
        for lvl in NORMALIZED_OFI_LEVELS:
            names.append(f"ofi_l{lvl}_over_depth_l{lvl}")
            names.append(f"ofi_l{lvl}_over_spread_bps")
            names.append(f"ofi_l{lvl}_over_depth_5bps")
        for level in ROLLING_OFI_LEVELS:
            for window in ROLLING_OFI_WINDOWS_MS:
                names.append(f"ofi_l{level}_sum_{window}ms")
                names.append(f"ofi_l{level}_sum_over_depth_{window}ms")
        for level in ROLLING_OFI_LEVELS:
            names.append(f"ofi_l{level}_accel_7500_minus_30000ms")
            names.append(f"ofi_l{level}_accel_15000_minus_60000ms")
        for level in ROLLING_OBI_LEVELS:
            for window in ROLLING_OBI_WINDOWS_MS:
                names.append(f"obi_l{level}_mean_{window}ms")
                names.append(f"obi_l{level}_slope_{window}ms")
                names.append(f"obi_l{level}_sign_persistence_{window}ms")
        for level in DEEP_MICRO_LEVELS:
            names.extend([
                f"micro_l{level}_minus_mid_bps",
                f"micro_l{level}_minus_mid_over_spread",
                f"vamp_l{level}_minus_mid_bps",
            ])
        names.extend([
            "micro_l5_slope_7500ms",
            "micro_l5_slope_30000ms",
            "micro_l10_slope_7500ms",
            "micro_l10_slope_30000ms",
            "micro_l1_minus_micro_l10_bps",
            "micro_l5_minus_micro_l20_bps",
        ])
        for band in BPS_DEPTH_BANDS:
            b = self._fmt_bps_band(band)
            names.extend([
                f"bid_depth_within_{b}bps",
                f"ask_depth_within_{b}bps",
                f"bid_notional_within_{b}bps",
                f"ask_notional_within_{b}bps",
                f"depth_imbalance_within_{b}bps",
                f"notional_imbalance_within_{b}bps",
            ])
        for band in BOOK_SHAPE_BANDS:
            b = self._fmt_bps_band(band)
            names.extend([
                f"max_bid_size_within_{b}bps",
                f"max_ask_size_within_{b}bps",
                f"max_bid_notional_within_{b}bps",
                f"max_ask_notional_within_{b}bps",
                f"dist_to_max_bid_wall_bps_within_{b}bps",
                f"dist_to_max_ask_wall_bps_within_{b}bps",
                f"bid_depth_hhi_within_{b}bps",
                f"ask_depth_hhi_within_{b}bps",
                f"bid_top1_share_within_{b}bps",
                f"ask_top1_share_within_{b}bps",
            ])
        names.extend([
            "book_slope_bid_top5",
            "book_slope_ask_top5",
            "book_slope_bid_5bps",
            "book_slope_ask_5bps",
            "book_convexity_bid_10bps",
            "book_convexity_ask_10bps",
        ])
        for notional in SLIPPAGE_NOTIONAL_USD:
            n = self._fmt_usd_notional(notional)
            names.extend([
                f"slippage_bps_to_buy_{n}",
                f"slippage_bps_to_sell_{n}",
                f"depth_needed_bps_to_buy_{n}",
                f"depth_needed_bps_to_sell_{n}",
                f"filled_fraction_to_buy_{n}",
                f"filled_fraction_to_sell_{n}",
            ])
        for notional in SLIPPAGE_NOTIONAL_USD:
            n = self._fmt_usd_notional(notional)
            names.extend([
                f"slippage_imbalance_bps_{n}",
                f"depth_needed_imbalance_bps_{n}",
                f"filled_fraction_imbalance_{n}",
            ])
        names.extend([
            "buy_slippage_slope_10000_to_50000usd",
            "sell_slippage_slope_10000_to_50000usd",
            "buy_slippage_slope_50000_to_250000usd",
            "sell_slippage_slope_50000_to_250000usd",
            "slippage_curve_convexity_buy",
            "slippage_curve_convexity_sell",
        ])
        for ms in FAST_WINDOWS_MS:
            names.extend([
                f"spread_delta_bps_{ms}ms",
                f"spread_change_count_{ms}ms",
                f"bid_price_change_count_{ms}ms",
                f"ask_price_change_count_{ms}ms",
                f"bid_price_change_rate_{ms}ms",
                f"ask_price_change_rate_{ms}ms",
                f"bid_l1_depletion_{ms}ms",
                f"ask_l1_depletion_{ms}ms",
                f"bid_l1_depletion_over_depth_{ms}ms",
                f"ask_l1_depletion_over_depth_{ms}ms",
            ])
        for ms in FAST_WINDOWS_MS:
            for level in (1, 2):
                names.extend([
                    f"bid_l{level}_add_rate_{ms}ms",
                    f"bid_l{level}_rem_rate_{ms}ms",
                    f"ask_l{level}_add_rate_{ms}ms",
                    f"ask_l{level}_rem_rate_{ms}ms",
                    f"bid_l{level}_add_rate_over_depth_{ms}ms",
                    f"bid_l{level}_rem_rate_over_depth_{ms}ms",
                    f"ask_l{level}_add_rate_over_depth_{ms}ms",
                    f"ask_l{level}_rem_rate_over_depth_{ms}ms",
                ])
        for ms in FLOW_WINDOWS_MS:
            names.extend([
                f"buy_vol_base_{ms}ms",
                f"sell_vol_base_{ms}ms",
                f"buy_notional_usd_{ms}ms",
                f"sell_notional_usd_{ms}ms",
                f"buy_count_{ms}ms",
                f"sell_count_{ms}ms",
                f"buy_mean_notional_usd_{ms}ms",
                f"sell_mean_notional_usd_{ms}ms",
                f"buy_max_notional_usd_{ms}ms",
                f"sell_max_notional_usd_{ms}ms",
                f"signed_notional_flow_usd_{ms}ms",
                f"signed_trade_count_imbalance_{ms}ms",
                f"trade_imbalance_notional_{ms}ms",
                f"trade_toxicity_notional_{ms}ms",
                f"plus_tick_fraction_{ms}ms",
                f"minus_tick_fraction_{ms}ms",
                f"zero_tick_fraction_{ms}ms",
                f"tick_sign_imbalance_{ms}ms",
                f"trade_count_{ms}ms",
                f"trade_count_per_second_{ms}ms",
                f"vwap_vs_mid_bps_{ms}ms",
                f"vwap_vs_micro_bps_{ms}ms",
                f"signed_trade_premium_bps_count_weighted_{ms}ms",
                f"signed_trade_premium_bps_volume_weighted_{ms}ms",
                f"buy_trade_premium_bps_{ms}ms",
                f"sell_trade_premium_bps_{ms}ms",
                f"aggressor_price_impact_bps_{ms}ms",
            ])
        for ms in FLOW_WINDOWS_MS:
            names.extend([
                f"cvd_change_usd_{ms}ms",
                f"cvd_slope_usd_per_sec_{ms}ms",
                f"cvd_minus_ema_usd_{ms}ms",
            ])
        names.extend([
            "imbalance_1000ms_minus_7500ms",
            "imbalance_3000ms_minus_15000ms",
            "imbalance_7500ms_minus_30000ms",
            "net_flow_3000ms_minus_30000ms",
            "ofi_imbalance_3000ms_minus_15000ms",
            "ofi_imbalance_7500ms_minus_30000ms",
            "consecutive_buy_trade_count",
            "consecutive_sell_trade_count",
        ])
        for window in TRADE_BURST_WINDOWS_MS:
            names.extend([
                f"max_buy_run_length_{window}ms",
                f"max_sell_run_length_{window}ms",
                f"trade_sign_autocorr_lag1_{window}ms",
                f"trade_sign_entropy_{window}ms",
                f"trade_burst_score_{window}ms",
                f"buy_trade_burst_score_{window}ms",
                f"sell_trade_burst_score_{window}ms",
            ])
        for ms in FLOW_WINDOWS_MS:
            for threshold in LARGE_TRADE_NOTIONAL_USD:
                thr = self._fmt_usd_notional(threshold)
                names.extend([
                    f"large_buy_count_ge_{thr}_{ms}ms",
                    f"large_sell_count_ge_{thr}_{ms}ms",
                    f"large_buy_notional_ge_{thr}_{ms}ms",
                    f"large_sell_notional_ge_{thr}_{ms}ms",
                    f"large_trade_imbalance_ge_{thr}_{ms}ms",
                ])
            names.extend([
                f"max_signed_trade_notional_usd_{ms}ms",
                f"top5_trade_notional_sum_usd_{ms}ms",
                f"large_trade_cluster_count_{ms}ms",
            ])
        names.extend([
            "time_since_large_buy_ms",
            "time_since_large_sell_ms",
            "last_large_buy_notional_usd",
            "last_large_sell_notional_usd",
            "return_since_last_large_buy_bps",
            "return_since_last_large_sell_bps",
            "ofi_l5_since_last_large_buy",
            "ofi_l5_since_last_large_sell",
            "trade_imbalance_since_last_large_buy",
            "trade_imbalance_since_last_large_sell",
            "large_buy_continuation_bps_7500ms",
            "large_buy_continuation_bps_15000ms",
            "large_sell_continuation_bps_7500ms",
            "large_sell_continuation_bps_15000ms",
        ])
        for ms in FLOW_WINDOWS_MS:
            names.extend([
                f"buy_flow_without_price_up_{ms}ms",
                f"sell_flow_without_price_down_{ms}ms",
                f"absorption_bid_{ms}ms",
                f"absorption_ask_{ms}ms",
                f"signed_flow_per_bp_move_{ms}ms",
                f"price_response_to_buy_flow_{ms}ms",
                f"price_response_to_sell_flow_{ms}ms",
            ])
        for ms in FLOW_WINDOWS_MS:
            names.append(f"return_std_bps_{ms}ms")
        for prev_ms, cur_ms in zip(FLOW_WINDOWS_MS[:-1], FLOW_WINDOWS_MS[1:]):
            names.append(f"variance_ratio_{cur_ms}ms_over_{prev_ms}ms")
        for ms in REGIME_WINDOWS_MS:
            names.extend([
                f"regime_volume_ewma_{ms}ms",
                f"regime_realized_vol_bps_{ms}ms",
                f"regime_vol_ewma_bps_{ms}ms",
            ])
            if ms <= 60_000:
                names.append(f"regime_flow_imbalance_{ms}ms")
            names.extend([
                f"realized_up_vol_bps_{ms}ms",
                f"realized_down_vol_bps_{ms}ms",
                f"down_up_vol_ratio_{ms}ms",
                f"bipower_variation_{ms}ms",
                f"jump_variation_{ms}ms",
                f"max_abs_return_bps_{ms}ms",
                f"return_skew_{ms}ms",
                f"return_kurtosis_{ms}ms",
            ])
        for ms in SPREAD_DEPTH_REGIME_WINDOWS_MS:
            names.extend([
                f"spread_mean_bps_{ms}ms",
                f"spread_std_bps_{ms}ms",
                f"spread_p90_bps_{ms}ms",
                f"spread_max_bps_{ms}ms",
                f"spread_min_bps_{ms}ms",
                f"spread_z_{ms}ms",
                f"spread_widening_slope_bps_per_sec_{ms}ms",
                f"spread_time_above_1bp_frac_{ms}ms",
                f"depth_5bps_mean_{ms}ms",
                f"depth_5bps_std_{ms}ms",
                f"depth_5bps_z_{ms}ms",
                f"depth_imbalance_5bps_mean_{ms}ms",
                f"depth_imbalance_5bps_slope_{ms}ms",
                f"liquidity_shock_bid_5bps_{ms}ms",
                f"liquidity_shock_ask_5bps_{ms}ms",
            ])
        for ms in FAST_WINDOWS_MS:
            names.extend([
                f"ofi_l1_pressure_ewma_{ms}ms",
                f"ofi_l1_pressure_over_depth_5bps_{ms}ms",
                f"ofi_l1_pressure_over_realized_vol_{ms}ms",
            ])
        for ms in INTERACTION_WINDOWS_MS:
            names.extend([
                f"flow_agrees_with_book_{ms}ms",
                f"flow_disagrees_with_book_{ms}ms",
                f"trade_imbalance_x_obi_5bps_{ms}ms",
                f"trade_imbalance_x_ofi_pressure_{ms}ms",
                f"ofi_pressure_x_spread_bps_{ms}ms",
                f"micro_premia_x_trade_imbalance_{ms}ms",
                f"micro_premia_x_depth_imbalance_5bps_{ms}ms",
                f"signed_flow_over_depth_5bps_{ms}ms",
                f"signed_flow_over_spread_bps_{ms}ms",
                f"abs_signed_flow_over_realized_vol_{ms}ms",
            ])
        for hl in self.ema_half_lives_ms:
            for name in self.ema_indicator_names:
                names.append(f"ema_{name}_{hl}ms")
        for hl in self.ema_half_lives_ms:
            for name in self.ema_indicator_names:
                names.append(f"resid_{name}_{hl}ms")
        for fast_ms, slow_ms, sig_ms in MACD_TRIPLETS_MS:
            names.extend([
                f"macd_micro_bps_{fast_ms}_{slow_ms}_{sig_ms}",
                f"macd_signal_bps_{fast_ms}_{slow_ms}_{sig_ms}",
                f"macd_hist_bps_{fast_ms}_{slow_ms}_{sig_ms}",
            ])
        for secs in VPIN_BUCKET_SECS:
            names.append(f"vpin_{str(secs).replace('.', 'p')}s")
        if len(names) != len(set(names)):
            seen = set()
            duplicates = sorted({n for n in names if n in seen or seen.add(n)})
            raise ValueError(f"Duplicate feature names in Stage 4 schema: {duplicates[:20]}")
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

            # Premium terms needed to reproduce the old scan exactly.
            # signed_premium_sum = 1e4 * (signed_price_sum / mid - signed_count)
            "signed_price_sum": 0.0,
            "signed_count": 0.0,

            # signed_premium_weighted =
            #   1e4 * (signed_price_notional_sum / mid - signed_notional)
            "signed_price_notional_sum": 0.0,
            "signed_notional_sum_for_premium": 0.0,

            # buy_trade_premium_bps =
            #   1e4 * (buy_price_sum / mid - buy_premium_count) / buy_premium_count
            "buy_price_sum": 0.0,
            "buy_premium_count": 0.0,

            # sell_trade_premium_bps =
            #   1e4 * (mid * sell_inv_price_sum - sell_premium_count) / sell_premium_count
            "sell_inv_price_sum": 0.0,
            "sell_premium_count": 0.0,

            # Monotonic max queues: entries are (ts_ms, notional_usd).
            "buy_max_q": deque(),
            "sell_max_q": deque(),
        }

    def _fmt_bps_band(self, band: float) -> str:
        if float(band).is_integer():
            return str(int(band))
        return str(float(band)).replace(".", "p")

    def _fmt_usd_notional(self, notional: float) -> str:
        return f"{int(round(float(notional)))}usd"

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

    def _compute_session_features(self, ts_ms: int) -> Dict[str, float]:
        dt_utc = datetime.fromtimestamp(float(ts_ms) / 1000.0, tz=timezone.utc)
        hour_float = dt_utc.hour + (dt_utc.minute / 60.0) + (dt_utc.second / 3600.0)
        dow = dt_utc.weekday()
        hour_phase = (2.0 * math.pi * hour_float) / 24.0
        dow_phase = (2.0 * math.pi * float(dow)) / 7.0
        is_weekend = 1.0 if dow >= 5 else 0.0
        h = dt_utc.hour
        return {
            "time_hour_sin": float(math.sin(hour_phase)),
            "time_hour_cos": float(math.cos(hour_phase)),
            "time_dow_sin": float(math.sin(dow_phase)),
            "time_dow_cos": float(math.cos(dow_phase)),
            "session_is_weekend": is_weekend,
            "session_is_asia": 1.0 if 0 <= h < 8 else 0.0,
            "session_is_europe": 1.0 if 7 <= h < 16 else 0.0,
            "session_is_us": 1.0 if 13 <= h < 22 else 0.0,
            "session_is_europe_us_overlap": 1.0 if 13 <= h < 16 else 0.0,
        }

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

    def _rolling_stats_values(self, vals: List[float]) -> Dict[str, float]:
        if not vals:
            return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "p90": 0.0}
        arr = np.asarray(vals, dtype=np.float64)
        return {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr, ddof=0)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "p90": float(np.quantile(arr, 0.90)),
        }

    def _rolling_mean_values(self, vals: List[float]) -> float:
        n = 0
        total = 0.0
        for value in vals:
            v = float(value)
            if math.isfinite(v):
                total += v
                n += 1
        return total / float(n) if n > 0 else 0.0

    def _rolling_mean_std_values(self, vals: List[float]) -> Tuple[float, float]:
        n = 0
        total = 0.0
        total_sq = 0.0
        for value in vals:
            v = float(value)
            if math.isfinite(v):
                total += v
                total_sq += v * v
                n += 1

        if n <= 0:
            return 0.0, 0.0

        mean = total / float(n)
        var = max(0.0, total_sq / float(n) - mean * mean)
        return mean, math.sqrt(var)

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

    def _rolling_value_slope(self, points: List[Tuple[int, float]]) -> float:
        return self._slope_from_points_no_numpy(points)

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

            state["signed_price_sum"] += px
            state["signed_count"] += 1.0
            state["signed_price_notional_sum"] += px * notion
            state["signed_notional_sum_for_premium"] += notion

            state["buy_price_sum"] += px
            state["buy_premium_count"] += 1.0

            q = state["buy_max_q"]
            while q and q[-1][1] <= notion:
                q.pop()
            q.append((ts_i, notion))

        elif ss < 0:
            state["sell_cnt"] += 1
            state["sell_vol"] += sz
            state["sell_notional"] += notion
            state["signed_notional"] -= notion

            state["signed_price_sum"] -= px
            state["signed_count"] -= 1.0
            state["signed_price_notional_sum"] -= px * notion
            state["signed_notional_sum_for_premium"] += notion

            if px > 0.0:
                state["sell_inv_price_sum"] += 1.0 / px
                state["sell_premium_count"] += 1.0

            q = state["sell_max_q"]
            while q and q[-1][1] <= notion:
                q.pop()
            q.append((ts_i, notion))
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

            state["signed_price_sum"] -= px
            state["signed_count"] -= 1.0
            state["signed_price_notional_sum"] -= px * notion
            state["signed_notional_sum_for_premium"] -= notion

            state["buy_price_sum"] -= px
            state["buy_premium_count"] -= 1.0

            q = state["buy_max_q"]
            if q and q[0][0] == ts_i and abs(q[0][1] - notion) <= 1e-12:
                q.popleft()

        elif ss < 0:
            state["sell_cnt"] -= 1
            state["sell_vol"] -= sz
            state["sell_notional"] -= notion
            state["signed_notional"] += notion

            state["signed_price_sum"] += px
            state["signed_count"] += 1.0
            state["signed_price_notional_sum"] += px * notion
            state["signed_notional_sum_for_premium"] -= notion

            if px > 0.0:
                state["sell_inv_price_sum"] -= 1.0 / px
                state["sell_premium_count"] -= 1.0

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
            "buy_premium_count",
            "sell_premium_count",
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
                "signed_price_sum",
                "signed_count",
                "signed_price_notional_sum",
                "signed_notional_sum_for_premium",
                "buy_price_sum",
                "buy_premium_count",
                "sell_inv_price_sum",
                "sell_premium_count",
            ):
                state[key] = 0.0
            state["buy_cnt"] = 0
            state["sell_cnt"] = 0
            state["plus_tick"] = 0
            state["minus_tick"] = 0
            state["zero_tick"] = 0
            state["buy_max_q"].clear()
            state["sell_max_q"].clear()
        self._large_trade_state_expire(window_ms, entry)

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

        signed_premium_sum = 0.0
        signed_premium_weighted = 0.0
        buy_premium_sum = 0.0
        sell_premium_sum = 0.0
        signed_notional_sum_for_premium = 0.0
        buy_premium_count = sell_premium_count = 0.0

        if mid > 0.0:
            signed_premium_sum = 1e4 * (
                float(state["signed_price_sum"]) / mid
                - float(state["signed_count"])
            )
            signed_premium_weighted = 1e4 * (
                float(state["signed_price_notional_sum"]) / mid
                - float(state["signed_notional"])
            )
            signed_notional_sum_for_premium = float(state["signed_notional_sum_for_premium"])

            buy_premium_count = float(state["buy_premium_count"])
            buy_premium_sum = 1e4 * (
                float(state["buy_price_sum"]) / mid
                - buy_premium_count
            )

            sell_premium_count = float(state["sell_premium_count"])
            sell_premium_sum = 1e4 * (
                mid * float(state["sell_inv_price_sum"])
                - sell_premium_count
            )

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
            "vwap_vs_micro_bps": 0.0,
            "signed_trade_premium_bps_count_weighted": self._safe_div(signed_premium_sum, tot_count_signed, 0.0),
            "signed_trade_premium_bps_volume_weighted": self._safe_div(signed_premium_weighted, signed_notional_sum_for_premium, 0.0),
            "buy_trade_premium_bps": self._safe_div(buy_premium_sum, buy_premium_count, 0.0),
            "sell_trade_premium_bps": self._safe_div(sell_premium_sum, sell_premium_count, 0.0),
            "aggressor_price_impact_bps": self._safe_div(signed_premium_weighted, signed_notional_sum_for_premium, 0.0),
        }
        if vol_sum > eps:
            vwap = pxv_sum / vol_sum
            stats["vwap_vs_mid_bps"] = (1e4 * (vwap / mid - 1.0)) if mid > 0 else 0.0
            stats["vwap_vs_micro_bps"] = (1e4 * (vwap / micro - 1.0)) if micro > 0 else 0.0
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

    def _cum_side_qty(self, side_levels: Sequence[Tuple[float, float]], n: int) -> float:
        return float(sum(float(size) for price, size in side_levels[:n] if float(price) > 0.0 and float(size) > 0.0))

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

    def _safe_corr_lag1(self, vals: List[float]) -> float:
        if len(vals) < 2:
            return 0.0
        x = np.asarray(vals[:-1], dtype=np.float64)
        y = np.asarray(vals[1:], dtype=np.float64)
        if x.size < 1 or y.size < 1:
            return 0.0
        sx = float(np.std(x))
        sy = float(np.std(y))
        if sx <= 1e-12 or sy <= 1e-12:
            return 1.0 if len(vals) >= 2 and all(v == vals[0] for v in vals) else 0.0
        c = float(np.corrcoef(x, y)[0, 1])
        return c if math.isfinite(c) else 0.0

    def _large_trade_state_insert(self, window_ms: int, entry: Tuple[int, float, float, float, str, float, float, float]) -> None:
        ts_ms, _price, _size, notional_usd, _side, side_sign, _tick_sign, _is_zero_tick = entry
        state = self.large_trade_states[window_ms]
        notion = float(notional_usd)
        ss = float(side_sign)
        ts_i = int(ts_ms)
        for threshold, agg in state.threshold_counts.items():
            if notion < threshold:
                continue
            if ss > 0:
                agg["buy_count"] += 1.0
                agg["buy_notional"] += notion
            elif ss < 0:
                agg["sell_count"] += 1.0
                agg["sell_notional"] += notion
        if notion > 0.0:
            heapq.heappush(state.max_heap, (-notion, ts_i, ss))
        if notion >= LARGE_TRADE_CLUSTER_THRESHOLD_USD:
            if not state.cluster_ts or (ts_i - state.cluster_ts[-1]) > LARGE_TRADE_CLUSTER_GAP_MS:
                state.cluster_count += 1
            state.cluster_ts.append(ts_i)

    def _large_trade_state_expire(self, window_ms: int, entry: Tuple[int, float, float, float, str, float, float, float]) -> None:
        ts_ms, _price, _size, notional_usd, _side, side_sign, _tick_sign, _is_zero_tick = entry
        state = self.large_trade_states[window_ms]
        notion = float(notional_usd)
        ss = float(side_sign)
        ts_i = int(ts_ms)
        for threshold, agg in state.threshold_counts.items():
            if notion < threshold:
                continue
            if ss > 0:
                agg["buy_count"] = max(0.0, agg["buy_count"] - 1.0)
                agg["buy_notional"] = max(0.0, agg["buy_notional"] - notion)
            elif ss < 0:
                agg["sell_count"] = max(0.0, agg["sell_count"] - 1.0)
                agg["sell_notional"] = max(0.0, agg["sell_notional"] - notion)
        if notion >= LARGE_TRADE_CLUSTER_THRESHOLD_USD and state.cluster_ts and state.cluster_ts[0] == ts_i:
            expired_ts = state.cluster_ts.popleft()
            state.cluster_count = max(0, state.cluster_count - 1)
            if state.cluster_ts and (state.cluster_ts[0] - expired_ts) <= LARGE_TRADE_CLUSTER_GAP_MS:
                state.cluster_count += 1

    def _large_trade_stats_from_state(self, ms: int, now_ms: int) -> Dict[str, float]:
        cutoff = int(now_ms) - int(ms)
        state = self.large_trade_states[ms]
        out: Dict[str, float] = {}
        for threshold in LARGE_TRADE_NOTIONAL_USD:
            thr_key = self._fmt_usd_notional(threshold)
            agg = state.threshold_counts[float(threshold)]
            lb_notional = float(agg["buy_notional"])
            ls_notional = float(agg["sell_notional"])
            out[f"large_buy_count_ge_{thr_key}_{ms}ms"] = float(agg["buy_count"])
            out[f"large_sell_count_ge_{thr_key}_{ms}ms"] = float(agg["sell_count"])
            out[f"large_buy_notional_ge_{thr_key}_{ms}ms"] = lb_notional
            out[f"large_sell_notional_ge_{thr_key}_{ms}ms"] = ls_notional
            out[f"large_trade_imbalance_ge_{thr_key}_{ms}ms"] = self._safe_div(lb_notional - ls_notional, lb_notional + ls_notional, 0.0)
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
        out[f"large_trade_cluster_count_{ms}ms"] = float(state.cluster_count)
        return out

    def _trade_burst_insert(self, window_ms: int, ts_ms: int, sign: int) -> None:
        if sign == 0:
            return
        st = self.trade_burst_states[window_ms]
        ts_i = int(ts_ms)
        run_id = -1
        if st.signs:
            prev_sign = st.signs[-1][1]
            st.pair_n += 1
            st.sum_x += prev_sign
            st.sum_y += sign
            st.sum_x2 += prev_sign * prev_sign
            st.sum_y2 += sign * sign
            st.sum_xy += prev_sign * sign
        if sign > 0:
            st.buy_count += 1
        else:
            st.sell_count += 1
        if st.runs and st.runs[-1][1] == sign:
            rid, _, length = st.runs.pop()
            length += 1
            st.runs.append((rid, sign, length))
            st.run_lengths[rid] = (sign, length)
            run_id = rid
        else:
            run_id = st.next_run_id
            st.next_run_id += 1
            st.runs.append((run_id, sign, 1))
            st.run_lengths[run_id] = (sign, 1)
        if sign > 0:
            heapq.heappush(st.buy_heap, (-st.run_lengths[run_id][1], run_id))
        else:
            heapq.heappush(st.sell_heap, (-st.run_lengths[run_id][1], run_id))
        st.signs.append((ts_i, sign, run_id))

    def _trade_burst_prune(self, window_ms: int, now_ms: int) -> None:
        cutoff = int(now_ms) - int(window_ms)
        st = self.trade_burst_states[window_ms]
        while st.signs and st.signs[0][0] < cutoff:
            _ts, sign, _rid = st.signs.popleft()
            if sign > 0:
                st.buy_count = max(0, st.buy_count - 1)
            else:
                st.sell_count = max(0, st.sell_count - 1)
            if st.signs:
                nxt = st.signs[0][1]
                st.pair_n = max(0, st.pair_n - 1)
                st.sum_x -= sign
                st.sum_y -= nxt
                st.sum_x2 -= sign * sign
                st.sum_y2 -= nxt * nxt
                st.sum_xy -= sign * nxt
            if st.runs:
                rid, rsign, rlen = st.runs[0]
                rlen -= 1
                st.runs.popleft()
                if rlen > 0:
                    st.runs.appendleft((rid, rsign, rlen))
                    st.run_lengths[rid] = (rsign, rlen)
                    if rsign > 0:
                        heapq.heappush(st.buy_heap, (-rlen, rid))
                    else:
                        heapq.heappush(st.sell_heap, (-rlen, rid))
                else:
                    st.run_lengths.pop(rid, None)

    def _corr_lag1_from_sign_state(self, st: TradeBurstWindowState) -> float:
        if st.buy_count < 0 or st.sell_count < 0 or st.pair_n < 0:
            raise ValueError(
                f"Invalid trade burst counts: buy={st.buy_count} sell={st.sell_count} pair_n={st.pair_n}"
            )
        total = int(st.buy_count + st.sell_count)
        if total < 2 or st.pair_n <= 0:
            return 0.0

        n = float(st.pair_n)
        var_x = st.sum_x2 - (st.sum_x * st.sum_x) / n
        var_y = st.sum_y2 - (st.sum_y * st.sum_y) / n

        if var_x <= 1e-12 or var_y <= 1e-12:
            if total >= 2 and (st.buy_count == total or st.sell_count == total):
                return 1.0
            return 0.0

        cov = st.sum_xy - (st.sum_x * st.sum_y) / n
        corr = cov / math.sqrt(max(var_x, 1e-12) * max(var_y, 1e-12))
        return float(corr) if math.isfinite(float(corr)) else 0.0

    def _trade_burst_features_from_state(self, ts_ms: int) -> Dict[str, float]:
        out: Dict[str, float] = {
            "consecutive_buy_trade_count": float(self.consecutive_buy_trade_count),
            "consecutive_sell_trade_count": float(self.consecutive_sell_trade_count),
        }
        for window in TRADE_BURST_WINDOWS_MS:
            self._trade_burst_prune(window, ts_ms)
            st = self.trade_burst_states[window]
            total_signed = st.buy_count + st.sell_count
            p_buy = self._safe_div(st.buy_count, max(total_signed, 1), 0.0)
            p_sell = self._safe_div(st.sell_count, max(total_signed, 1), 0.0)
            entropy = 0.0
            if total_signed > 0:
                if p_buy > 0.0:
                    entropy -= p_buy * math.log(p_buy)
                if p_sell > 0.0:
                    entropy -= p_sell * math.log(p_sell)
                entropy = self._safe_div(entropy, math.log(2.0), 0.0)
            def _max_run(heap: List[Tuple[int, int]], sign_expected: int) -> int:
                while heap:
                    neg_len, rid = heap[0]
                    cur = st.run_lengths.get(rid)
                    if cur is None or cur[0] != sign_expected or cur[1] != -neg_len:
                        heapq.heappop(heap)
                        continue
                    return int(-neg_len)
                return 0
            max_buy_run = _max_run(st.buy_heap, 1)
            max_sell_run = _max_run(st.sell_heap, -1)
            autocorr = self._corr_lag1_from_sign_state(st)
            out.update({
                f"max_buy_run_length_{window}ms": float(max_buy_run),
                f"max_sell_run_length_{window}ms": float(max_sell_run),
                f"trade_sign_autocorr_lag1_{window}ms": float(autocorr) if math.isfinite(autocorr) else 0.0,
                f"trade_sign_entropy_{window}ms": float(entropy),
                f"trade_burst_score_{window}ms": self._safe_div(max(max_buy_run, max_sell_run), max(total_signed, 1), 0.0),
                f"buy_trade_burst_score_{window}ms": self._safe_div(max_buy_run, max(total_signed, 1), 0.0),
                f"sell_trade_burst_score_{window}ms": self._safe_div(max_sell_run, max(total_signed, 1), 0.0),
            })
            for key in (
                f"trade_sign_autocorr_lag1_{window}ms",
                f"trade_sign_entropy_{window}ms",
                f"trade_burst_score_{window}ms",
                f"buy_trade_burst_score_{window}ms",
                f"sell_trade_burst_score_{window}ms",
            ):
                if not math.isfinite(float(out[key])):
                    raise ValueError(f"Non-finite trade burst feature {key}={out[key]!r}")
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

    def _rolling_slope_r2(self, points: List[Tuple[int, float]]) -> Tuple[float, float]:
        if len(points) < 3:
            return 0.0, 0.0
        t0 = points[0][0]
        v0 = float(points[0][1])
        if v0 <= 0.0 or not math.isfinite(v0):
            return 0.0, 0.0
        x_vals = [(ts - t0) / 1000.0 for ts, _ in points]
        x = np.asarray(x_vals, dtype=np.float64)
        y_vals: List[float] = []
        for _, value in points:
            v = float(value)
            if v <= 0.0 or not math.isfinite(v):
                return 0.0, 0.0
            y_vals.append(1e4 * math.log(v / v0))
        y = np.asarray(y_vals, dtype=np.float64)
        eps = 1e-12
        x_mean = float(np.mean(x))
        y_mean = float(np.mean(y))
        x_var = float(np.mean((x - x_mean) ** 2))
        y_var = float(np.mean((y - y_mean) ** 2))
        if x_var <= eps:
            return 0.0, 0.0
        cov_xy = float(np.mean((x - x_mean) * (y - y_mean)))
        slope = cov_xy / x_var
        if y_var <= eps:
            return float(slope), 0.0
        intercept = y_mean - slope * x_mean
        y_hat = slope * x + intercept
        ss_res = float(np.sum((y - y_hat) ** 2))
        ss_tot = float(np.sum((y - y_mean) ** 2))
        r2 = 0.0 if ss_tot <= eps else max(0.0, min(1.0, 1.0 - ss_res / ss_tot))
        return float(slope), float(r2)

    def _range_features(self, points: List[Tuple[int, float]], current: float) -> Tuple[float, float, float, float, float, float]:
        if len(points) < 3:
            return 0.5, 0.0, 0.0, 0.0, 0.0, 0.0
        values = [float(v) for _, v in points if v > 0.0 and math.isfinite(v)]
        if len(values) < 3:
            return 0.5, 0.0, 0.0, 0.0, 0.0, 0.0
        low = min(values)
        high = max(values)
        cur = float(current)
        if high <= low:
            position = 0.5
        else:
            position = max(0.0, min(1.0, (cur - low) / (high - low)))
        dist_to_high_bps = 1e4 * math.log(high / cur) if high > 0.0 and cur > 0.0 else 0.0
        dist_to_low_bps = 1e4 * math.log(cur / low) if cur > 0.0 and low > 0.0 else 0.0
        rolling_range_bps = 1e4 * math.log(high / low) if high > 0.0 and low > 0.0 else 0.0
        breakout_up = 1.0 if cur >= high else 0.0
        breakout_down = 1.0 if cur <= low else 0.0
        return (
            float(position),
            float(dist_to_high_bps),
            float(dist_to_low_bps),
            float(rolling_range_bps),
            float(breakout_up),
            float(breakout_down),
        )

    def _return_shape_features(self, points: List[Tuple[int, float]]) -> Tuple[float, float, float]:
        if len(points) < 3:
            return 0.0, 0.0, 0.0
        vals = [float(v) for _, v in points if v > 0.0 and math.isfinite(v)]
        if len(vals) < 3:
            return 0.0, 0.0, 0.0
        returns: List[float] = []
        for prev, cur in zip(vals[:-1], vals[1:]):
            if prev <= 0.0 or cur <= 0.0:
                continue
            returns.append(1e4 * math.log(cur / prev))
        if len(returns) < 2:
            return 0.0, 0.0, 0.0
        signs = [1.0 if r > 0 else (-1.0 if r < 0 else 0.0) for r in returns]
        denom = float(len(returns))
        sign_persistence = abs(sum(signs)) / denom if denom > 0 else 0.0
        up_return_fraction = float(sum(1 for r in returns if r > 0.0)) / denom if denom > 0 else 0.0
        if len(returns) < 3:
            autocorr = 0.0
        else:
            r0 = np.asarray(returns[:-1], dtype=np.float64)
            r1 = np.asarray(returns[1:], dtype=np.float64)
            s0 = float(np.std(r0))
            s1 = float(np.std(r1))
            autocorr = float(np.corrcoef(r0, r1)[0, 1]) if s0 > 1e-12 and s1 > 1e-12 else 0.0
        if not math.isfinite(autocorr):
            autocorr = 0.0
        return float(sign_persistence), float(up_return_fraction), float(autocorr)

    def _depth_within_bps(self, levels: List[Tuple[float, float]], mid: float, band_bps: float, is_bid: bool) -> dict:
        eps = 1e-12
        out = {
            "size": 0.0,
            "notional": 0.0,
            "max_size": 0.0,
            "max_notional": 0.0,
            "dist_to_max_bps": 0.0,
            "hhi": 0.0,
            "top1_share": 0.0,
        }
        if mid <= 0.0 or not math.isfinite(mid):
            return out
        selected: List[Tuple[float, float, float]] = []
        for price, size in levels:
            p = float(price)
            s = float(size)
            if p <= 0.0 or s <= 0.0 or not math.isfinite(p) or not math.isfinite(s):
                continue
            if is_bid:
                if p > mid:
                    continue
                dist = 1e4 * (mid - p) / mid
            else:
                if p < mid:
                    continue
                dist = 1e4 * (p - mid) / mid
            if dist <= float(band_bps):
                selected.append((p, s, dist))
        if not selected:
            return out
        total_size = sum(s for _, s, _ in selected)
        if total_size <= eps:
            return out
        total_notional = sum(p * s for p, s, _ in selected)
        max_price, max_size, max_dist = max(selected, key=lambda t: t[1])
        max_notional = max(p * s for p, s, _ in selected)
        shares = [s / total_size for _, s, _ in selected]
        out["size"] = float(total_size)
        out["notional"] = float(total_notional)
        out["max_size"] = float(max_size)
        out["max_notional"] = float(max_notional)
        out["dist_to_max_bps"] = float(max_dist)
        out["hhi"] = float(sum(sh * sh for sh in shares))
        out["top1_share"] = float(max_size / total_size)
        return out

    def _slippage_for_notional(self, levels: List[Tuple[float, float]], mid: float, notional_usd: float, is_buy: bool) -> dict:
        bad = {"slippage_bps": 10_000.0, "depth_needed_bps": 10_000.0, "filled_fraction": 0.0}
        if mid <= 0.0 or notional_usd <= 0.0:
            return dict(bad)
        filled_notional = 0.0
        filled_base = 0.0
        last_price = None
        for price, size in levels:
            p = float(price)
            s = float(size)
            if p <= 0.0 or s <= 0.0 or not math.isfinite(p) or not math.isfinite(s):
                continue
            available_notional = p * s
            take_notional = min(notional_usd - filled_notional, available_notional)
            if take_notional <= 0.0:
                continue
            take_base = take_notional / p
            filled_notional += take_notional
            filled_base += take_base
            last_price = p
            if filled_notional >= notional_usd - 1e-9:
                break
        if filled_base <= 0.0 or last_price is None:
            return dict(bad)
        filled_fraction = max(0.0, min(1.0, filled_notional / max(notional_usd, 1e-12)))
        if filled_fraction < 1.0 - 1e-9:
            return {"slippage_bps": 10_000.0, "depth_needed_bps": 10_000.0, "filled_fraction": float(filled_fraction)}
        vwap = filled_notional / filled_base
        if is_buy:
            slippage_bps = 1e4 * (vwap / mid - 1.0)
            depth_needed_bps = 1e4 * max(last_price - mid, 0.0) / mid
        else:
            slippage_bps = 1e4 * (mid / vwap - 1.0) if vwap > 0.0 else 1e4 * ((mid - vwap) / max(mid, 1e-12))
            depth_needed_bps = 1e4 * max(mid - last_price, 0.0) / mid
        return {
            "slippage_bps": float(max(slippage_bps, 0.0)),
            "depth_needed_bps": float(max(depth_needed_bps, 0.0)),
            "filled_fraction": float(filled_fraction),
        }

    def _book_slope_bps_per_level(self, levels: List[Tuple[float, float]], mid: float, max_levels: int, is_bid: bool) -> float:
        if mid <= 0.0:
            return 0.0
        y: List[float] = []
        for price, size in levels:
            p = float(price)
            s = float(size)
            if p <= 0.0 or s <= 0.0:
                continue
            dist = (1e4 * (mid - p) / mid) if is_bid else (1e4 * (p - mid) / mid)
            if dist < 0.0:
                continue
            y.append(float(dist))
            if len(y) >= int(max_levels):
                break
        if len(y) < 2:
            return 0.0
        x = list(range(len(y)))
        return float(self._lin_slope(x, y))

    def _book_convexity_within_bps(self, levels, mid, band_bps, is_bid) -> float:
        eps = 1e-12
        near_band = min(2.0, float(band_bps) / 2.0)
        near = self._depth_within_bps(levels, mid, near_band, is_bid)["size"]
        total = self._depth_within_bps(levels, mid, float(band_bps), is_bid)["size"]
        return 0.0 if total <= eps else float(near / total)

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

    def event_density_1000ms(self) -> float:
        return self.event_density(1_000)

    def event_density_3000ms(self) -> float:
        return self.event_density(3_000)

    def event_density_7500ms(self) -> float:
        return self.event_density(7_500)

    def event_density_15000ms(self) -> float:
        return self.event_density(15_000)

    def event_density_30000ms(self) -> float:
        return self.event_density(30_000)

    def event_density_60000ms(self) -> float:
        return self.event_density(60_000)

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
        any_event_dt_ms = (
            1.0
            if self._last_any_event_ts is None
            else max(1.0, float(ts_ms - int(self._last_any_event_ts)))
        )
        prev_bid_l1 = self.prev_bsz
        prev_ask_l1 = self.prev_asz
        prev_bid_l2 = self.prev_bsz2
        prev_ask_l2 = self.prev_asz2

        is_trade = (etype == 'trade')
        for w, deq in self._event_density_deques.items():
            self._append_ts_with_guard(deq, ts_ms, w, is_ob_event=(etype == 'ob'))

        if etype == 'trade':
            self._update_trade_windows(ts_ms, payload, any_event_dt_ms)
            self.trade_fast_path_count += 1

            # Trade rows update trade/event-density state only.
            # They must not build feature vectors, update feature z-score state,
            # append price-history rows, update book EMAs, or call _zscore().
            self.last_ts = ts_ms
            self._last_any_event_ts = int(ts_ms)

            # Mid is ignored by offline_ingest for trades, but return a best-effort value
            # for API consistency.
            bid1, ask1, _, _ = self._book_best()
            mid = 0.5 * (bid1 + ask1) if bid1 > 0.0 and ask1 > 0.0 else 0.0
            return ts_ms, self._trade_fast_path_empty_feature, mid, True, any_event_dt_ms

        if etype == 'ob':
            self._prune_replen_windows(ts_ms)
            for window in self._trade_window_deques:
                self._prune_trade_window(ts_ms, window)
            for window in self.trade_burst_states:
                self._trade_burst_prune(window, ts_ms)
            tp_code, bids, asks = payload
            self._update_book_from_ob(tp_code, bids, asks)
            for window, deq in self._quote_window_deques.items():
                self._append_ts_with_guard(deq, ts_ms, window, is_ob_event=True)
        else:
            raise ValueError(f"Unsupported event type in _dispatch_parsed_event: {etype!r}")
        self._ensure_book_ladders()
        session_features = self._compute_session_features(ts_ms)
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

        micro_premia = (micro - mid) / max(spread, 1e-12)
        micro_minus_mid_bps = 1e4 * (micro / mid - 1.0) if mid > 0.0 and micro > 0.0 else 0.0
        micro_minus_mid_over_spread = (micro - mid) / max(spread, 1e-12)
        if self._last_ob_feature_ts is None:
            ob_dt_ms = 1.0
        else:
            ob_dt_ms = max(1.0, float(ts_ms - int(self._last_ob_feature_ts)))

        xb, yb = self._levels_to_xy(self.bid_lvls, mid, True, 5)
        xa, ya = self._levels_to_xy(self.ask_lvls, mid, False, 5)
        slope_b = self._lin_slope(xb, yb)
        slope_a = self._lin_slope(xa, ya)

        for ms in FAST_WINDOWS_MS:
            self.ofi_pressure_by_window[ms] = self._ewma_update(self.ofi_pressure_by_window[ms], ofi_l1, ob_dt_ms, ms)
        ofi_pressure_by_ms = {ms: self.ofi_pressure_by_window[ms] for ms in FAST_WINDOWS_MS}
        trade_stats_by_ms = {ms: self._compute_trade_window_stats(ms, ts_ms, mid, micro) for ms in self.trade_windows}
        if etype == "ob":
            self.last_ob_ofi_l5 = float(ofi_l5)
            self.last_ob_trade_imbalance_30000ms = float(trade_stats_by_ms[30_000]["trade_imbalance_notional"])
        cvd_stats_by_ms: Dict[int, Dict[str, float]] = {}
        for ms in self.trade_windows:
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
        large_stats_by_ms = {ms: self._large_trade_stats_from_state(ms, ts_ms) for ms in self.trade_windows}
        trade_burst_features = self._trade_burst_features_from_state(ts_ms)
        if self.prev_bid1_price is None or bid1 != self.prev_bid1_price:
            self.last_bid_price_change_ts = ts_ms
        if self.prev_ask1_price is None or ask1 != self.prev_ask1_price:
            self.last_ask_price_change_ts = ts_ms
        if self.prev_mid_price_for_age is None or mid != self.prev_mid_price_for_age:
            self.last_mid_change_ts = ts_ms
        if self.prev_spread_for_age is None:
            self.last_spread_widen_ts = ts_ms
            self.last_spread_tighten_ts = ts_ms
        else:
            if spread > self.prev_spread_for_age:
                self.last_spread_widen_ts = ts_ms
            if spread < self.prev_spread_for_age:
                self.last_spread_tighten_ts = ts_ms

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
        time_since_bid_price_change_ms = float(ts_ms - self.last_bid_price_change_ts) if self.last_bid_price_change_ts is not None else 0.0
        time_since_ask_price_change_ms = float(ts_ms - self.last_ask_price_change_ts) if self.last_ask_price_change_ts is not None else 0.0
        time_since_mid_change_ms = float(ts_ms - self.last_mid_change_ts) if self.last_mid_change_ts is not None else 0.0
        time_since_spread_widen_ms = float(ts_ms - self.last_spread_widen_ts) if self.last_spread_widen_ts is not None else 0.0
        time_since_spread_tighten_ms = float(ts_ms - self.last_spread_tighten_ts) if self.last_spread_tighten_ts is not None else 0.0
        best_bid_lifetime_ms = time_since_bid_price_change_ms
        best_ask_lifetime_ms = time_since_ask_price_change_ms
        mid_price_staleness_ms = time_since_mid_change_ms

        def _ofi_since(ts0: Optional[int]) -> float:
            if ts0 is None:
                return 0.0
            return float(sum(v for t, v in self.ofi_level_histories[5] if ts0 <= t <= ts_ms))

        def _trade_imb_since(ts0: Optional[int]) -> float:
            if ts0 is None:
                return 0.0
            entries = [(s, n) for t, s, n in self.trade_sign_history if ts0 <= t <= ts_ms and s != 0]
            if not entries:
                return 0.0
            signed = sum(float(s) * float(n) for s, n in entries)
            total = sum(float(n) for _, n in entries)
            return self._safe_div(signed, total, 0.0)

        large_trade_cont_features = {
            "last_large_buy_notional_usd": float(self.last_large_buy_notional_usd),
            "last_large_sell_notional_usd": float(self.last_large_sell_notional_usd),
            "return_since_last_large_buy_bps": self._bps(mid, self.last_large_buy_mid or 0.0, 0.0),
            "return_since_last_large_sell_bps": self._bps(self.last_large_sell_mid or 0.0, mid, 0.0),
            "ofi_l5_since_last_large_buy": _ofi_since(self.last_large_buy_ts),
            "ofi_l5_since_last_large_sell": -_ofi_since(self.last_large_sell_ts),
            "trade_imbalance_since_last_large_buy": _trade_imb_since(self.last_large_buy_ts),
            "trade_imbalance_since_last_large_sell": -_trade_imb_since(self.last_large_sell_ts),
            "large_buy_continuation_bps_7500ms": self._bps(mid, self.last_large_buy_mid or 0.0, 0.0)
            if (self.last_large_buy_ts is not None and (ts_ms - self.last_large_buy_ts) <= 7_500) else 0.0,
            "large_buy_continuation_bps_15000ms": self._bps(mid, self.last_large_buy_mid or 0.0, 0.0)
            if (self.last_large_buy_ts is not None and (ts_ms - self.last_large_buy_ts) <= 15_000) else 0.0,
            "large_sell_continuation_bps_7500ms": self._bps(self.last_large_sell_mid or 0.0, mid, 0.0)
            if (self.last_large_sell_ts is not None and (ts_ms - self.last_large_sell_ts) <= 7_500) else 0.0,
            "large_sell_continuation_bps_15000ms": self._bps(self.last_large_sell_mid or 0.0, mid, 0.0)
            if (self.last_large_sell_ts is not None and (ts_ms - self.last_large_sell_ts) <= 15_000) else 0.0,
        }
        self._add_return(ts_ms, mid, ob_dt_ms)
        return_var = {ms: stats.mean_var()[1] for ms, stats in self.return_histories.items()}
        return_std_bps = {ms: math.sqrt(var) for ms, var in return_var.items()}
        variance_ratio_adjacent = {}
        for prev_ms, cur_ms in zip(self.return_windows_ms[:-1], self.return_windows_ms[1:]):
            var_prev = return_var[prev_ms]
            var_cur = return_var[cur_ms]
            variance_ratio_adjacent[(cur_ms, prev_ms)] = (var_cur / max((cur_ms / prev_ms) * var_prev, 1e-12)) if var_prev > 0 else 0.0

        regime_vol_ewma = {ms: math.sqrt(max(self.rv_ewma[ms], 1e-18)) for ms in self.regime_windows_ms}
        regime_realized = {ms: self.realized_vol[ms] for ms in self.regime_windows_ms}
        regime_volume = {ms: self.volume_ewma[ms] for ms in self.regime_windows_ms}
        for ms in self.regime_windows_ms:
            if ms <= 60_000:
                self.flow_regime[ms] = trade_stats_by_ms[ms]["trade_imbalance_notional"]
        regime_flow_snapshot = {ms: self.flow_regime[ms] for ms in self.regime_windows_ms}

        vpin_features = []
        for secs in self.vpin_bucket_secs:
            phi = self.vpin_state[secs]["phi"]
            vpin_features.append((sum(phi) / len(phi)) if phi else 0.0)

        replen_rates = self._replenishment_rates()
        price_features_by_window: Dict[int, Tuple[float, ...]] = {}
        for w in PRICE_WINDOWS_MS:
            past_mid = self._mid_asof_history.asof(ts_ms - w)
            past_micro = self._micro_asof_history.asof(ts_ms - w)
            mid_ret_bps = self._bps_return(mid, past_mid) if past_mid is not None else 0.0
            micro_ret_bps = self._bps_return(micro, past_micro) if past_micro is not None else 0.0
            state = self._price_window_mid_states[w]
            state.prune(ts_ms)  # critical: prune to current row timestamp before reading
            slope, r2 = state.slope_r2()
            pos, d_high, d_low, rng, br_up, br_down = state.range_features(mid)
            sign_persistence, up_frac, autocorr = state.return_shape_features()
            price_features_by_window[w] = (
                mid_ret_bps, micro_ret_bps, slope, r2,
                pos, d_high, d_low, rng, br_up, br_down,
                sign_persistence, up_frac, autocorr,
            )
        bid_depth_5bps = self._depth_within_bps(self.bid_lvls, mid, 5.0, is_bid=True)
        ask_depth_5bps = self._depth_within_bps(self.ask_lvls, mid, 5.0, is_bid=False)
        depth_5bps_total = bid_depth_5bps["size"] + ask_depth_5bps["size"]
        depth_5bps_imbalance = self._safe_div(
            bid_depth_5bps["size"] - ask_depth_5bps["size"],
            bid_depth_5bps["size"] + ask_depth_5bps["size"],
            0.0,
        )
        self._append_metric_history(self._spread_bps_history, ts_ms, spread_bps, self._regime_metric_keep_ms)
        self._append_metric_history(self._bid_depth_5bps_history, ts_ms, bid_depth_5bps["size"], self._regime_metric_keep_ms)
        self._append_metric_history(self._ask_depth_5bps_history, ts_ms, ask_depth_5bps["size"], self._regime_metric_keep_ms)
        self._append_metric_history(self._depth_5bps_total_history, ts_ms, depth_5bps_total, self._regime_metric_keep_ms)
        self._append_metric_history(self._depth_5bps_imbalance_history, ts_ms, depth_5bps_imbalance, self._regime_metric_keep_ms)
        for ms in SPREAD_DEPTH_REGIME_WINDOWS_MS:
            self._spread_bps_regime_states[ms].update(ts_ms, spread_bps)
            self._bid_depth_5bps_regime_states[ms].update(ts_ms, bid_depth_5bps["size"])
            self._ask_depth_5bps_regime_states[ms].update(ts_ms, ask_depth_5bps["size"])
            self._depth_5bps_total_regime_states[ms].update(ts_ms, depth_5bps_total)
            self._depth_5bps_imbalance_regime_states[ms].update(ts_ms, depth_5bps_imbalance)
        band_depth_stats: Dict[float, Dict[str, dict]] = {}
        for band in BPS_DEPTH_BANDS:
            band_depth_stats[band] = {
                "bid": self._depth_within_bps(self.bid_lvls, mid, band, is_bid=True),
                "ask": self._depth_within_bps(self.ask_lvls, mid, band, is_bid=False),
            }
        shape_stats: Dict[float, Dict[str, dict]] = {}
        for band in BOOK_SHAPE_BANDS:
            shape_stats[band] = {
                "bid": self._depth_within_bps(self.bid_lvls, mid, band, is_bid=True),
                "ask": self._depth_within_bps(self.ask_lvls, mid, band, is_bid=False),
            }
        slippage_by_notional: Dict[float, Dict[str, dict]] = {}
        for notional in SLIPPAGE_NOTIONAL_USD:
            slippage_by_notional[notional] = {
                "buy": self._slippage_for_notional(self.ask_lvls, mid, notional, is_buy=True),
                "sell": self._slippage_for_notional(self.bid_lvls, mid, notional, is_buy=False),
            }
        rolling_ofi_sums: Dict[Tuple[int, int], float] = {}
        rolling_obi_stats: Dict[Tuple[int, int], Tuple[float, float, float]] = {}
        if etype == "ob":
            for level in ROLLING_OFI_LEVELS:
                value = ofi_by_level[level]
                self._append_metric_history(self.ofi_level_histories[level], ts_ms, value, keep_ms=120_000)
                for window in ROLLING_OFI_WINDOWS_MS:
                    self._rolling_ofi_states[(level, window)].update(ts_ms, value)
            for level in ROLLING_OBI_LEVELS:
                value = obi_by_level[level]
                self._append_metric_history(self.obi_level_histories[level], ts_ms, value, keep_ms=120_000)
                for window in ROLLING_OBI_WINDOWS_MS:
                    self._rolling_obi_states[(level, window)].update(ts_ms, value)
        for level in ROLLING_OFI_LEVELS:
            for window in ROLLING_OFI_WINDOWS_MS:
                rolling_ofi_sums[(level, window)] = self._rolling_ofi_states[(level, window)].sum_value()
        for level in ROLLING_OBI_LEVELS:
            current_sign = self._sign(obi_by_level[level])
            for window in ROLLING_OBI_WINDOWS_MS:
                state = self._rolling_obi_states[(level, window)]
                mean_val = state.mean()
                slope = state.slope()
                persistence = state.persistence(current_sign)
                rolling_obi_stats[(level, window)] = (mean_val, slope, persistence)
        deep_micro_features: Dict[str, float] = {}
        deep_micro_minus_mid_bps: Dict[int, float] = {}
        micro_price_by_level: Dict[int, float] = {}
        for level in DEEP_MICRO_LEVELS:
            bid_px_n, bid_qty_n = self._weighted_side_price(self.bid_lvls, level)
            ask_px_n, ask_qty_n = self._weighted_side_price(self.ask_lvls, level)
            den = bid_qty_n + ask_qty_n
            if den > 1e-12:
                micro_l = (ask_px_n * bid_qty_n + bid_px_n * ask_qty_n) / den
                # VAMP convention for stage4_v4: same-side weighted average
                vamp_l = (bid_px_n * bid_qty_n + ask_px_n * ask_qty_n) / den
            else:
                micro_l = 0.0
                vamp_l = 0.0
            micro_price_by_level[level] = float(micro_l)
            deep_micro_minus_mid_bps[level] = self._bps(micro_l, mid, 0.0)
            deep_micro_features[f"micro_l{level}_minus_mid_bps"] = deep_micro_minus_mid_bps[level]
            deep_micro_features[f"micro_l{level}_minus_mid_over_spread"] = self._safe_div(micro_l - mid, max(spread, 1e-12), 0.0)
            deep_micro_features[f"vamp_l{level}_minus_mid_bps"] = self._bps(vamp_l, mid, 0.0)
        if etype == "ob":
            self._append_metric_history(self.deep_micro_histories[5], ts_ms, deep_micro_minus_mid_bps.get(5, 0.0), keep_ms=120_000)
            self._append_metric_history(self.deep_micro_histories[10], ts_ms, deep_micro_minus_mid_bps.get(10, 0.0), keep_ms=120_000)
        for level in (5, 10):
            for window in (7_500, 30_000):
                points = self._metric_values(self.deep_micro_histories[level], ts_ms, window)
                xs = [(t - ts_ms) / 1000.0 for t, _ in points]
                ys = [v for _, v in points]
                deep_micro_features[f"micro_l{level}_slope_{window}ms"] = self._lin_slope(xs, ys) if len(ys) >= 2 else 0.0
        deep_micro_features["micro_l1_minus_micro_l10_bps"] = self._bps(micro, micro_price_by_level.get(10, 0.0), 0.0)
        deep_micro_features["micro_l5_minus_micro_l20_bps"] = self._bps(micro_price_by_level.get(5, 0.0), micro_price_by_level.get(20, 0.0), 0.0)
        slippage_asymmetry_features: Dict[str, float] = {}
        notional_map = {int(n): n for n in SLIPPAGE_NOTIONAL_USD}
        for notional in SLIPPAGE_NOTIONAL_USD:
            n = self._fmt_usd_notional(notional)
            buy = slippage_by_notional[notional]["buy"]
            sell = slippage_by_notional[notional]["sell"]
            slippage_asymmetry_features[f"slippage_imbalance_bps_{n}"] = float(sell["slippage_bps"] - buy["slippage_bps"])
            slippage_asymmetry_features[f"depth_needed_imbalance_bps_{n}"] = float(buy["depth_needed_bps"] - sell["depth_needed_bps"])
            slippage_asymmetry_features[f"filled_fraction_imbalance_{n}"] = float(buy["filled_fraction"] - sell["filled_fraction"])
        buy_sl_10 = slippage_by_notional[notional_map[10_000]]["buy"]["slippage_bps"]
        buy_sl_50 = slippage_by_notional[notional_map[50_000]]["buy"]["slippage_bps"]
        buy_sl_250 = slippage_by_notional[notional_map[250_000]]["buy"]["slippage_bps"]
        sell_sl_10 = slippage_by_notional[notional_map[10_000]]["sell"]["slippage_bps"]
        sell_sl_50 = slippage_by_notional[notional_map[50_000]]["sell"]["slippage_bps"]
        sell_sl_250 = slippage_by_notional[notional_map[250_000]]["sell"]["slippage_bps"]
        buy_slope_10_50 = self._safe_div(buy_sl_50 - buy_sl_10, 40_000.0, 0.0)
        sell_slope_10_50 = self._safe_div(sell_sl_50 - sell_sl_10, 40_000.0, 0.0)
        buy_slope_50_250 = self._safe_div(buy_sl_250 - buy_sl_50, 200_000.0, 0.0)
        sell_slope_50_250 = self._safe_div(sell_sl_250 - sell_sl_50, 200_000.0, 0.0)
        slippage_asymmetry_features.update({
            "buy_slippage_slope_10000_to_50000usd": buy_slope_10_50,
            "sell_slippage_slope_10000_to_50000usd": sell_slope_10_50,
            "buy_slippage_slope_50000_to_250000usd": buy_slope_50_250,
            "sell_slippage_slope_50000_to_250000usd": sell_slope_50_250,
            "slippage_curve_convexity_buy": buy_slope_50_250 - buy_slope_10_50,
            "slippage_curve_convexity_sell": sell_slope_50_250 - sell_slope_10_50,
        })
        indicator_values = {
            "spread_bps": spread_bps,
            "gap_a_bps": gap_a_bps,
            "gap_b_bps": gap_b_bps,
            "micro_premia": micro_premia,
            "micro_minus_mid_bps": micro_minus_mid_bps,
            "depth_imbalance_within_1bps": self._safe_div(
                band_depth_stats[1.0]["bid"]["size"] - band_depth_stats[1.0]["ask"]["size"],
                band_depth_stats[1.0]["bid"]["size"] + band_depth_stats[1.0]["ask"]["size"],
                0.0,
            ),
            "depth_imbalance_within_2bps": self._safe_div(
                band_depth_stats[2.0]["bid"]["size"] - band_depth_stats[2.0]["ask"]["size"],
                band_depth_stats[2.0]["bid"]["size"] + band_depth_stats[2.0]["ask"]["size"],
                0.0,
            ),
            "depth_imbalance_within_5bps": self._safe_div(
                band_depth_stats[5.0]["bid"]["size"] - band_depth_stats[5.0]["ask"]["size"],
                band_depth_stats[5.0]["bid"]["size"] + band_depth_stats[5.0]["ask"]["size"],
                0.0,
            ),
            "depth_imbalance_within_10bps": self._safe_div(
                band_depth_stats[10.0]["bid"]["size"] - band_depth_stats[10.0]["ask"]["size"],
                band_depth_stats[10.0]["bid"]["size"] + band_depth_stats[10.0]["ask"]["size"],
                0.0,
            ),
            "notional_imbalance_within_1bps": self._safe_div(
                band_depth_stats[1.0]["bid"]["notional"] - band_depth_stats[1.0]["ask"]["notional"],
                band_depth_stats[1.0]["bid"]["notional"] + band_depth_stats[1.0]["ask"]["notional"],
                0.0,
            ),
            "notional_imbalance_within_2bps": self._safe_div(
                band_depth_stats[2.0]["bid"]["notional"] - band_depth_stats[2.0]["ask"]["notional"],
                band_depth_stats[2.0]["bid"]["notional"] + band_depth_stats[2.0]["ask"]["notional"],
                0.0,
            ),
            "notional_imbalance_within_5bps": self._safe_div(
                band_depth_stats[5.0]["bid"]["notional"] - band_depth_stats[5.0]["ask"]["notional"],
                band_depth_stats[5.0]["bid"]["notional"] + band_depth_stats[5.0]["ask"]["notional"],
                0.0,
            ),
            "notional_imbalance_within_10bps": self._safe_div(
                band_depth_stats[10.0]["bid"]["notional"] - band_depth_stats[10.0]["ask"]["notional"],
                band_depth_stats[10.0]["bid"]["notional"] + band_depth_stats[10.0]["ask"]["notional"],
                0.0,
            ),
            "obi_l1": obi_by_level[1],
            "obi_l3": obi_by_level[3],
            "obi_l5": obi_by_level[5],
            "obi_l10": obi_by_level[10],
            "ofi_l1_over_depth_l1": self._safe_div(ofi_by_level[1], cum_bid_by_level[1] + cum_ask_by_level[1], 0.0),
            "ofi_l3_over_depth_l3": self._safe_div(ofi_by_level[3], cum_bid_by_level[3] + cum_ask_by_level[3], 0.0),
            "ofi_l5_over_depth_l5": self._safe_div(ofi_by_level[5], cum_bid_by_level[5] + cum_ask_by_level[5], 0.0),
            "ofi_l10_over_depth_l10": self._safe_div(ofi_by_level[10], cum_bid_by_level[10] + cum_ask_by_level[10], 0.0),
            "signed_notional_flow_usd_30000ms": trade_stats_by_ms[30_000]["signed_notional_flow_usd"],
            "trade_imbalance_notional_30000ms": trade_stats_by_ms[30_000]["trade_imbalance_notional"],
            "vwap_vs_mid_bps_30000ms": trade_stats_by_ms[30_000]["vwap_vs_mid_bps"],
        }
        for required_name in self.ema_indicator_names:
            if required_name not in indicator_values:
                raise ValueError(f"Missing EMA indicator value for {required_name}")
        self._update_indicator_emas(indicator_values, ob_dt_ms)

        bid_5bps_levels = [(p, s) for p, s in self.bid_lvls if p > 0.0 and s > 0.0 and mid > 0.0 and p <= mid and (1e4 * (mid - p) / mid) <= 5.0]
        ask_5bps_levels = [(p, s) for p, s in self.ask_lvls if p > 0.0 and s > 0.0 and mid > 0.0 and p >= mid and (1e4 * (p - mid) / mid) <= 5.0]
        feat_list: List[float] = []
        feat_list.extend([
            session_features["time_hour_sin"],
            session_features["time_hour_cos"],
            session_features["time_dow_sin"],
            session_features["time_dow_cos"],
            session_features["session_is_weekend"],
            session_features["session_is_asia"],
            session_features["session_is_europe"],
            session_features["session_is_us"],
            session_features["session_is_europe_us_overlap"],
        ])
        for w in PRICE_WINDOWS_MS:
            feat_list.extend(price_features_by_window[w])
        feat_list.extend([
            spread_bps, gap_a_bps, gap_b_bps, bsz1, asz1, micro_premia,
            micro_minus_mid_bps, micro_minus_mid_over_spread,
            dt_since_trade,
            time_since_bid_price_change_ms,
            time_since_ask_price_change_ms,
            time_since_mid_change_ms,
            time_since_spread_widen_ms,
            time_since_spread_tighten_ms,
            best_bid_lifetime_ms,
            best_ask_lifetime_ms,
            mid_price_staleness_ms,
        ])
        for lvl in BOOK_DEPTH_FEATURE_LEVELS:
            feat_list.extend([cum_bid_by_level[lvl], cum_ask_by_level[lvl]])
        for lvl in BOOK_DEPTH_FEATURE_LEVELS:
            feat_list.append(obi_by_level[lvl])
        for lvl in BOOK_DEPTH_FEATURE_LEVELS:
            feat_list.append(ofi_by_level[lvl])
        for lvl in NORMALIZED_OFI_LEVELS:
            ofi_val = ofi_by_level[lvl]
            level_depth = cum_bid_by_level[lvl] + cum_ask_by_level[lvl]
            feat_list.append(self._safe_div(ofi_val, level_depth, 0.0))
            feat_list.append(self._safe_div(ofi_val, max(spread_bps, 0.1), 0.0))
            feat_list.append(self._safe_div(ofi_val, depth_5bps_total, 0.0))
        for level in ROLLING_OFI_LEVELS:
            depth_l = cum_bid_by_level[level] + cum_ask_by_level[level]
            for window in ROLLING_OFI_WINDOWS_MS:
                ofi_sum = rolling_ofi_sums[(level, window)]
                feat_list.append(ofi_sum)
                feat_list.append(self._safe_div(ofi_sum, depth_l, 0.0))
        for level in ROLLING_OFI_LEVELS:
            feat_list.append(rolling_ofi_sums[(level, 7_500)] - rolling_ofi_sums[(level, 30_000)])
            feat_list.append(rolling_ofi_sums[(level, 15_000)] - rolling_ofi_sums[(level, 60_000)])
        for level in ROLLING_OBI_LEVELS:
            for window in ROLLING_OBI_WINDOWS_MS:
                mean_val, slope, persistence = rolling_obi_stats[(level, window)]
                feat_list.extend([mean_val, slope, persistence])
        for level in DEEP_MICRO_LEVELS:
            feat_list.extend([
                deep_micro_features[f"micro_l{level}_minus_mid_bps"],
                deep_micro_features[f"micro_l{level}_minus_mid_over_spread"],
                deep_micro_features[f"vamp_l{level}_minus_mid_bps"],
            ])
        feat_list.extend([
            deep_micro_features["micro_l5_slope_7500ms"],
            deep_micro_features["micro_l5_slope_30000ms"],
            deep_micro_features["micro_l10_slope_7500ms"],
            deep_micro_features["micro_l10_slope_30000ms"],
            deep_micro_features["micro_l1_minus_micro_l10_bps"],
            deep_micro_features["micro_l5_minus_micro_l20_bps"],
        ])
        for band in BPS_DEPTH_BANDS:
            bid_stats = band_depth_stats[band]["bid"]
            ask_stats = band_depth_stats[band]["ask"]
            bid_size = bid_stats["size"]
            ask_size = ask_stats["size"]
            bid_notional = bid_stats["notional"]
            ask_notional = ask_stats["notional"]
            feat_list.extend([
                bid_size,
                ask_size,
                bid_notional,
                ask_notional,
                self._safe_div(bid_size - ask_size, bid_size + ask_size, 0.0),
                self._safe_div(bid_notional - ask_notional, bid_notional + ask_notional, 0.0),
            ])
        for band in BOOK_SHAPE_BANDS:
            bid_stats = shape_stats[band]["bid"]
            ask_stats = shape_stats[band]["ask"]
            feat_list.extend([
                bid_stats["max_size"],
                ask_stats["max_size"],
                bid_stats["max_notional"],
                ask_stats["max_notional"],
                bid_stats["dist_to_max_bps"],
                ask_stats["dist_to_max_bps"],
                bid_stats["hhi"],
                ask_stats["hhi"],
                bid_stats["top1_share"],
                ask_stats["top1_share"],
            ])
        feat_list.extend([
            self._book_slope_bps_per_level(self.bid_lvls, mid, 5, True),
            self._book_slope_bps_per_level(self.ask_lvls, mid, 5, False),
            self._book_slope_bps_per_level(bid_5bps_levels, mid, len(bid_5bps_levels), True),
            self._book_slope_bps_per_level(ask_5bps_levels, mid, len(ask_5bps_levels), False),
            self._book_convexity_within_bps(self.bid_lvls, mid, 10.0, True),
            self._book_convexity_within_bps(self.ask_lvls, mid, 10.0, False),
        ])
        for notional in SLIPPAGE_NOTIONAL_USD:
            buy = slippage_by_notional[notional]["buy"]
            sell = slippage_by_notional[notional]["sell"]
            feat_list.extend([
                buy["slippage_bps"],
                sell["slippage_bps"],
                buy["depth_needed_bps"],
                sell["depth_needed_bps"],
                buy["filled_fraction"],
                sell["filled_fraction"],
            ])
        for notional in SLIPPAGE_NOTIONAL_USD:
            n = self._fmt_usd_notional(notional)
            feat_list.extend([
                slippage_asymmetry_features[f"slippage_imbalance_bps_{n}"],
                slippage_asymmetry_features[f"depth_needed_imbalance_bps_{n}"],
                slippage_asymmetry_features[f"filled_fraction_imbalance_{n}"],
            ])
        feat_list.extend([
            slippage_asymmetry_features["buy_slippage_slope_10000_to_50000usd"],
            slippage_asymmetry_features["sell_slippage_slope_10000_to_50000usd"],
            slippage_asymmetry_features["buy_slippage_slope_50000_to_250000usd"],
            slippage_asymmetry_features["sell_slippage_slope_50000_to_250000usd"],
            slippage_asymmetry_features["slippage_curve_convexity_buy"],
            slippage_asymmetry_features["slippage_curve_convexity_sell"],
        ])

        for ms in FAST_WINDOWS_MS:
            window_seconds = max(ms / 1000.0, 1e-9)
            bid_price_change_count = float(len(self._bid_price_change_deques[ms]))
            ask_price_change_count = float(len(self._ask_price_change_deques[ms]))
            bid_l1_depletion = l1_depletion[ms][0]
            ask_l1_depletion = l1_depletion[ms][1]
            feat_list.extend([
                spread_delta_bps[ms],
                float(len(self._spread_change_deques[ms])),
                bid_price_change_count,
                ask_price_change_count,
                bid_price_change_count / window_seconds,
                ask_price_change_count / window_seconds,
                bid_l1_depletion,
                ask_l1_depletion,
                self._safe_div(bid_l1_depletion, max(bsz1, 1e-9), 0.0),
                self._safe_div(ask_l1_depletion, max(asz1, 1e-9), 0.0),
            ])
        for ms in FAST_WINDOWS_MS:
            rates = replen_rates[ms]
            for level in (1, 2):
                bid_level_size = bsz1 if level == 1 else bsz2
                ask_level_size = asz1 if level == 1 else asz2
                bid_add_rate = rates[("bid", level, "add")] * 1000.0
                bid_rem_rate = rates[("bid", level, "rem")] * 1000.0
                ask_add_rate = rates[("ask", level, "add")] * 1000.0
                ask_rem_rate = rates[("ask", level, "rem")] * 1000.0
                feat_list.extend([
                    bid_add_rate,
                    bid_rem_rate,
                    ask_add_rate,
                    ask_rem_rate,
                    self._safe_div(bid_add_rate, max(bid_level_size, 1e-9), 0.0),
                    self._safe_div(bid_rem_rate, max(bid_level_size, 1e-9), 0.0),
                    self._safe_div(ask_add_rate, max(ask_level_size, 1e-9), 0.0),
                    self._safe_div(ask_rem_rate, max(ask_level_size, 1e-9), 0.0),
                ])

        for ms in FLOW_WINDOWS_MS:
            s = trade_stats_by_ms[ms]
            feat_list.extend([
                s["buy_vol_base"],
                s["sell_vol_base"],
                s["buy_notional_usd"],
                s["sell_notional_usd"],
                s["buy_count"],
                s["sell_count"],
                s["buy_mean_notional_usd"],
                s["sell_mean_notional_usd"],
                s["buy_max_notional_usd"],
                s["sell_max_notional_usd"],
                s["signed_notional_flow_usd"],
                s["signed_trade_count_imbalance"],
                s["trade_imbalance_notional"],
                s["trade_toxicity_notional"],
                s["plus_tick_fraction"],
                s["minus_tick_fraction"],
                s["zero_tick_fraction"],
                s["tick_sign_imbalance"],
                s["trade_count"],
                s["trade_count_per_second"],
                s["vwap_vs_mid_bps"],
                s["vwap_vs_micro_bps"],
                s["signed_trade_premium_bps_count_weighted"],
                s["signed_trade_premium_bps_volume_weighted"],
                s["buy_trade_premium_bps"],
                s["sell_trade_premium_bps"],
                s["aggressor_price_impact_bps"],
            ])
        for ms in FLOW_WINDOWS_MS:
            c = cvd_stats_by_ms[ms]
            feat_list.extend([
                c["cvd_change_usd"],
                c["cvd_slope_usd_per_sec"],
                c["cvd_minus_ema_usd"],
            ])
        feat_list.extend([
            trade_stats_by_ms[1_000]["trade_imbalance_notional"] - trade_stats_by_ms[7_500]["trade_imbalance_notional"],
            trade_stats_by_ms[3_000]["trade_imbalance_notional"] - trade_stats_by_ms[15_000]["trade_imbalance_notional"],
            trade_stats_by_ms[7_500]["trade_imbalance_notional"] - trade_stats_by_ms[30_000]["trade_imbalance_notional"],
            (trade_stats_by_ms[3_000]["signed_notional_flow_usd"] / 100_000.0) - (trade_stats_by_ms[30_000]["signed_notional_flow_usd"] / 100_000.0),
            self.ofi_pressure_by_window[3_000] - self.ofi_pressure_by_window[15_000],
            self.ofi_pressure_by_window[7_500] - self.ofi_pressure_by_window[30_000],
            trade_burst_features["consecutive_buy_trade_count"],
            trade_burst_features["consecutive_sell_trade_count"],
        ])
        for window in TRADE_BURST_WINDOWS_MS:
            feat_list.extend([
                trade_burst_features[f"max_buy_run_length_{window}ms"],
                trade_burst_features[f"max_sell_run_length_{window}ms"],
                trade_burst_features[f"trade_sign_autocorr_lag1_{window}ms"],
                trade_burst_features[f"trade_sign_entropy_{window}ms"],
                trade_burst_features[f"trade_burst_score_{window}ms"],
                trade_burst_features[f"buy_trade_burst_score_{window}ms"],
                trade_burst_features[f"sell_trade_burst_score_{window}ms"],
            ])
        for ms in FLOW_WINDOWS_MS:
            large_stats = large_stats_by_ms[ms]
            for threshold in LARGE_TRADE_NOTIONAL_USD:
                thr = self._fmt_usd_notional(threshold)
                feat_list.extend([
                    large_stats[f"large_buy_count_ge_{thr}_{ms}ms"],
                    large_stats[f"large_sell_count_ge_{thr}_{ms}ms"],
                    large_stats[f"large_buy_notional_ge_{thr}_{ms}ms"],
                    large_stats[f"large_sell_notional_ge_{thr}_{ms}ms"],
                    large_stats[f"large_trade_imbalance_ge_{thr}_{ms}ms"],
                ])
            feat_list.extend([
                large_stats[f"max_signed_trade_notional_usd_{ms}ms"],
                large_stats[f"top5_trade_notional_sum_usd_{ms}ms"],
                large_stats[f"large_trade_cluster_count_{ms}ms"],
            ])
        feat_list.extend([
            float(ts_ms - self.last_large_buy_ts) if self.last_large_buy_ts is not None else 0.0,
            float(ts_ms - self.last_large_sell_ts) if self.last_large_sell_ts is not None else 0.0,
            large_trade_cont_features["last_large_buy_notional_usd"],
            large_trade_cont_features["last_large_sell_notional_usd"],
            large_trade_cont_features["return_since_last_large_buy_bps"],
            large_trade_cont_features["return_since_last_large_sell_bps"],
            large_trade_cont_features["ofi_l5_since_last_large_buy"],
            large_trade_cont_features["ofi_l5_since_last_large_sell"],
            large_trade_cont_features["trade_imbalance_since_last_large_buy"],
            large_trade_cont_features["trade_imbalance_since_last_large_sell"],
            large_trade_cont_features["large_buy_continuation_bps_7500ms"],
            large_trade_cont_features["large_buy_continuation_bps_15000ms"],
            large_trade_cont_features["large_sell_continuation_bps_7500ms"],
            large_trade_cont_features["large_sell_continuation_bps_15000ms"],
        ])
        for ms in FLOW_WINDOWS_MS:
            s = trade_stats_by_ms[ms]
            mid_ret_bps = price_features_by_window[ms][0] if ms in price_features_by_window else (
                self._bps_return(mid, self._series_asof(ts_ms - ms, "mid")) if self._series_asof(ts_ms - ms, "mid") is not None else 0.0
            )
            buy_notional_scaled = s["buy_notional_usd"] / 100_000.0
            sell_notional_scaled = s["sell_notional_usd"] / 100_000.0
            signed_notional_scaled = s["signed_notional_flow_usd"] / 100_000.0
            buy_flow_without_price_up = buy_notional_scaled * math.exp(max(-max(mid_ret_bps, 0.0), -50.0))
            sell_flow_without_price_down = sell_notional_scaled * math.exp(max(-max(-mid_ret_bps, 0.0), -50.0))
            absorption_ask = buy_notional_scaled / max(0.25, max(mid_ret_bps, 0.0))
            absorption_bid = sell_notional_scaled / max(0.25, max(-mid_ret_bps, 0.0))
            signed_flow_per_bp_move = signed_notional_scaled / max(0.25, abs(mid_ret_bps))
            price_response_to_buy_flow = max(mid_ret_bps, 0.0) / max(buy_notional_scaled, 1e-9)
            price_response_to_sell_flow = max(-mid_ret_bps, 0.0) / max(sell_notional_scaled, 1e-9)
            absorption_values = [
                buy_flow_without_price_up,
                sell_flow_without_price_down,
                absorption_bid,
                absorption_ask,
                signed_flow_per_bp_move,
                price_response_to_buy_flow,
                price_response_to_sell_flow,
            ]
            for i, value in enumerate(absorption_values):
                if not math.isfinite(float(value)):
                    raise ValueError(f"Non-finite absorption stat idx={i} value={value!r} at ts_ms={ts_ms} window={ms}")
            feat_list.extend(absorption_values)

        for ms in FLOW_WINDOWS_MS:
            feat_list.append(return_std_bps[ms])
        for prev_ms, cur_ms in zip(FLOW_WINDOWS_MS[:-1], FLOW_WINDOWS_MS[1:]):
            feat_list.append(variance_ratio_adjacent[(cur_ms, prev_ms)])

        for ms in REGIME_WINDOWS_MS:
            dist = self._regime_distribution(ms)
            feat_list.extend([
                regime_volume[ms],
                regime_realized[ms],
                regime_vol_ewma[ms],
            ])
            if ms <= 60_000:
                feat_list.append(regime_flow_snapshot[ms])
            feat_list.extend([
                dist["realized_up_vol_bps"],
                dist["realized_down_vol_bps"],
                dist["down_up_vol_ratio"],
                dist["bipower_variation"],
                dist["jump_variation"],
                dist["max_abs_return_bps"],
                dist["return_skew"],
                dist["return_kurtosis"],
            ])
        for ms in SPREAD_DEPTH_REGIME_WINDOWS_MS:
            spread_state = self._spread_bps_regime_states[ms]
            depth_state = self._depth_5bps_total_regime_states[ms]
            imb_state = self._depth_5bps_imbalance_regime_states[ms]
            bid_state = self._bid_depth_5bps_regime_states[ms]
            ask_state = self._ask_depth_5bps_regime_states[ms]

            spread_mean, spread_std = spread_state.mean_std()
            spread_stats = {
                "mean": spread_mean,
                "std": spread_std,
                "p90": spread_state.p90(),
                "max": spread_state.max(),
                "min": spread_state.min(),
            }
            spread_z = 0.0 if spread_std <= 1e-9 else (spread_bps - spread_mean) / max(spread_std, 1e-9)
            spread_slope = spread_state.slope()
            spread_above = spread_state.frac_above(1.0)

            depth_mean, depth_std = depth_state.mean_std()
            depth_z = 0.0 if depth_std <= 1e-9 else (depth_5bps_total - depth_mean) / max(depth_std, 1e-9)
            imb_mean = imb_state.mean()
            imb_slope = imb_state.slope()

            bid_mean = bid_state.mean()
            ask_mean = ask_state.mean()

            feat_list.extend([
                spread_stats["mean"],
                spread_stats["std"],
                spread_stats["p90"],
                spread_stats["max"],
                spread_stats["min"],
                spread_z,
                spread_slope,
                spread_above,
                depth_mean,
                depth_std,
                depth_z,
                imb_mean,
                imb_slope,
                (bid_depth_5bps["size"] / max(bid_mean, 1e-9)) - 1.0,
                (ask_depth_5bps["size"] / max(ask_mean, 1e-9)) - 1.0,
            ])
        bid_depth_5bps_base = float(band_depth_stats[5.0]["bid"]["size"])
        ask_depth_5bps_base = float(band_depth_stats[5.0]["ask"]["size"])
        bid_notional_5bps_base = float(band_depth_stats[5.0]["bid"]["notional"])
        ask_notional_5bps_base = float(band_depth_stats[5.0]["ask"]["notional"])
        depth_imbalance_5bps = self._safe_div(
            bid_depth_5bps_base - ask_depth_5bps_base,
            bid_depth_5bps_base + ask_depth_5bps_base,
            0.0,
        )
        depth_5bps_total_base = bid_depth_5bps_base + ask_depth_5bps_base
        depth_5bps_total_notional = bid_notional_5bps_base + ask_notional_5bps_base

        for ms in FAST_WINDOWS_MS:
            ofi_pressure = ofi_pressure_by_ms[ms]
            feat_list.extend([
                ofi_pressure,
                self._safe_div(ofi_pressure, max(depth_5bps_total_base, 1e-9), 0.0),
                self._safe_div(ofi_pressure, max(self._realized_vol_for_pressure(ms), 1e-9), 0.0),
            ])

        for ms in INTERACTION_WINDOWS_MS:
            trade_imbalance = float(trade_stats_by_ms[ms]["trade_imbalance_notional"])
            signed_flow_usd = float(trade_stats_by_ms[ms]["signed_notional_flow_usd"])
            signed_flow_scaled = signed_flow_usd / 100_000.0
            ofi_pressure = float(ofi_pressure_by_ms[ms])
            realized_vol = float(self.realized_vol[ms])
            flow_sign = 1 if trade_imbalance > 0.0 else (-1 if trade_imbalance < 0.0 else 0)
            book_sign = 1 if depth_imbalance_5bps > 0.0 else (-1 if depth_imbalance_5bps < 0.0 else 0)
            flow_agrees_with_book = 1.0 if (flow_sign != 0 and book_sign != 0 and flow_sign == book_sign) else 0.0
            flow_disagrees_with_book = 1.0 if (flow_sign != 0 and book_sign != 0 and flow_sign != book_sign) else 0.0
            feat_list.extend([
                flow_agrees_with_book,
                flow_disagrees_with_book,
                trade_imbalance * depth_imbalance_5bps,
                trade_imbalance * ofi_pressure,
                ofi_pressure * spread_bps,
                micro_premia * trade_imbalance,
                micro_premia * depth_imbalance_5bps,
                self._safe_div(signed_flow_usd, max(depth_5bps_total_notional, 1e-9), 0.0),
                self._safe_div(signed_flow_scaled, max(spread_bps, 0.1), 0.0),
                self._safe_div(abs(signed_flow_scaled), max(realized_vol, 1e-9), 0.0),
            ])

        for hl in self.ema_half_lives_ms:
            state = self.ema_states[hl]
            for name in self.ema_indicator_names:
                if state[name] is None:
                    raise ValueError(f"Uninitialized EMA state for {name} hl={hl}")
                feat_list.append(state[name])
        for hl in self.ema_half_lives_ms:
            state = self.ema_states[hl]
            for name in self.ema_indicator_names:
                if state[name] is None:
                    raise ValueError(f"Uninitialized EMA state for {name} hl={hl}")
                ema_val = state[name]
                feat_list.append(indicator_values[name] - ema_val)

        macd_features = []
        for fast_ms, slow_ms, sig_ms in MACD_TRIPLETS_MS:
            st = self.macd_state[(fast_ms, slow_ms, sig_ms)]
            if st["fast"] is None:
                st["fast"] = float(micro)
            else:
                st["fast"] = self._ewma_update(float(st["fast"]), float(micro), ob_dt_ms, int(fast_ms))
            if st["slow"] is None:
                st["slow"] = float(micro)
            else:
                st["slow"] = self._ewma_update(float(st["slow"]), float(micro), ob_dt_ms, int(slow_ms))
            fast_ema = float(st["fast"])
            slow_ema = float(st["slow"])
            raw_bps = 1e4 * math.log(fast_ema / slow_ema) if (fast_ema > 0.0 and slow_ema > 0.0) else 0.0
            if not bool(st["signal_initialized"]):
                st["signal"] = float(raw_bps)
                st["signal_initialized"] = True
            else:
                st["signal"] = self._ewma_update(float(st["signal"]), float(raw_bps), ob_dt_ms, int(sig_ms))
            sig_bps = float(st["signal"])
            hist_bps = float(raw_bps - sig_bps)
            macd_features.extend([raw_bps, sig_bps, hist_bps])
        feat_list.extend(macd_features)
        feat_list.extend(vpin_features)
        names = self.feature_names()
        if len(feat_list) != len(names):
            raise ValueError(
                f"Feature vector/name length mismatch: len(feat_list)={len(feat_list)} "
                f"len(feature_names)={len(names)}"
            )
        feat = np.asarray(feat_list, dtype=np.float64)
        if self.strict_feature_validation:
            if not np.all(np.isfinite(feat)):
                bad_idx = np.flatnonzero(~np.isfinite(feat))
                details = [
                    f"{int(i)}:{names[int(i)]}={feat[int(i)]!r}"
                    for i in bad_idx[:20]
                ]
                raise FloatingPointError("Non-finite feature values: " + ", ".join(details))
        feat_z = self._zscore(feat, ts_ms)
        self._append_price_history(ts_ms, mid, micro)
        self.prev_bid1_price = bid1
        self.prev_ask1_price = ask1
        self.prev_mid_price_for_age = mid
        self.prev_spread_for_age = spread
        self.last_ts = ts_ms
        self._last_ob_feature_ts = int(ts_ms)
        self._last_any_event_ts = int(ts_ms)

        self.ob_feature_build_count += 1
        return ts_ms, feat_z, mid, is_trade, ob_dt_ms

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
        self.last_trade_price = price
        self.last_is_rpi = is_rpi

        notional_usd = price * size
        if side_sign > 0:
            self.consecutive_buy_trade_count += 1
            self.consecutive_sell_trade_count = 0
            trade_sign = 1
        elif side_sign < 0:
            self.consecutive_sell_trade_count += 1
            self.consecutive_buy_trade_count = 0
            trade_sign = -1
        else:
            self.consecutive_buy_trade_count = 0
            self.consecutive_sell_trade_count = 0
            trade_sign = 0
        self.trade_sign_history.append((int(ts_ms), int(trade_sign), float(notional_usd)))
        trade_cutoff = int(ts_ms) - int(max(TRADE_BURST_WINDOWS_MS) + 5_000)
        while self.trade_sign_history and self.trade_sign_history[0][0] < trade_cutoff:
            self.trade_sign_history.popleft()
        for window in TRADE_BURST_WINDOWS_MS:
            self._trade_burst_insert(window, int(ts_ms), int(trade_sign))

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

        if notional_usd >= LARGE_TRADE_CLOCK_THRESHOLD_USD:
            bid1_now, ask1_now, _, _ = self._book_best()
            mid_now = 0.5 * (bid1_now + ask1_now) if bid1_now > 0.0 and ask1_now > 0.0 else 0.0
            if side_sign > 0:
                self.last_large_buy_ts = int(ts_ms)
                self.last_large_buy_mid = float(mid_now)
                self.last_large_buy_notional_usd = float(notional_usd)
                self.last_large_buy_ofi_l5_at_event = float(self.last_ob_ofi_l5)
                self.last_large_buy_trade_imbalance_at_event = float(self.last_ob_trade_imbalance_30000ms)
            elif side_sign < 0:
                self.last_large_sell_ts = int(ts_ms)
                self.last_large_sell_mid = float(mid_now)
                self.last_large_sell_notional_usd = float(notional_usd)
                self.last_large_sell_ofi_l5_at_event = float(self.last_ob_ofi_l5)
                self.last_large_sell_trade_imbalance_at_event = float(self.last_ob_trade_imbalance_30000ms)

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
        v_per_sec = max(self.volume_ewma[15_000], 1e-9)
        for secs, st in self.vpin_state.items():
            Vb = max(v_per_sec * float(secs), 1e-9)
            st["Vb"] = Vb if st["Vb"] is None else (0.9 * st["Vb"] + 0.1 * Vb)
            if side_sign > 0:
                st["cum_buy"] += size
            elif side_sign < 0:
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
        if ms in self.realized_vol:
            return float(self.realized_vol.get(ms, 0.0))
        return float(self.realized_vol.get(3_000, 0.0))

    def _feature_z_half_life_ms(self, feature_name: str) -> Optional[int]:
        if (
            feature_name.startswith("time_hour_")
            or feature_name.startswith("time_dow_")
            or feature_name.startswith("session_")
        ):
            return None
        if any(tok in feature_name for tok in (
            "ofi_l", "obi_l", "pressure", "persistence",
            "micro_l", "vamp_l", "slippage_imbalance",
            "depth_needed_imbalance", "filled_fraction_imbalance",
            "trade_burst", "run_length", "trade_sign_entropy",
            "large_buy_continuation", "large_sell_continuation",
            "return_since_last_large", "ofi_l5_since_last_large",
            "trade_imbalance_since_last_large",
        )):
            return 60_000
        if any(
            tok in feature_name
            for tok in (
                "flow_agrees_with_book",
                "flow_disagrees_with_book",
                "trade_imbalance_x",
                "ofi_pressure_x",
                "micro_premia_x",
                "signed_flow_over",
                "abs_signed_flow_over",
            )
        ):
            return 60_000
        if any(tok in feature_name for tok in ("spread_delta", "ofi_", "obi_", "pressure", "replen", "change_count", "depletion")):
            return 30_000
        if any(tok in feature_name for tok in ("trade", "flow", "large", "vpin", "cvd", "aggressor", "absorption")):
            return 60_000
        if any(tok in feature_name for tok in ("price", "trend", "range", "macd", "return", "vol", "regime", "slippage", "depth", "liquidity", "spread_mean", "spread_std", "spread_p90", "spread_z")):
            return 120_000
        return 120_000

    def _zscore(self, x: np.ndarray, ts_ms: int) -> np.ndarray:
        if x.ndim != 1:
            raise ValueError(f"_zscore expects 1D feature vector, got shape={x.shape}")
        names = self.feature_names()
        if len(x) != len(names):
            raise ValueError(f"_zscore feature length mismatch: got={len(x)} expected={len(names)}")
        if not np.all(np.isfinite(x)):
            raise ValueError("Non-finite values passed to _zscore")
        x64 = x.astype(np.float64, copy=False)
        dim = int(x64.shape[0])
        self._ensure_zscore_metadata(names, dim)

        if self._z_mask is None or self._z_half_lives_arr is None:
            raise RuntimeError("z-score metadata was not initialized")

        dt_ms = 1.0 if self._last_z_ts_ms is None else float(max(1, int(ts_ms) - int(self._last_z_ts_ms)))
        self._last_z_ts_ms = int(ts_ms)

        if self._feat_dim is None:
            self._feat_dim = dim
            self.z_mean = x64.copy()
            self.z_var = np.ones_like(x64, dtype=np.float64)
            out = x64.copy()
            out[self._z_mask] = 0.0
            out32 = out.astype(np.float32, copy=False)
            if not np.all(np.isfinite(out32)):
                raise ValueError("Non-finite values produced by _zscore")
            return out32

        if dim != int(self._feat_dim):
            raise ValueError(f"_zscore feature dimension changed: got={dim} expected={self._feat_dim}")
        if self.z_mean is None or self.z_var is None:
            raise RuntimeError("z-score state is uninitialized despite _feat_dim being set")

        out = x64.copy()
        mask = self._z_mask
        if np.any(mask):
            hl = self._z_half_lives_arr[mask]
            alpha = 1.0 - np.exp(-math.log(2.0) * dt_ms / hl)
            old_mean = self.z_mean[mask]
            old_var = self.z_var[mask]
            x_m = x64[mask]
            diff = x_m - old_mean
            new_mean = old_mean + alpha * diff
            new_var = (1.0 - alpha) * (old_var + alpha * diff * diff)
            self.z_mean[mask] = new_mean
            self.z_var[mask] = new_var
            z = (x_m - new_mean) / np.sqrt(np.maximum(new_var, 1e-6))
            out[mask] = np.clip(z, -10.0, 10.0)

        out32 = out.astype(np.float32, copy=False)
        if not np.all(np.isfinite(out32)):
            raise ValueError("Non-finite values produced by _zscore")
        return out32

    def _ensure_zscore_metadata(self, names: List[str], dim: int) -> None:
        if self._z_half_lives_ms is not None:
            return

        half_lives = [self._feature_z_half_life_ms(name) for name in names]
        self._z_half_lives_ms = half_lives
        mask = np.asarray([hl is not None for hl in half_lives], dtype=bool)
        hl_arr = np.ones((dim,), dtype=np.float64)
        for i, hl in enumerate(half_lives):
            if hl is not None:
                hl_arr[i] = float(max(1, int(hl)))
        self._z_mask = mask
        self._z_half_lives_arr = hl_arr

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------
    def on_fast_event(self, e: Any) -> Tuple[int, np.ndarray, float, bool, float]:
        """Fast ingest path for compact tuples emitted by offline_ingest.py."""
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

    def on_event(self, e: Any) -> Tuple[int, np.ndarray, float, bool, float]:
        """Slow compatibility path for callers that still pass generic event shapes."""
        etype, ts_ms, payload = self._parse_event(e)

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
    vals = metric_payload.get("edge_spearman_q50plus", [])
    dir_vals = metric_payload.get("dir_bal_acc_q50plus", [])
    if idx >= len(vals) or idx >= len(dir_vals):
        return float("nan"), PRIMARY_METRIC
    edge_value = float(vals[idx])
    dir_guard_value = float(dir_vals[idx])
    if not math.isfinite(edge_value):
        return float("nan"), PRIMARY_METRIC
    if not math.isfinite(dir_guard_value) or dir_guard_value < PRIMARY_DIR_BAL_ACC_GUARD:
        return float("nan"), PRIMARY_METRIC
    return edge_value, PRIMARY_METRIC

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
