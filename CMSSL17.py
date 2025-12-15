import os, math, copy, json, csv, zipfile, io, glob, re, gzip, contextlib, time
from collections import deque
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Deque, Any, List, Dict, Tuple, Generator, Optional, Iterable, Union
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import math
from einops import rearrange, repeat
import torch._functorch.config as ft_config
from sklearn.decomposition import PCA
from datetime import datetime

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

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# ---------------------------  Core hyper-params  ---------------------------
LOOKBACK        = 128        # number of tokens spanning ~10s
WINDOW_MS       = 10_000     # time-based window span (10s)
PAD_DT_FOR_LEFT = 0.0
BATCH_SIZE      = 256
DMODEL          = 256
MAMBA_LAYERS    = 2
CONV_KERNELS    = [9,17,25,33]
DFF_CONV        = 2 * DMODEL

# Prediction horizons (in milliseconds)
HORIZONS_MS     = [250, 500, 1000]
NUM_HORIZONS    = len(HORIZONS_MS)
HORIZON_WEIGHTS = [0.05, 0.1, 1.0]

# Masking / SSL schedule
SSL_PRETRAIN_EPOCHS = 0      # Pretrain epochs (recon + CPC only)
MASK_PRETRAIN       = 0.0    # Pretrain mask ratio
MASK_FINETUNE       = 0.0    # Fine-tune mask ratio

DIR_MASK_TAIL_FRACTION = 0.01
EPOCHS          = 200
LR              = 5e-4
CLIP_GRAD       = 10000
PATIENCE        = 15
# Number of auxiliary channels appended after the base feature vector
# These correspond to [log_dt_ms, is_trade, events_100ms]
AUX_DIM        = 3
NUM_HEADS       = 8
WARMUP_EPOCHS   = max(1, int(EPOCHS * 0.05))  # Warmup over first 5% of epochs

# Loss mixing (fixed lambdas), with EMA normalization per loss
EMA_DECAY       = 0.99
LAMBDA_BCE      = 0.20
LAMBDA_RECON_FT = 0.00
LAMBDA_CPC_FT   = 0.00
LAMBDA_RECON_PT = 0.00
LAMBDA_CPC_PT   = 0.00

# Huber deltas (per horizon).
# We start from calibrated 250ms thresholds (1e-4 return, 0.02 log-vol)
# and scale them ~sqrt(horizon) to reflect the diffusive growth in magnitude
# as horizons lengthen.
_DELTA_BASE_H = HORIZONS_MS[0]
DELTA_RET       = [1e-4 * math.sqrt(h / _DELTA_BASE_H) for h in HORIZONS_MS]
DELTA_LOGVOL    = [0.02 * math.sqrt(h / _DELTA_BASE_H) for h in HORIZONS_MS]

# CPC settings
CPC_DELTAS_MS   = [25, 50, 100]  # 25/50/100 ms

# File Locations
DATA_ROOT = os.environ.get("BYBIT_DATA_ROOT", os.path.expanduser("~/Gabriel"))

# PCA
USE_PCA = False
PCA_VAR = 0.99
#---------------------------------------------------------------------------


def _list_files(patterns):
    out = []
    for pat in patterns:
        out.extend(glob.glob(pat))
    return out

def _week_key(path: str, prefix: str) -> str:
    # e.g. BTCUSDT_TH_2024-W35.zip -> 2024-W35
    base = os.path.basename(path)
    base = re.sub(r'\.(zip|gz)$', '', base)  # strip extension
    return base.replace(prefix, "")

def _pair_by_week(data_root: str):
    ob_candidates = _list_files([
        os.path.join(data_root, "OB", "BTCUSDT_OB_*.zip"),
        os.path.join(data_root, "OB", "BTCUSDT_OB_*.gz"),
    ])
    th_candidates = _list_files([
        os.path.join(data_root, "TH", "BTCUSDT_TH_*.zip"),
        os.path.join(data_root, "TH", "BTCUSDT_TH_*.gz"),
    ])

    ob_map = { _week_key(p, "BTCUSDT_OB_"): p for p in ob_candidates }
    th_map = { _week_key(p, "BTCUSDT_TH_"): p for p in th_candidates }

    common = sorted(set(ob_map) & set(th_map))
    if not common:
        raise FileNotFoundError(
            f"No matching weeks found under {data_root}. "
            f"Looked for OB/TH .zip or .gz"
        )

    # Warn if anything is missing
    missing_ob = sorted(set(th_map) - set(ob_map))
    missing_th = sorted(set(ob_map) - set(th_map))
    if missing_ob:
        print(f"Warning: missing OB for weeks: {missing_ob}")
    if missing_th:
        print(f"Warning: missing TH for weeks: {missing_th}")

    return [(ob_map[w], th_map[w]) for w in common]

def _parse_week_key_any(base: str):
    # base is like "BTCUSDT_OB_01-06-2025-to-07-06-2025" or "BTCUSDT_OB_2025-06-01-to-2025-06-07"
    wk = re.sub(r'^(BTCUSDT_(?:OB|TH)_)', '', base)
    wk = re.sub(r'\.(zip|gz)$', '', wk)
    m = re.match(r"(\d{2}-\d{2}-\d{4})-to-(\d{2}-\d{2}-\d{4})", wk)
    if m:
        s = datetime.strptime(m.group(1), "%d-%m-%Y")
        e = datetime.strptime(m.group(2), "%d-%m-%Y")
        return s, e, wk
    m = re.match(r"(\d{4}-\d{2}-\d{2})-to-(\d{4}-\d{2}-\d{2})", wk)
    if m:
        s = datetime.strptime(m.group(1), "%Y-%m-%d")
        e = datetime.strptime(m.group(2), "%Y-%m-%d")
        return s, e, wk
    raise ValueError(f"Unrecognized week key: {base}")

def _slice_last_weeks_pairs(week_files, last_end_iso="2025-08-27", k=28):
    rows = []
    for ob_p, th_p in week_files:
        base = os.path.basename(ob_p)
        s,e,_ = _parse_week_key_any(base.replace("BTCUSDT_TH_", "BTCUSDT_OB_"))
        rows.append((e, s, ob_p, th_p))
    rows.sort()
    target_end = datetime.strptime(last_end_iso, "%Y-%m-%d")
    idx = max(i for i,(e,_,_,_) in enumerate(rows) if e <= target_end)
    lo = max(0, idx - (k - 1))
    sel = rows[lo:idx+1]
    return [(ob, th) for (_,_,ob,th) in sel]

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
    def __init__(self, patch_count, patch_stride, patch_size, seq_len, channels, device='cuda:0'):
        super().__init__()
        self.device = device
        self.seq_len = seq_len
        self.channels = channels
        self.patch_size = patch_size
        self.patch_count = patch_count
        self.patch_stride = patch_stride
        self._generate_anchor(device=device)
    def _generate_anchor(self, device="cuda:0"):
        anchors = []
        self.S_bias = (self.patch_size - 1) / 2
        for i in range(self.patch_count):
            x = i * self.patch_stride + 0.5 * (self.patch_size - 1)
            anchors.append(x)
        anchors = torch.as_tensor(anchors, device=device)
        self.register_buffer("anchor", anchors)
    def forward(self, boxes):
        self.bound = self.decode(boxes)
        points = self.meshgrid(self.bound)
        return points, self.bound
    def decode(self, rel_codes):
        boxes = self.anchor
        dx = rel_codes[:, :, :, 0]
        ds = torch.relu(rel_codes[:, :, :, 1] + self.S_bias)
        pred_boxes = torch.zeros_like(rel_codes)
        ref_x = boxes.view(1, boxes.shape[0], 1)
        pred_boxes[:, :, :, 0] = (dx + ref_x - ds) 
        pred_boxes[:, :, :, 1] = (dx + ref_x + ds) 
        pred_boxes /= (self.seq_len - 1)
        pred_boxes = pred_boxes.clamp_(min=0., max=1.)
        return pred_boxes	
    def meshgrid(self, boxes):
        B, patch_count, C = boxes.shape[0], boxes.shape[1], boxes.shape[2]
        channel_boxes = torch.zeros((boxes.shape[0], boxes.shape[1], 2)).to(self.device)
        channel_boxes[:, :, 1] = 1.0
        xs = boxes.view(B*patch_count, C, 2)
        xs = torch.nn.functional.interpolate(xs, size=self.patch_size, mode='linear', align_corners=True)
        ys = torch.nn.functional.interpolate(channel_boxes, size=self.channels, mode='linear', align_corners=True)
        xs = xs.view(B, patch_count, C, self.patch_size, 1)
        ys = ys.unsqueeze(3).expand(B, patch_count, C, self.patch_size).unsqueeze(-1)
        grid = torch.stack([xs, ys], dim = -1)
        return grid

class OffsetPredictor(nn.Module):
    def __init__(self, in_feats, patch_size, stride, use_zero_init=True):
        super().__init__()
        self.stride = stride
        self.channel = in_feats
        self.patch_size = patch_size
        self.offset_predictor = nn.Sequential(
            nn.Conv1d(1, 64, patch_size, stride=stride, padding=0),
            nn.GELU(),
            nn.Conv1d(64, 2, 1, 1, padding=0)
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
        self.dw_norm = nn.LayerNorm(d_model) if norm != 'batch' else nn.BatchNorm1d(d_model)  # switchable
        self.ff = nn.Sequential(nn.Conv1d(d_model, d_ff, 1, 1), 
                                get_activation_fn(activation), 
                                nn.Dropout(dropout), 
                                nn.Conv1d(d_ff, d_model, 1, 1))
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
            if self.training:
                large_out = self.DW_conv_large(src)
                small_out = self.DW_conv_small(src)
                out_x = large_out + small_out
            else:
                self._get_merged_param()
                out_x = self.DW_infer(src)
        else:
            out_x = self.DW_conv(src)

        residual_src = self.sublayerconnect1(src, self.dw_act(out_x))
        normed_src = residual_src.permute(0, 2, 1) if self.norm_tp != 'batch' else residual_src
        normed_src = self.dw_norm(normed_src)
        normed_src = normed_src.permute(0, 2, 1) if self.norm_tp != 'batch' else normed_src
        ff_out = self.ff(normed_src)
        residual_src2 = self.sublayerconnect2(normed_src, ff_out)
        normed_src2 = residual_src2.permute(0, 2, 1) if self.norm_tp != 'batch' else residual_src2
        normed_src2 = self.norm_ffn(normed_src2)
        normed_src2 = normed_src2.permute(0, 2, 1) if self.norm_tp != 'batch' else normed_src2
        return normed_src2

class ConvEncoder(nn.Module):
    def __init__(self, d_model, d_ff, kernel_size=[3,5,7,11,13,17], dropout=0.1, activation='gelu', 
                 n_layers=2, enable_res_param=True, norm='batch', re_param=True, small_ks=3):
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
                 enable_res_param=True, norm='batch', re_param=True, re_param_kernel=3, patch_size=8, stride=4):
        super(ConvTimeNetFeatureExtractor, self).__init__()
        self.depatch = DepatchSampling(in_feats=in_feats, seq_len=seq_len, patch_size=patch_size, stride=stride)
        self.patch_count = (seq_len - patch_size) // stride + 1
        self.patch_size = patch_size
        self.d_model_internal = max(1, d_model // in_feats)
        self.output_linear = nn.Linear(patch_size, self.d_model_internal)
        self.encoder = ConvEncoder(d_model=self.d_model_internal, d_ff=d_ff, kernel_size=dw_ks, dropout=dropout, activation=act,
                                   n_layers=n_layers, enable_res_param=enable_res_param, norm=norm, re_param=re_param, small_ks=re_param_kernel)
        self.final_proj = nn.Linear(self.d_model_internal * in_feats, d_model)
    def forward(self, x):
        out_patch = self.depatch(x)  # [B, feats, patch_count, patch_size]
        out = self.output_linear(out_patch)  # [B, feats, patch_count, d_model_internal]
        B = out.shape[0]
        u = out.reshape(B * out.shape[1], out.shape[2], self.d_model_internal)  # [B * feats, patch_count, d_model_internal]
        u = u.permute(0, 2, 1)  # [B * feats, d_model_internal, patch_count]
        out = self.encoder(u)  # [B * feats, d_model_internal, patch_count]
        out = out.permute(0, 2, 1)  # [B * feats, patch_count, d_model_internal]
        out = out.reshape(B, out.shape[0] // B, out.shape[1], self.d_model_internal)  # [B, feats, patch_count, d_model_internal]
        out = out.permute(0, 2, 3, 1)  # [B, patch_count, d_model_internal, feats]
        out = out.reshape(B, self.depatch.patch_count, self.d_model_internal * out.shape[3])  # [B, patch_count, d_model_internal * feats]
        out = self.final_proj(out)  # [B, patch_count, d_model]
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
        # (3) Switch BatchNorm -> LayerNorm in ConvTimeNet
        self.depatch_proj_encoder = ConvTimeNetFeatureExtractor(
            in_feats=args.vocab_size, seq_len=args.seq_in, d_model=args.d_model, 
            dw_ks=[3,5,7,11,13,17], n_layers=2, d_ff=256, dropout=0.1, act='gelu', 
            enable_res_param=True, norm='layer', re_param=True, re_param_kernel=3, 
            patch_size=8, stride=4
        )
        # Mamba backbone (forward/backward fusion) + pooling
        self.mamba = Mamba(args, ff_hid=DMODEL)

        # SSL bits
        self.mask_token = nn.Parameter(torch.randn(1, 1, args.d_model))
        self.cpc_deltas = CPC_DELTAS_MS
        self.cpc_predictors = nn.ModuleDict({
            f"ms{d}": nn.Linear(self.args.d_model, 2 * self.args.d_model, bias=False)
            for d in CPC_DELTAS_MS
        })
        # EMA teacher for CPC targets (teacher = EMA(student))
        self.mamba_teacher = copy.deepcopy(self.mamba)
        for p in self.mamba_teacher.parameters():
            p.requires_grad = False
        self.teacher_momentum = 0.99

        # Heads
        fused_dim = args.d_model * 2
        head_hidden_dim = fused_dim * 2
        self.return_head = nn.Sequential(
            nn.Linear(fused_dim, head_hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(head_hidden_dim, NUM_HORIZONS)
        )
        self.volatility_head = nn.Sequential(
            nn.Linear(fused_dim, head_hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(head_hidden_dim, NUM_HORIZONS)  # predicts log-vol per horizon
        )
        self.direction_head = nn.Sequential(
            nn.Linear(fused_dim, head_hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(head_hidden_dim, NUM_HORIZONS)
        )

    @torch.no_grad()
    def update_teacher(self, m: float = None):
        """EMA update for teacher parameters."""
        if m is None:
            m = self.teacher_momentum
        for p_t, p_s in zip(self.mamba_teacher.parameters(), self.mamba.parameters()):
            p_t.data.mul_(m).add_(p_s.data, alpha=(1.0 - m))

    def compute_cpc_loss(self, h_student: torch.Tensor, h_teacher: torch.Tensor, dt_patch: torch.Tensor) -> torch.Tensor:
        """CPC loss between student forward states and teacher bidirectional targets.

        Args:
            h_student: [B, P, D] forward student states (anchors).
            h_teacher: [B, P, 2D] teacher fused states computed without future leakage.
            dt_patch: [B, P] patch-level time deltas.
        """
        # dt_patch encodes ms progression along the patch dimension
        cum_t = torch.cumsum(dt_patch, dim=1)  # ms from left to right in patch space

        total = 0.0
        count = 0
        B, P, D = h_student.shape
        for dms in CPC_DELTAS_MS:
            # For each (b, i), we need j >= i with cum_t[b,j] - cum_t[b,i] >= dms
            # Build j by scanning once per sequence via broadcasting
            t_i = cum_t.unsqueeze(2)                # [B,P,1]
            t_j = cum_t.unsqueeze(1)                # [B,1,P]
            # mask of valid targets
            valid = (t_j - t_i) >= dms
            # take first True along last dim
            idx_j = valid.float().argmax(dim=2)     # [B,P] (argmax=0 if none; handle via mask)
            has = valid.any(dim=2)
            # gather teacher targets at j
            gather_j = idx_j
            b_idx = torch.arange(B, device=h_student.device).unsqueeze(-1).expand(B, P)
            h_tgt = h_teacher[b_idx, gather_j]  # [B,P,2D]
            proj = self.cpc_predictors[f"ms{dms}"](h_student)  # [B,P,2D]
            # InfoNCE-like cosine distance with stop-grad target
            loss = 1.0 - F.cosine_similarity(proj, h_tgt.detach(), dim=-1)  # [B,P]
            total += (loss * has.float()).sum()
            count += has.float().sum().clamp_min(1.0)
        return total / count


    def forward(self, x, mask_ratio=0.0, mask_idx: torch.Tensor = None):
        """
        Training path returns: pooled, ret_pred, vol_pred, dir_logits,
        h_clean (student fused), h_masked (student fused on masked input), mask_idx, cpc_loss.
        Eval path returns predictions only.
        """
        x_permuted = x.permute(0, 2, 1)
        h_tokens = self.depatch_proj_encoder(x_permuted)                   # [B, L, D] (ConvTimeNet projection applied)
        # Tokens expose log1p(dt_ms); invert to recover raw millisecond gaps for CPC geometry.
        log_dt = x[..., -3]
        log_dt = log_dt.clamp_min(0.0)
        dt_raw = torch.expm1(log_dt).clamp_min(0.0)
        ps = self.depatch_proj_encoder.depatch.patch_size
        stride = self.depatch_proj_encoder.depatch.box_coder.patch_stride
        dt_patch = dt_raw.unfold(1, ps, stride).sum(-1)

        # Student (clean)
        pooled, h_clean, h_clean_fwd = self.mamba(h_tokens, embedded=True)
        ret = self.return_head(pooled)
        vol = self.volatility_head(pooled)
        dir_logits = self.direction_head(pooled)

        # Teacher (clean, no grad) for CPC (bidirectional over observed past window)
        h_teacher_fused = self.compute_teacher_cpc_targets(h_tokens)

        # Masked pass (student) for reconstruction distillation in Mamba space
        B, L, D = h_tokens.shape
        if mask_idx is None:
            mcnt = max(1, int(mask_ratio * L))
            mask_idx = torch.stack(
                [torch.randperm(L, device=x.device)[:mcnt] for _ in range(B)]
            )  # [B, mcnt]
        else:
            mcnt = mask_idx.shape[1]
        h_masked_input = h_tokens.clone()
        batch_idx = torch.arange(B, device=x.device).unsqueeze(1).expand(-1, mcnt)
        h_masked_input[batch_idx, mask_idx] = self.mask_token  # replace masked tokens

        _, h_masked, _ = self.mamba(h_masked_input, embedded=True)

        # CPC loss (computed here so both SAM passes align)
        cpc_loss = self.compute_cpc_loss(h_clean_fwd, h_teacher_fused, dt_patch)
        
        return ret, vol, dir_logits, h_clean, h_masked, mask_idx, cpc_loss

    @torch.no_grad()
    def compute_teacher_cpc_targets(self, h_tokens: torch.Tensor) -> torch.Tensor:
        """Compute teacher representations for CPC without peeking past prediction horizons.

        Runs the teacher forward stack once (causal) and recomputes the backward
        stack on prefixes so each position only uses information available up to
        that timestep. Returns fused [B, L, 2D] tensors.
        """
        teacher = self.mamba_teacher

        # Forward stack is causal, so a single pass suffices.
        x_fwd = teacher._run_stack(h_tokens, teacher.blocks_fwd, teacher.ffns_fwd)
        h_fwd = teacher.norm_fwd(x_fwd)

        B, L, _ = h_tokens.shape
        bwd_states = []
        for end in range(1, L + 1):
            prefix = h_tokens[:, :end]
            rev_prefix = torch.flip(prefix, dims=[1])
            x_bwd_rev = teacher._run_stack(rev_prefix, teacher.blocks_bwd, teacher.ffns_bwd)
            # The first element in reversed space corresponds to the latest original index.
            x_bwd_last = x_bwd_rev[:, 0]
            h_bwd_last = teacher.norm_bwd(x_bwd_last)
            bwd_states.append(h_bwd_last)

        h_bwd = torch.stack(bwd_states, dim=1)
        return torch.cat([h_fwd, h_bwd], dim=-1)

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
                p.data = self.state[p]["old_p"]
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
        # OB is line-delimited JSON
        with _open_text(self.ob_zip) as f:
            for line in f:
                if not line:
                    continue
                obj = json.loads(line)
                ts = int(obj.get("ts", obj.get("cts", 0)))
                seq = obj["data"].get("seq", 0)
                yield ts, seq, obj

    def trade_iter(self):
        # TH is CSV with a 'timestamp' column in seconds
        with _open_text(self.th_zip) as f:
            reader = csv.DictReader(f)
            seq = 0
            for row in reader:
                seq += 1
                ts = int(float(row["timestamp"]) * 1000)
                row["seq"] = seq
                yield ts, seq, row


def merge_event_time(ob_iter, tr_iter, B: int = 0):
    """Merge OB and trade iterators by timestamp and sequence."""
    ob_item = next(ob_iter, None)
    tr_item = next(tr_iter, None)
    last_ts = -1
    while ob_item or tr_item:
        if ob_item and (tr_item is None or ob_item[0] < tr_item[0]):
            ts, seq, data = ob_item
            ob_item = next(ob_iter, None)
            etype = "ob"
        else:
            # Prefer the trade when timestamps tie to preserve causal ordering.
            ts, seq, data = tr_item
            tr_item = next(tr_iter, None)
            etype = "trade"
        if ts + B < last_ts:
            raise ValueError("Non-monotonic timestamps in event stream")
        last_ts = ts
        yield etype, ts, seq, data


# ---------------------  Rolling normalization  ---------------------

class RollingZScore:
    def __init__(self, window_ms: int = 10000):
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


# -------------------------  Feature engine  -------------------------
class FeatureEngine:
    
    def __init__(
        self,
        depth: int = 10,
        z_hl_ms: int = 500,
        vpin_target_bucket_secs: float = 2.0,
    ):
        self.depth = int(depth)
        self.z_hl_ms = int(z_hl_ms)
        self.vpin_target_bucket_secs = float(vpin_target_bucket_secs)

        # ---------- Book state ----------
        self.bids: Dict[float, float] = {}
        self.asks: Dict[float, float] = {}
        self.bid_lvls: List[Tuple[float, float]] = []  # sorted desc by price
        self.ask_lvls: List[Tuple[float, float]] = []  # sorted asc by price
        self.prev_bsz: float = 0.0
        self.prev_asz: float = 0.0
        self.prev_bsz2: float = 0.0
        self.prev_asz2: float = 0.0
        self.prev_cum_bid3: float = 0.0
        self.prev_cum_ask3: float = 0.0
        self.prev_cum_bid5: float = 0.0
        self.prev_cum_ask5: float = 0.0

        # ---------- Time bookkeeping ----------
        self.last_ts: Optional[int] = None
        self._last_event_ts: Optional[int] = None
        self.last_trade_ts: Optional[int] = None
        self.last_bid1_update_ts: Optional[int] = None
        self.last_ask1_update_ts: Optional[int] = None

        # ---------- Rolling return histories ----------
        # Deques of (ts_ms, logret) to compute σ and VR
        self.ret_hist_20ms: Deque[Tuple[int, float]] = deque()
        self.ret_hist_25ms: Deque[Tuple[int, float]] = deque()
        self.ret_hist_50ms: Deque[Tuple[int, float]] = deque()
        self.ret_hist_100ms: Deque[Tuple[int, float]] = deque()
        self.ret_hist_250ms: Deque[Tuple[int, float]] = deque()
        self.ret_hist_500ms: Deque[Tuple[int, float]] = deque()
        self.ret_hist_1s: Deque[Tuple[int, float]] = deque()
        self.ret_hist_5s: Deque[Tuple[int, float]] = deque()

        self.regime_windows_ms: Tuple[int, ...] = (500, 1_000, 5_000)
        self.rv_ewma: Dict[int, float] = {ms: 0.0 for ms in self.regime_windows_ms}
        self.realized_vol: Dict[int, float] = {ms: 0.0 for ms in self.regime_windows_ms}
        self.volume_ewma: Dict[int, float] = {ms: 0.0 for ms in self.regime_windows_ms}
        self.flow_regime: Dict[int, float] = {ms: 0.0 for ms in self.regime_windows_ms}
        self._regime_return_deques: Dict[int, Deque[Tuple[int, float]]] = {
            500: self.ret_hist_500ms,
            1_000: self.ret_hist_1s,
            5_000: self.ret_hist_5s,
        }
        self.last_mid_for_ret: Optional[float] = None

        # ---------- Spread ----------
        self.last_spread: Optional[float] = None
        self.last_spread_ts: Optional[int] = None
        self.spread_delta_windows: Tuple[int, ...] = (25, 50, 100, 250, 500)
        self.spread_history: Deque[Tuple[int, float]] = deque()
        self.spread_history_window_ms: int = 1_000
        self.spread_changes_25ms: Deque[int] = deque()
        self.spread_changes_50ms: Deque[int] = deque()
        self.spread_changes_100ms: Deque[int] = deque()
        self.spread_changes_250ms: Deque[int] = deque()
        self.spread_changes_500ms: Deque[int] = deque()
        self.spread_changes_1s: Deque[int] = deque()
        self._spread_change_deques: Dict[int, Deque[int]] = {
            25: self.spread_changes_25ms,
            50: self.spread_changes_50ms,
            100: self.spread_changes_100ms,
            250: self.spread_changes_250ms,
            500: self.spread_changes_500ms,
            1000: self.spread_changes_1s,
        }

        # ---------- Best-level churn & depletion ----------
        self.bestlvl_windows: Tuple[int, ...] = (50, 100, 250, 500, 1_000)
        self._bid1_change_deques: Dict[int, Deque[int]] = {ms: deque() for ms in self.bestlvl_windows}
        self._ask1_change_deques: Dict[int, Deque[int]] = {ms: deque() for ms in self.bestlvl_windows}
        self.bid1_changes_1s = self._bid1_change_deques[1_000]
        self.ask1_changes_1s = self._ask1_change_deques[1_000]
        self.last_bid1 = None; self.last_ask1 = None
        self.neg_dbsz_250 = 0.0; self.neg_dasz_250 = 0.0
        self.sz_deltas_250ms: Deque[Tuple[int,float,float]] = deque()

        # ---------- Liquidity replenishment tracking (L1/L2) ----------
        self.replen_windows_ms: Tuple[int, ...] = (25, 50, 100, 250, 500)
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

        # ---------- Trades windows for short VWAP & bursts ----------
        # (ts, price, size, side, tick_sign, is_zero_tick)
        self.trades_25ms: Deque[Tuple[int, float, float, str, int, int]] = deque()
        self.trades_50ms: Deque[Tuple[int, float, float, str, int, int]] = deque()
        self.trades_100ms: Deque[Tuple[int, float, float, str, int, int]] = deque()
        self.trades_250ms: Deque[Tuple[int, float, float, str, int, int]] = deque()
        self.trades_500ms: Deque[Tuple[int, float, float, str, int, int]] = deque()
        self.trades_1s: Deque[Tuple[int, float, float, str, int, int]] = deque()
        self.trades_5s: Deque[Tuple[int, float, float, str, int, int]] = deque()

        self.trade_windows: Tuple[int, ...] = (25, 50, 100, 250, 500)
        self._trade_window_deques: Dict[int, Deque[Tuple[int, float, float, str, int, int]]] = {
            25: self.trades_25ms,
            50: self.trades_50ms,
            100: self.trades_100ms,
            250: self.trades_250ms,
            500: self.trades_500ms,
        }

        # Tick-direction & RPI tracking
        self.last_tick_sign: int = 0
        self.last_is_zero_tick: int = 0
        self.last_trade_price: Optional[float] = None
        self.last_is_rpi: int = 0

        # ---------- Quote windows (25/50/100/250/500/1000 ms) ----------
        self.quotes_25ms: Deque[int] = deque()
        self.quotes_50ms: Deque[int] = deque()
        self.quotes_100ms: Deque[int] = deque()
        self.quotes_250ms: Deque[int] = deque()
        self.quotes_500ms: Deque[int] = deque()
        self.quotes_1s: Deque[int] = deque()
        self._quote_window_deques: Dict[int, Deque[int]] = {
            25: self.quotes_25ms,
            50: self.quotes_50ms,
            100: self.quotes_100ms,
            250: self.quotes_250ms,
            500: self.quotes_500ms,
        }

        # ---------- Event density (25/100/250/500 ms) ----------
        self.ev_25ms: Deque[int] = deque()
        self.ev_100ms: Deque[int] = deque()
        self.ev_250ms: Deque[int] = deque()
        self.ev_500ms: Deque[int] = deque()
        self.ev_1s:    Deque[int] = deque()

        # ---------- VPIN state ----------
        self.vpin_Vb: Optional[float] = None          # dynamic bucket size (in base-volume)
        self.vpin_cum_buy: float = 0.0
        self.vpin_cum_sell: float = 0.0
        self.vpin_cum: float = 0.0
        self.vpin_phi: Deque[float] = deque(maxlen=50)

        # ---------- EWMAs for microprice and spread (short/med/long) ----------
        self.ema_mp_25: Optional[float] = None
        self.ema_mp_100: Optional[float] = None
        self.ema_mp_500: Optional[float] = None
        self.ema_sp_25: Optional[float] = None
        self.ema_sp_100: Optional[float] = None
        self.ema_sp_500: Optional[float] = None
        self.ema_sp_1000: Optional[float] = None

        # ---------- RSI on microprice (EWMA gains/losses) ----------
        self.rsi_gain: Optional[float] = None
        self.rsi_loss: Optional[float] = None

        # ---------- Decayed pressure (EWMA of OFI L1) ----------
        self.press_100ms: float = 0.0
        self.press_1s: float = 0.0
        self.press_5s: float = 0.0

        # ---------- Fast EMA state for microstructure signals ----------
        self.ema_half_lives_ms = (25, 100, 500)
        self.ema_indicator_names = (
            "bid1",
            "ask1",
            "mid",
            "spread",
            "spread_norm",
            "micro",
            "smart",
            "gap_a",
            "gap_b",
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

    def feature_dim(self) -> int:
        """Return feature dimension including Filiary channels."""
        if self._feat_dim is None:
            raise ValueError("Feature dimension unknown before first event")
        return self._feat_dim + AUX_DIM

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

    def _compute_trade_window_stats(
        self, trades: Deque[Tuple[int, float, float, str, int, int]], mid: float
    ) -> Dict[str, float]:
        buy_vol = sell_vol = 0.0
        buy_cnt = sell_cnt = 0
        buy_max = sell_max = 0.0
        trade_through = 0.0
        for entry in trades:
            _, price, size, side, *_ = entry
            if side == "buy":
                buy_vol += size
                buy_cnt += 1
                buy_max = max(buy_max, size)
                direction = 1.0
            else:
                sell_vol += size
                sell_cnt += 1
                sell_max = max(sell_max, size)
                direction = -1.0
            if mid > 0.0:
                trade_through += direction * ((price / mid) - 1.0)

        buy_mean = buy_vol / buy_cnt if buy_cnt > 0 else 0.0
        sell_mean = sell_vol / sell_cnt if sell_cnt > 0 else 0.0
        net_flow = buy_vol - sell_vol
        total_vol = buy_vol + sell_vol
        denom = max(total_vol, 1e-12)
        imbalance = net_flow / denom
        toxicity = abs(net_flow) / denom

        return {
            "buy_vol": buy_vol,
            "sell_vol": sell_vol,
            "buy_cnt": float(buy_cnt),
            "sell_cnt": float(sell_cnt),
            "buy_mean": buy_mean,
            "sell_mean": sell_mean,
            "buy_max": buy_max,
            "sell_max": sell_max,
            "net_flow": net_flow,
            "imbalance": imbalance,
            "toxicity": toxicity,
            "trade_through": trade_through if mid > 0.0 else 0.0,
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
        # cache sorted ladders (top N)
        self.bid_lvls = sorted(self.bids.items(), key=lambda x: x[0], reverse=True)[: self.depth]
        self.ask_lvls = sorted(self.asks.items(), key=lambda x: x[0], reverse=False)[: self.depth]

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

    def _event_density(self, deq: Deque[int], window_ms: int) -> float:
        if not deq:
            return 0.0
        now = deq[-1]
        self._prune_ts_deque(deq, now, window_ms)
        window_secs = window_ms / 1000.0
        return len(deq) / window_secs if window_secs > 0 else 0.0

    def event_density_25ms(self) -> float:
        return self._event_density(self.ev_25ms, 25)

    def event_density_100ms(self) -> float:
        # events per 0.1s
        return self._event_density(self.ev_100ms, 100)

    def event_density_250ms(self) -> float:
        return self._event_density(self.ev_250ms, 250)

    def event_density_500ms(self) -> float:
        return self._event_density(self.ev_500ms, 500)

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
                dq.append((ts_ms, value))
                sums_map[key] += value

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
        - Dict-like OB: {'type': 'snapshot'|'delta', 'data': {...}, 'ts': int, ...}
        - Dict-like trade: {'timestamp': float|str, 'price': str|float, 'size': str|float, 'side': 'Buy'|'Sell'|'buy'|'sell', ...}
        Returns: (etype, ts_ms, payload)
        etype in {'ob','trade'}
        """
        # Tuple form
        if isinstance(e, tuple) and len(e) == 4 and isinstance(e[0], str):
            etype = e[0].lower()
            ts_ms = int(e[1])
            payload = e[3]  # ignore sequence
            return etype, ts_ms, payload

        if isinstance(e, tuple) and len(e) == 3 and isinstance(e[0], str):
            etype = e[0].lower()
            ts_ms = int(e[2])
            return etype, ts_ms, e[1]

        if isinstance(e, dict):
            # OB event?
            if 'data' in e and 'ts' in e and ('orderbook' in str(e.get('topic','')) or e.get('type') in ('snapshot','delta')):
                ts_ms = int(e['ts'])
                return 'ob', ts_ms, e
            # Trade event?
            if 'timestamp' in e and 'price' in e and 'size' in e and 'side' in e:
                t_raw = e['timestamp']
                # Bybit sample shows seconds with fractional; convert to ms
                ts_ms = int(float(t_raw) * 1000.0)
                return 'trade', ts_ms, e

        raise ValueError(f"Unrecognized event shape: {type(e)} :: {e}")

    def _update_book_from_ob(self, ob_evt: dict):
        tp = ob_evt.get('type') or ob_evt.get('data', {}).get('type') or ob_evt.get('DataType') or 'delta'
        data = ob_evt.get('data', ob_evt)

        bids = data.get('b', [])
        asks = data.get('a', [])

        if tp == 'snapshot':
            self.bids = {float(p): float(q) for p, q in bids[: self.depth]}
            self.asks = {float(p): float(q) for p, q in asks[: self.depth]}
        else:  # delta
            for p, q in bids:
                p = float(p); q = float(q)
                if q == 0.0:
                    self.bids.pop(p, None)
                else:
                    self.bids[p] = q
            for p, q in asks:
                p = float(p); q = float(q)
                if q == 0.0:
                    self.asks.pop(p, None)
                else:
                    self.asks[p] = q

        self._sorted_ladders()

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

    def _update_trade_windows(self, ts_ms: int, trade_evt: dict, dt_ms: float):
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
            self._prune_deque_ms(deq, ts_ms, window)
        self.trades_1s.append(entry)
        self._prune_deque_ms(self.trades_1s, ts_ms, 1_000)
        self.trades_5s.append(entry)
        self._prune_deque_ms(self.trades_5s, ts_ms, 5_000)

        # Update volume-regime (vol/sec) EWMAs using provided dt_ms
        vol_rate = size / (dt_ms / 1000.0)  # base per second
        for hl in self.regime_windows_ms:
            self.volume_ewma[hl] = self._ewma_update(self.volume_ewma[hl], vol_rate, dt_ms, hl)

        # VPIN bucket sizing and accumulation
        v_per_sec = max(self.volume_ewma[1_000], 1e-9)
        Vb = max(v_per_sec * self.vpin_target_bucket_secs, 1e-9)
        self.vpin_Vb = Vb if self.vpin_Vb is None else (0.9 * self.vpin_Vb + 0.1 * Vb)

        if side == 'buy':
            self.vpin_cum_buy += size
        else:
            self.vpin_cum_sell += size
        self.vpin_cum += size

        # Close as many buckets as are filled (proportionally closing the last)
        while self.vpin_cum >= (self.vpin_Vb or 1e9):
            if self.vpin_Vb is None:
                break
            # proportionally split exactly Vb from current cum pools
            total = max(self.vpin_cum, 1e-12)
            scale = (self.vpin_Vb) / total
            buy_bucket = self.vpin_cum_buy * scale
            sell_bucket = self.vpin_cum_sell * scale
            phi = abs(buy_bucket - sell_bucket) / max(self.vpin_Vb, 1e-12)
            self.vpin_phi.append(phi)

            # subtract the closed bucket
            self.vpin_cum_buy -= buy_bucket
            self.vpin_cum_sell -= sell_bucket
            self.vpin_cum -= self.vpin_Vb

        self.last_trade_ts = ts_ms

    def _add_return(self, ts_ms: int, mid: float):
        if mid <= 0.0:
            return 0.0
        if self.last_mid_for_ret is None:
            self.last_mid_for_ret = mid
            return 0.0
        r = math.log(mid / self.last_mid_for_ret) if self.last_mid_for_ret > 0 else 0.0
        self.last_mid_for_ret = mid

        # push to windows
        self.ret_hist_20ms.append((ts_ms, r))
        self.ret_hist_25ms.append((ts_ms, r))
        self.ret_hist_50ms.append((ts_ms, r))
        self.ret_hist_100ms.append((ts_ms, r))
        self.ret_hist_250ms.append((ts_ms, r))
        self.ret_hist_500ms.append((ts_ms, r))
        self.ret_hist_1s.append((ts_ms, r))
        self.ret_hist_5s.append((ts_ms, r))

        self._prune_deque_ms(self.ret_hist_20ms, ts_ms, 20)
        self._prune_deque_ms(self.ret_hist_25ms, ts_ms, 25)
        self._prune_deque_ms(self.ret_hist_50ms, ts_ms, 50)
        self._prune_deque_ms(self.ret_hist_100ms, ts_ms, 100)
        self._prune_deque_ms(self.ret_hist_250ms, ts_ms, 250)
        self._prune_deque_ms(self.ret_hist_500ms, ts_ms, 500)
        self._prune_deque_ms(self.ret_hist_1s, ts_ms, 1_000)
        self._prune_deque_ms(self.ret_hist_5s, ts_ms, 5_000)

        # update short-horizon EWMA and realized vol caches
        dt_ms = 1.0 if self._last_event_ts is None else max(1.0, ts_ms - self._last_event_ts)
        r2 = r * r
        for hl in self.regime_windows_ms:
            self.rv_ewma[hl] = self._ewma_update(self.rv_ewma[hl], r2, dt_ms, hl)

        for ms, deq in self._regime_return_deques.items():
            self.realized_vol[ms] = math.sqrt(sum(val * val for _, val in deq))
        return r

    def _stats_from_returns(self, deq: Deque[Tuple[int, float]]) -> Tuple[float, float]:
        """Return (mean, variance) of returns in a deque window."""
        n = len(deq)
        if n <= 1:
            return 0.0, 0.0
        vals = [x for _, x in deq]
        m = float(sum(vals) / n)
        var = float(sum((v - m) * (v - m) for v in vals) / (n - 1))
        return m, var

    def _zscore(self, x: np.ndarray, dt_ms: float) -> np.ndarray:
        """Per-feature EWMA mean/var rolling z-score."""
        eps = 1e-9
        if self._feat_dim is None:
            self._feat_dim = int(x.shape[0])
            self.z_mean = x.astype(np.float64).copy()
            self.z_m2 = (x.astype(np.float64) ** 2).copy()
            return np.zeros_like(x, dtype=np.float32)

        hl = self._alpha_half_life_ms(self.z_hl_ms)
        alpha = 1.0 - math.pow(0.5, max(1.0, dt_ms) / float(hl))

        # Update EWMA mean and second moment
        self.z_mean = (1.0 - alpha) * self.z_mean + alpha * x
        self.z_m2   = (1.0 - alpha) * self.z_m2 + alpha * (x * x)
        var = np.maximum(self.z_m2 - self.z_mean * self.z_mean, eps)
        z = (x - self.z_mean) / np.sqrt(var)
        return z.astype(np.float32)

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------
    def on_event(self, e: Any) -> Tuple[int, np.ndarray, float, bool, float]:
        """
        Process a single merged event and return:
            ts_ms, feature_vector (z-scored), mid, is_trade, dt_ms
        """
        etype, ts_ms, payload = self._parse_event(e)
        dt_ms = 1.0 if self._last_event_ts is None else max(1.0, ts_ms - self._last_event_ts)
        prev_bid_l1 = self.prev_bsz
        prev_ask_l1 = self.prev_asz
        prev_bid_l2 = self.prev_bsz2
        prev_ask_l2 = self.prev_asz2

        self._prune_replen_windows(ts_ms)

        # Event density
        self.ev_25ms.append(ts_ms)
        self._prune_ts_deque(self.ev_25ms, ts_ms, 25)
        self.ev_100ms.append(ts_ms)
        self._prune_ts_deque(self.ev_100ms, ts_ms, 100)
        self.ev_250ms.append(ts_ms)
        self._prune_ts_deque(self.ev_250ms, ts_ms, 250)
        self.ev_500ms.append(ts_ms)
        self._prune_ts_deque(self.ev_500ms, ts_ms, 500)
        self.ev_1s.append(ts_ms)
        self._prune_ts_deque(self.ev_1s, ts_ms, 1000)

        # Update book/trades
        is_trade = (etype == 'trade')
        if etype == 'ob':
            self._update_book_from_ob(payload)
            for window, deq in self._quote_window_deques.items():
                deq.append(ts_ms)
                self._prune_ts_deque(deq, ts_ms, window)
            self.quotes_1s.append(ts_ms)
            self._prune_ts_deque(self.quotes_1s, ts_ms, 1_000)
        else:
            self._update_trade_windows(ts_ms, payload, dt_ms)

        # Compute basic ladders + best quotes
        self._sorted_ladders()
        bid1, ask1, bsz1, asz1 = self._book_best()
        mid = 0.5 * (bid1 + ask1) if (bid1 > 0 and ask1 > 0) else 0.0

        # Microprice and SmartPrice (inverse-size weighting)
        if (bsz1 + asz1) > 0:
            micro = (ask1 * bsz1 + bid1 * asz1) / (bsz1 + asz1)
            # smart weights: w_b = 1/bsz, w_a = 1/asz
            wb = 1.0 / max(bsz1, 1e-12)
            wa = 1.0 / max(asz1, 1e-12)
            smart = (bid1 * wb + ask1 * wa) / (wb + wa)
        else:
            micro = smart = mid

        spread = max(0.0, ask1 - bid1)
        spread_norm = spread / max(mid, 1e-9)

        self._prune_deque_ms(self.spread_history, ts_ms, self.spread_history_window_ms)
        spread_deltas: Dict[int, float] = {}
        for window in self.spread_delta_windows:
            target = ts_ms - window
            ref_val = None
            for t_prev, spread_prev in reversed(self.spread_history):
                ref_val = spread_prev
                if t_prev <= target:
                    break
            if ref_val is None:
                ref_val = spread
            spread_deltas[window] = spread - ref_val
        self.spread_history.append((ts_ms, spread))

        # Gaps (best->second)
        ask2 = self.ask_lvls[1][0] if len(self.ask_lvls) > 1 else ask1
        bid2 = self.bid_lvls[1][0] if len(self.bid_lvls) > 1 else bid1
        gap_a = max(0.0, ask2 - ask1)
        gap_b = max(0.0, bid1 - bid2)

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

        # Cum depths
        cum_bid1 = self._cum_depth(self.bid_lvls, 1)
        cum_ask1 = self._cum_depth(self.ask_lvls, 1)
        cum_bid3 = self._cum_depth(self.bid_lvls, 3)
        cum_ask3 = self._cum_depth(self.ask_lvls, 3)
        cum_bid5 = self._cum_depth(self.bid_lvls, 5)
        cum_ask5 = self._cum_depth(self.ask_lvls, 5)
        cum_bid10 = self._cum_depth(self.bid_lvls, 10)
        cum_ask10 = self._cum_depth(self.ask_lvls, 10)

        # OFI (L1/L3/L5)
        ofi_l1 = (bsz1 - self.prev_bsz) - (asz1 - self.prev_asz)
        ofi_l3 = (cum_bid3 - self.prev_cum_bid3) - (cum_ask3 - self.prev_cum_ask3)
        ofi_l5 = (cum_bid5 - self.prev_cum_bid5) - (cum_ask5 - self.prev_cum_ask5)
        self.prev_bsz, self.prev_asz = bsz1, asz1
        self.prev_bsz2, self.prev_asz2 = bsz2, asz2
        self.prev_cum_bid3, self.prev_cum_ask3 = cum_bid3, cum_ask3
        self.prev_cum_bid5, self.prev_cum_ask5 = cum_bid5, cum_ask5

        # OBI (L1, L3 and L5)
        obi_l1 = (bsz1 - asz1) / max(bsz1 + asz1, 1e-12)
        obi_l3 = (cum_bid3 - cum_ask3) / max(cum_bid3 + cum_ask3, 1e-12)
        obi_l5 = (cum_bid5 - cum_ask5) / max(cum_bid5 + cum_ask5, 1e-12)

        # Micro-premia vs spread
        micro_premia = (micro - mid) / max(spread, 1e-9)
        smart_premia = (smart - mid) / max(spread, 1e-9)

        # Slopes/shape (top-K vs distance from mid)
        xb, yb = self._levels_to_xy(self.bid_lvls, mid, True, 5)
        xa, ya = self._levels_to_xy(self.ask_lvls, mid, False, 5)
        slope_b = self._lin_slope(xb, yb)
        slope_a = self._lin_slope(xa, ya)

        indicator_values = {
            "bid1": bid1,
            "ask1": ask1,
            "mid": mid,
            "spread": spread,
            "spread_norm": spread_norm,
            "micro": micro,
            "smart": smart,
            "gap_a": gap_a,
            "gap_b": gap_b,
            "cum_bid1": cum_bid1,
            "cum_ask1": cum_ask1,
            "cum_bid3": cum_bid3,
            "cum_ask3": cum_ask3,
            "slope_a": slope_a,
            "slope_b": slope_b,
            "obi_l1": obi_l1,
            "obi_l3": obi_l3,
            "obi_l5": obi_l5,
            "ofi_l1": ofi_l1,
            "ofi_l3": ofi_l3,
            "ofi_l5": ofi_l5,
            "micro_premia": micro_premia,
            "smart_premia": smart_premia,
        }
        self._update_indicator_emas(indicator_values, dt_ms)

        # Pressure (EWMA of OFI L1)
        self.press_50ms  = self._ewma_update(getattr(self, 'press_50ms', 0.0), ofi_l1, dt_ms,  50)
        self.press_100ms = self._ewma_update(self.press_100ms, ofi_l1, dt_ms, 100)
        self.press_250ms = self._ewma_update(getattr(self, 'press_250ms',0.0), ofi_l1, dt_ms, 250)
        self.press_1s    = self._ewma_update(self.press_1s,    ofi_l1, dt_ms, 1_000)
        self.press_2s    = self._ewma_update(getattr(self, 'press_2s',   0.0), ofi_l1, dt_ms, 2_000)

        # Short VWAPs (25/50/100/250/500 ms) and rel to mid/micro
        def vwap_in(win: Deque[Tuple[int, float, float, str, int, int]]) -> float:
            vol = sum(s for _, _, s, *_ in win)
            if vol <= 1e-12:
                return mid
            pxv = sum(p * s for _, p, s, *_ in win)
            return pxv / vol

        vwap_per_ms: Dict[int, float] = {
            ms: vwap_in(self._trade_window_deques[ms]) for ms in self.trade_windows
        }
        vwap_vs_mid = {
            ms: (vwap_per_ms[ms] / max(mid, 1e-12)) - 1.0 if mid > 0 else 0.0
            for ms in self.trade_windows
        }
        vwap_vs_micro = {
            ms: (vwap_per_ms[ms] / max(micro, 1e-12)) - 1.0 if micro > 0 else 0.0
            for ms in self.trade_windows
        }

        # Trade stats per horizon and 1s tempo
        trade_stats = {
            ms: self._compute_trade_window_stats(self._trade_window_deques[ms], mid)
            for ms in self.trade_windows
        }
        stats_1s = self._compute_trade_window_stats(self.trades_1s, mid)
        stats_5s = self._compute_trade_window_stats(self.trades_5s, mid)

        # Quote/trade tempo (1s windows)
        quotes_1s = float(len(self.quotes_1s))
        trade_cnt_100ms = float(trade_stats[100]["buy_cnt"] + trade_stats[100]["sell_cnt"])
        trade_cnt_1s = float(stats_1s["buy_cnt"] + stats_1s["sell_cnt"])
        buy_vol_1s = stats_1s["buy_vol"]
        sell_vol_1s = stats_1s["sell_vol"]
        buy_cnt_1s = stats_1s["buy_cnt"]
        sell_cnt_1s = stats_1s["sell_cnt"]
        buy_mean_1s = stats_1s["buy_mean"]
        sell_mean_1s = stats_1s["sell_mean"]
        buy_max_1s = stats_1s["buy_max"]
        sell_max_1s = stats_1s["sell_max"]
        tot_1s = buy_vol_1s + sell_vol_1s
        trade_imb_1s = stats_1s["imbalance"]
        net_vol_1s = stats_1s["net_flow"]
        trade_through_1s = stats_1s["trade_through"]

        regime_flow = {
            500: trade_stats.get(500, stats_1s)["imbalance"],
            1_000: stats_1s["imbalance"],
            5_000: stats_5s["imbalance"],
        }
        for ms, val in regime_flow.items():
            self.flow_regime[ms] = val

        # Spread
        if self.last_spread is None or spread != self.last_spread:
            for window, deq in self._spread_change_deques.items():
                deq.append(ts_ms)
                self._prune_ts_deque(deq, ts_ms, window)
            self.last_spread = spread
            self.last_spread_ts = ts_ms
        else:
            for window, deq in self._spread_change_deques.items():
                self._prune_ts_deque(deq, ts_ms, window)

        # Best-level churn & depletion
        bid_level_changed = (
            self.last_bid1 is None
            or bid1 != self.last_bid1
            or bsz1 != prev_bid_l1
        )
        ask_level_changed = (
            self.last_ask1 is None
            or ask1 != self.last_ask1
            or asz1 != prev_ask_l1
        )

        for window, dq in self._bid1_change_deques.items():
            if bid_level_changed:
                dq.append(ts_ms)
            self._prune_ts_deque(dq, ts_ms, window)

        for window, dq in self._ask1_change_deques.items():
            if ask_level_changed:
                dq.append(ts_ms)
            self._prune_ts_deque(dq, ts_ms, window)

        if bid_level_changed:
            self.last_bid1_update_ts = ts_ms
        if ask_level_changed:
            self.last_ask1_update_ts = ts_ms

        self.last_bid1, self.last_ask1 = bid1, ask1

        # size depletion (only negative deltas accumulated over 250ms)
        db = bsz1 - self.prev_bsz
        da = asz1 - self.prev_asz
        self.sz_deltas_250ms.append((ts_ms, min(db,0.0), min(da,0.0)))
        self._prune_deque_ms(self.sz_deltas_250ms, ts_ms, 250)
        neg_depl_b = sum(x for _, x, _ in self.sz_deltas_250ms)
        neg_depl_a = sum(x for _, _, x in self.sz_deltas_250ms)

        dt_since_trade = float(ts_ms - self.last_trade_ts) if self.last_trade_ts is not None else 0.0
        dt_since_bid1_update = (
            float(ts_ms - self.last_bid1_update_ts)
            if self.last_bid1_update_ts is not None
            else 0.0
        )
        dt_since_ask1_update = (
            float(ts_ms - self.last_ask1_update_ts)
            if self.last_ask1_update_ts is not None
            else 0.0
        )

        quote_counts = {ms: len(self._quote_window_deques[ms]) for ms in self.trade_windows}
        spread_change_counts = {ms: len(self._spread_change_deques[ms]) for ms in self.trade_windows}
        bid1_change_counts = {ms: len(self._bid1_change_deques[ms]) for ms in self.bestlvl_windows}
        ask1_change_counts = {ms: len(self._ask1_change_deques[ms]) for ms in self.bestlvl_windows}
        time_since_spread_change = (ts_ms - (self.last_spread_ts or ts_ms))
        n_spread_chg_250ms = spread_change_counts.get(250, 0)
        n_spread_chg_1s = len(self._spread_change_deques[1000])

        # Returns & vol stats (populate histories + compute σ and VR)
        self._add_return(ts_ms, mid)
        _, var_20 = self._stats_from_returns(self.ret_hist_20ms)
        _, var_25 = self._stats_from_returns(self.ret_hist_25ms)
        _, var_50 = self._stats_from_returns(self.ret_hist_50ms)
        _, var_100 = self._stats_from_returns(self.ret_hist_100ms)
        _, var_250 = self._stats_from_returns(self.ret_hist_250ms)
        _, var_500 = self._stats_from_returns(self.ret_hist_500ms)
        _, var_1s = self._stats_from_returns(self.ret_hist_1s)
        _, var_5s = self._stats_from_returns(self.ret_hist_5s)
        std_20 = math.sqrt(max(0.0, var_20))
        std_25 = math.sqrt(max(0.0, var_25))
        std_50 = math.sqrt(max(0.0, var_50))
        std_100 = math.sqrt(max(0.0, var_100))
        std_250 = math.sqrt(max(0.0, var_250))
        std_500 = math.sqrt(max(0.0, var_500))
        std_1s = math.sqrt(max(0.0, var_1s))
        std_5s = math.sqrt(max(0.0, var_5s))
        # Variance ratio: 1s variance vs 10 * 100ms variance (1s = 10×100ms)
        vr = (var_1s / max(10.0 * var_100, 1e-12)) if var_100 > 0 else 0.0
        vr_1s_250 = var_1s / max(4.0 * var_250, 1e-12) if var_250 > 0 else 0.0
        var_ratio_500_100 = var_500 / max(var_100, 1e-12) if var_100 > 0 else 0.0
        var_ratio_100_20 = var_100 / max(var_20, 1e-12) if var_20 > 0 else 0.0

        # Short-horizon regime summaries (vol & flow)
        regime_vol_ewma = {ms: math.sqrt(max(self.rv_ewma[ms], 1e-18)) for ms in self.regime_windows_ms}
        regime_realized = {ms: self.realized_vol[ms] for ms in self.regime_windows_ms}
        regime_volume = {ms: self.volume_ewma[ms] for ms in self.regime_windows_ms}
        regime_flow_snapshot = {ms: self.flow_regime[ms] for ms in self.regime_windows_ms}
        vol_ratio_short_long = regime_volume[500] / max(regime_volume[5_000], 1e-9)
        realized_ratio_short_long = regime_realized[500] / max(regime_realized[5_000], 1e-12)
        flow_diff_short_long = regime_flow_snapshot[500] - regime_flow_snapshot[5_000]

        # EMA (microprice & spread) and RSI(micro)
        for attr, val, hl, use_half_life in [
            ("ema_mp_25", micro, 50, False),
            ("ema_mp_100", micro, 200, False),
            ("ema_mp_500", micro, 800, False),
            ("ema_sp_25", spread, 50, False),
            ("ema_sp_100", spread, 100, True),
            ("ema_sp_500", spread, 500, True),
            ("ema_sp_1000", spread, 1000, True),
        ]:
            cur = getattr(self, attr)
            if use_half_life:
                prev = cur if cur is not None else val
                updated = self._ewma_update(prev, val, dt_ms, hl)
            else:
                alpha = 1.0 - math.exp(-dt_ms / float(max(1, hl)))
                updated = (1.0 - alpha) * (cur if cur is not None else val) + alpha * val
            setattr(self, attr, updated)

        spread_ema_vals = {
            100: self.ema_sp_100 if self.ema_sp_100 is not None else spread,
            500: self.ema_sp_500 if self.ema_sp_500 is not None else spread,
            1000: self.ema_sp_1000 if self.ema_sp_1000 is not None else spread,
        }
        spread_ratios = {
            hl: spread / max(val, 1e-9)
            for hl, val in spread_ema_vals.items()
        }
        spread_log_ratios = {
            hl: math.log(max(spread, 1e-9)) - math.log(max(val, 1e-9))
            for hl, val in spread_ema_vals.items()
        }

        spread_delta_norms: Dict[int, float] = {}
        for window, delta in spread_deltas.items():
            if window <= 50:
                baseline = self.ema_sp_25 if self.ema_sp_25 is not None else spread
            elif window <= 150:
                baseline = self.ema_sp_100 if self.ema_sp_100 is not None else spread
            else:
                baseline = self.ema_sp_500 if self.ema_sp_500 is not None else spread
            spread_delta_norms[window] = delta / max(baseline, 1e-9)

        # RSI on microprice (use EWMA gains/losses with ~100ms smoothing)
        delta_mp = micro - (self.ema_mp_25 if self.ema_mp_25 is not None else micro)
        micro_minus_ema25 = delta_mp
        micro_minus_ema100 = micro - (self.ema_mp_100 if self.ema_mp_100 is not None else micro)
        gain = max(delta_mp, 0.0)
        loss = max(-delta_mp, 0.0)
        alpha_rsi = 1.0 - math.exp(-dt_ms / 200.0)
        self.rsi_gain = (1.0 - alpha_rsi) * (self.rsi_gain or 0.0) + alpha_rsi * gain
        self.rsi_loss = (1.0 - alpha_rsi) * (self.rsi_loss or 0.0) + alpha_rsi * loss
        rs = self.rsi_gain / max(self.rsi_loss, 1e-12)
        rsi = 100.0 - 100.0 / (1.0 + rs)

        # MACD/CCI on μ & SmartPrice (use ms-based EMAs)
        def ema_ms(prev: float, x: float, hl_ms: float) -> float:
            a = 1.0 - math.exp(-dt_ms / max(1.0, hl_ms))
            return (1 - a) * prev + a * x if prev is not None else x

        # MACD for microprice
        self.macd_fast = ema_ms(getattr(self, 'macd_fast', None), micro, 150.0)
        self.macd_slow = ema_ms(getattr(self, 'macd_slow', None), micro, 450.0)
        macd_raw = (self.macd_fast - self.macd_slow) if (self.macd_fast is not None and self.macd_slow is not None) else 0.0
        self.macd_sig = ema_ms(getattr(self, 'macd_sig', None), macd_raw, 250.0)
        macd_hist = macd_raw - (self.macd_sig if self.macd_sig is not None else 0.0)

        # CCI (micro) using EWMA mean & mean abs dev proxy
        self.cci_mean = ema_ms(getattr(self, 'cci_mean', None), micro, 200.0)
        mad = abs(micro - (self.cci_mean if self.cci_mean is not None else micro))
        self.cci_mad = ema_ms(getattr(self, 'cci_mad', None), mad, 200.0)
        cci = 0.015 * ((micro - (self.cci_mean or micro)) / max(self.cci_mad or 1e-12, 1e-12))

        # VPIN value (avg of last M φ)
        vpin = (sum(self.vpin_phi) / len(self.vpin_phi)) if self.vpin_phi else 0.0

        horizon_list = list(self.trade_windows)
        replen_rates = self._replenishment_rates()

        # Build raw feature vector
        feat_list = [
            # --- price/state ---
            bid1, ask1, mid, micro, smart, spread, spread_norm, gap_a, gap_b,
            bsz1, asz1,

            # --- depth/shape ---
            cum_bid5, cum_ask5, cum_bid10, cum_ask10,
            slope_b, slope_a,

            # --- imbalance & premia ---
            ofi_l1, ofi_l3, ofi_l5,
            obi_l1, obi_l3, obi_l5,
            micro_premia, smart_premia,

            # --- pressure (decayed OFI) ---
            getattr(self, 'press_50ms', 0.0),
            self.press_100ms,
            getattr(self, 'press_250ms', 0.0),
            self.press_1s,
            getattr(self, 'press_2s', 0.0),

            # --- trade flow (true ~1s) ---
            buy_vol_1s, sell_vol_1s, buy_cnt_1s, sell_cnt_1s,
            buy_mean_1s, sell_mean_1s, buy_max_1s, sell_max_1s,
            trade_imb_1s, net_vol_1s, trade_through_1s,

            # --- tempo ---
            quotes_1s,
            trade_cnt_100ms,
            trade_cnt_1s,
            self.event_density_25ms(),
            self.event_density_100ms(),  # events per 0.1s (helper)
            self.event_density_250ms(),
            self.event_density_500ms(),
            dt_since_trade,
            float(self.last_tick_sign),
            float(self.last_is_zero_tick),
            float(self.last_is_rpi),

            # --- best-level churn & depletion & spread-change stats ---
            float(bid1_change_counts.get(50, 0)),
            float(bid1_change_counts.get(100, 0)),
            float(bid1_change_counts.get(250, 0)),
            float(bid1_change_counts.get(500, 0)),
            float(bid1_change_counts.get(1_000, 0)),
            float(ask1_change_counts.get(50, 0)),
            float(ask1_change_counts.get(100, 0)),
            float(ask1_change_counts.get(250, 0)),
            float(ask1_change_counts.get(500, 0)),
            float(ask1_change_counts.get(1_000, 0)),
            neg_depl_b, neg_depl_a,
            float(time_since_spread_change),
            float(n_spread_chg_250ms), float(n_spread_chg_1s),
            dt_since_bid1_update,
            dt_since_ask1_update,
            spread_deltas.get(25, 0.0), spread_deltas.get(50, 0.0),
            spread_deltas.get(100, 0.0), spread_deltas.get(250, 0.0),
            spread_deltas.get(500, 0.0),
            spread_delta_norms.get(25, 0.0), spread_delta_norms.get(50, 0.0),
            spread_delta_norms.get(100, 0.0), spread_delta_norms.get(250, 0.0),
            spread_delta_norms.get(500, 0.0),

            # --- returns & vol stats ---
            std_20, std_25, std_50, std_100, std_250, std_500, std_1s, std_5s,
            vr, vr_1s_250,
            var_ratio_500_100, var_ratio_100_20,

            # --- EMAs & technicals ---
            (self.ema_mp_25 if self.ema_mp_25 is not None else micro),
            (self.ema_mp_100 if self.ema_mp_100 is not None else micro),
            (self.ema_mp_500 if self.ema_mp_500 is not None else micro),
            (self.ema_sp_25 if self.ema_sp_25 is not None else spread),
            (self.ema_sp_100 if self.ema_sp_100 is not None else spread),
            (self.ema_sp_500 if self.ema_sp_500 is not None else spread),
            (self.ema_sp_1000 if self.ema_sp_1000 is not None else spread),
            spread_ratios[100], spread_ratios[500], spread_ratios[1000],
            spread_log_ratios[100], spread_log_ratios[500], spread_log_ratios[1000],
            micro_minus_ema25,
            micro_minus_ema100,
            rsi,
            macd_raw,
            (self.macd_sig if getattr(self, 'macd_sig', None) is not None else 0.0),
            macd_hist,
            cci,

            # --- regime & risks ---
            self.volume_ewma[500], self.volume_ewma[1_000], self.volume_ewma[5_000],
            regime_realized[500], regime_realized[1_000], regime_realized[5_000],
            regime_vol_ewma[500], regime_vol_ewma[1_000], regime_vol_ewma[5_000],
            self.flow_regime[500], self.flow_regime[1_000], self.flow_regime[5_000],
            vol_ratio_short_long, realized_ratio_short_long, flow_diff_short_long,
            vpin,
        ]

        # --- per-horizon trade stats & liquidity metrics (25/50/100/250/500 ms) ---
        for ms in horizon_list:
            stats = trade_stats[ms]
            feat_list.extend([
                stats["buy_vol"], stats["sell_vol"],
                stats["buy_cnt"], stats["sell_cnt"],
                stats["buy_mean"], stats["sell_mean"],
                stats["buy_max"], stats["sell_max"],
                stats["net_flow"], stats["imbalance"],
                stats["toxicity"], stats["trade_through"],
                float(quote_counts.get(ms, 0)),
                float(spread_change_counts.get(ms, 0)),
            ])

        # --- L1/L2 liquidity replenishment/cancellation rates (size/ms) ---
        for ms in horizon_list:
            rates = replen_rates[ms]
            for level in (1, 2):
                feat_list.extend([
                    rates[("bid", level, "add")],
                    rates[("bid", level, "rem")],
                    rates[("ask", level, "add")],
                    rates[("ask", level, "rem")],
                ])

        # --- VWAPs vs mid/micro across horizons ---
        for ms in horizon_list:
            feat_list.append(vwap_vs_mid[ms])
        for ms in horizon_list:
            feat_list.append(vwap_vs_micro[ms])

        # --- indicator EMAs (25/100/500 ms) ---
        for hl in self.ema_half_lives_ms:
            state = self.ema_states[hl]
            for name in self.ema_indicator_names:
                ema_val = state[name]
                if ema_val is None:
                    ema_val = indicator_values[name]
                feat_list.append(ema_val)

        # --- indicator EMA residuals (raw - EMA) ---
        for hl in self.ema_half_lives_ms:
            state = self.ema_states[hl]
            for name in self.ema_indicator_names:
                ema_val = state[name]
                if ema_val is None:
                    ema_val = indicator_values[name]
                feat_list.append(indicator_values[name] - ema_val)

        feat = np.array(feat_list, dtype=np.float64)

        # Rolling z-score normalization
        feat_z = self._zscore(feat, dt_ms)

        # Update end-of-event time markers
        self.last_ts = ts_ms
        self._last_event_ts = ts_ms

        return ts_ms, feat_z, mid, is_trade, dt_ms


class LabelBuilder:
    def __init__(self, delta_ms: int = 5, horizons_ms: Optional[List[int]] = None):
        self.delta = int(delta_ms)
        self.horizons = sorted(horizons_ms if horizons_ms is not None else HORIZONS_MS)
        assert len(self.horizons) > 0, "At least one horizon required"
        self.max_h = int(self.horizons[-1])

        # Decisions waiting to reach t_delta (entry time): items are t_delta timestamps
        self.wait_delta: Deque[int] = deque()
        # Decisions with entry price recorded, waiting to reach final maturity:
        # entries are (t_ready, t_delta, mid_entry)
        self.wait_mature: Deque[Tuple[int, int, float]] = deque()

        # Maintain recent midprice history as (timestamp, mid)
        self.price_history: Deque[Tuple[int, float]] = deque()
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
            vols = []
            for horizon in self.horizons:
                mid_T = self._price_at(t_delta + int(horizon))
                mid_T_safe = max(e, mid_T)
                y_ret = math.log(mid_T_safe / mid0_safe)
                returns.append(y_ret)
                vols.append(0.5 * (y_ret ** 2))

            out.append(np.array(returns + vols, dtype=np.float32))

        self.last_ts = t
        self.last_mid = m
        return out

    # ---- price history helpers ----
    def _record_price(self, t: int, m: float):
        self.price_history.append((t, m))
        cutoff = t - self.history_span
        while len(self.price_history) > 1 and self.price_history[0][0] < cutoff:
            self.price_history.popleft()

    def _price_at(self, t_query: int) -> float:
        if not self.price_history:
            return self.last_mid if self.last_mid is not None else 0.0

        for ts, mid in reversed(self.price_history):
            if ts <= t_query:
                return mid

        # If query precedes the oldest stored timestamp, fall back to the earliest mid
        return self.price_history[0][1]


class HFTDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = X.astype(np.float32)
        self.y = y.astype(np.float32)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        return torch.from_numpy(self.X[idx]), torch.from_numpy(self.y[idx])
    

def build_sequence_from_tokens(tokens: Deque[np.ndarray], lookback: int) -> np.ndarray:
    """
    Build a fixed-length [L, F] sequence from a deque of tokens (each 1D np.array of size F).
    - If len(tokens) >= L: trim older (deque already keeps last L if maxlen=L).
    - If len(tokens) <  L: left-pad by repeating the earliest token.
      Important: we set aux Δt for pads to 0 so padding doesn't distort time/CPC.
    """
    assert len(tokens) >= 1
    if len(tokens) >= lookback:
        return np.stack(list(tokens), axis=0)

    pad_n = lookback - len(tokens)
    first = tokens[0].copy()
    # last channels are [log_dt_ms, is_trade, events_100ms] — leave is_trade=0, density=0 for pads
    first[-3] = PAD_DT_FOR_LEFT  # log_dt_ms (log1p(0) = 0)
    first[-2] = 0.0              # is_trade
    first[-1] = 0.0              # events_100ms
    pad_block = np.repeat(first[None, :], pad_n, axis=0)
    arr = np.stack(list(tokens), axis=0)
    return np.concatenate([pad_block, arr], axis=0)


def stream_bybit(week_files: List[Tuple[str, str]]) -> Tuple[np.ndarray, np.ndarray, int]:
    fe = FeatureEngine()
    labeler = LabelBuilder(delta_ms=5, horizons_ms=HORIZONS_MS)

    tokens: Deque[np.ndarray] = deque(maxlen=LOOKBACK)  # token = [features..., log_dt_ms, is_trade, events_100ms]
    pending_seqs: Deque[np.ndarray] = deque()           # sequences waiting for labels (FIFO)

    X_list: List[np.ndarray] = []
    y_list: List[np.ndarray] = []

    parsed_weeks: List[Tuple[datetime, datetime, str, str]] = []
    for ob_zip, th_zip in week_files:
        ob_base = os.path.basename(ob_zip)
        th_base = os.path.basename(th_zip)

        # Normalise prefix so both files parse through _parse_week_key_any
        ob_key = ob_base.replace("BTCUSDT_TH_", "BTCUSDT_OB_", 1)
        th_key = th_base.replace("BTCUSDT_TH_", "BTCUSDT_OB_", 1)

        try:
            start_ob, end_ob, _ = _parse_week_key_any(ob_key)
        except ValueError as exc:
            raise ValueError(f"Failed to parse week range from OB file '{ob_base}': {exc}") from exc

        try:
            start_th, end_th, _ = _parse_week_key_any(th_key)
        except ValueError as exc:
            raise ValueError(f"Failed to parse week range from TH file '{th_base}': {exc}") from exc

        if (start_ob, end_ob) != (start_th, end_th):
            raise ValueError(
                "Mismatch between OB/TH week ranges: "
                f"OB='{ob_base}' ({start_ob.date()}→{end_ob.date()}) vs "
                f"TH='{th_base}' ({start_th.date()}→{end_th.date()})"
            )

        parsed_weeks.append((start_ob, end_ob, ob_zip, th_zip))

    for idx in range(1, len(parsed_weeks)):
        prev_start, prev_end, prev_ob, prev_th = parsed_weeks[idx - 1]
        curr_start, curr_end, curr_ob, curr_th = parsed_weeks[idx]
        if curr_end <= prev_end:
            raise ValueError(
                "Week files must be strictly increasing by end date: "
                f"'{os.path.basename(curr_ob)}'/'{os.path.basename(curr_th)}' (end={curr_end.date()}) "
                f"not after '{os.path.basename(prev_ob)}'/'{os.path.basename(prev_th)}' (end={prev_end.date()})"
            )

    total_weeks = len(parsed_weeks)
    last_global_ts: Optional[int] = None
    for w_idx, (start_dt, end_dt, ob_zip, th_zip) in enumerate(parsed_weeks, 1):
        print(f"[week {w_idx}/{total_weeks}] OB={os.path.basename(ob_zip)} | TH={os.path.basename(th_zip)}")
        raw = BybitRawIter(ob_zip, th_zip)
        merged_iter = merge_event_time(raw.ob_iter(), raw.trade_iter(), B=0)

        last_log = time.time()
        event_count = 0
        last_ts_ms = None

        def process_event(e: Any) -> int:
            nonlocal event_count, last_ts_ms, last_log

            event_count += 1

            # 1) Update features with this event
            ts_ms, feat, mid, is_trade, dt_ms = fe.on_event(e)
            last_ts_ms = ts_ms

            # 2) Build token with aux channels
            events_100ms = fe.event_density_100ms()
            log_dt_ms = float(np.log1p(dt_ms))
            token = np.concatenate(
                [feat, np.array([log_dt_ms, float(is_trade), events_100ms], dtype=np.float32)]
            ).astype(np.float32)
            tokens.append(token)

            # 3) Build sequence at every event
            seq = build_sequence_from_tokens(tokens, LOOKBACK)
            pending_seqs.append(seq)

            # 4–6) Labels
            labeler.on_decision(int(ts_ms))
            matured_list = labeler.on_event(int(ts_ms), float(mid))
            for y in matured_list:
                if pending_seqs:
                    X_list.append(pending_seqs.popleft())
                    y_list.append(y.astype(np.float32))
                else:
                    # Shouldn't happen with FIFO; guard anyway
                    pass

            if event_count % 500_000 == 0 or (time.time() - last_log) > 5:
                print(f"  [progress] events={event_count:,}  last_ts={last_ts_ms}")
                last_log = time.time()

            return int(ts_ms)

        first_event = next(merged_iter, None)
        if first_event is not None:
            ts_candidate = int(first_event[1])
            if last_global_ts is not None and ts_candidate < last_global_ts:
                raise ValueError("Non-monotonic timestamps across weeks")
            last_global_ts = process_event(first_event)

        for e in merged_iter:
            ts_candidate = int(e[1])
            if last_global_ts is not None and ts_candidate < last_global_ts:
                raise ValueError("Non-monotonic timestamps across weeks")
            last_global_ts = process_event(e)

        print(f"[week {w_idx}] done: events={event_count:,}, sequences={len(X_list):,} so far")

    # At the end there may be some pending sequences without matured labels
    # (e.g., decisions near the file tail). We drop those quietly.
    if len(X_list) == 0:
        feat_dim = fe.feature_dim() if fe._feat_dim is not None else 0
        return (
            np.empty((0, LOOKBACK, 0), dtype=np.float32),
            np.empty((0, 2 * NUM_HORIZONS), dtype=np.float32),
            feat_dim,
        )

    X = np.stack(X_list, axis=0).astype(np.float32)  # [N, L, F]
    Y = np.stack(y_list, axis=0).astype(np.float32)  # [N, 2 * NUM_HORIZONS] -> [returns || log-vol targets]
    feat_dim = fe.feature_dim()
    return X, Y, feat_dim


# --------------------  Utils: EMA-normalized losses + Huber  ---------------------
def huber_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    delta: Union[float, Iterable[float], torch.Tensor],
    weights: Optional[torch.Tensor] = None,
    reduction: str = "mean",
) -> torch.Tensor:
    """Huber loss with optional per-dimension weights and reduction control."""
    diff = pred - target
    abs_diff = diff.abs()
    delta_tensor = torch.as_tensor(delta, device=pred.device, dtype=pred.dtype)
    if delta_tensor.ndim == 0:
        delta_tensor = delta_tensor.view(1)
    delta_tensor = delta_tensor.view(1, -1)

    if diff.ndim == 0:
        delta_tensor = delta_tensor.view(())
    else:
        target_shape = (1,) * (diff.ndim - 1) + (-1,)
        delta_tensor = delta_tensor.view(*target_shape)
        if delta_tensor.shape[-1] not in (1, diff.shape[-1]):
            raise ValueError(
                "delta must broadcast to the last dimension of pred/target"
            )
        if delta_tensor.shape[-1] == 1 and diff.shape[-1] != 1:
            delta_tensor = delta_tensor.expand(*delta_tensor.shape[:-1], diff.shape[-1])

    quadratic = torch.minimum(abs_diff, delta_tensor)
    linear = abs_diff - quadratic
    loss = 0.5 * quadratic**2 / delta_tensor + linear

    if weights is not None:
        loss = loss * weights

    if reduction == "mean":
        return loss.mean()
    if reduction == "none":
        return loss
    if reduction == "dim_mean":
        return loss.mean(dim=0)
    raise ValueError(f"Unsupported reduction '{reduction}'")

def ema_update(name: str, value: float, ema_dict: Dict[str, float], decay: float = EMA_DECAY) -> float:
    old = ema_dict.get(name, 1.0)
    new = decay * old + (1.0 - decay) * value
    ema_dict[name] = new
    return new

def binary_auc_from_logits(logits: torch.Tensor, targets_pos: torch.Tensor) -> float:
    """
    Compute ROC AUC from logits without sklearn.
    logits: shape [N], raw logits
    targets_pos: shape [N], 0/1
    """
    s = logits.detach().cpu().numpy().astype(np.float64)
    y = targets_pos.detach().cpu().numpy().astype(np.int32)
    # rank-based AUC
    order = np.argsort(s)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(len(s), dtype=np.float64) + 1.0
    n_pos = y.sum()
    n_neg = len(y) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float('nan')
    auc = (ranks[y==1].sum() - n_pos*(n_pos+1)/2.0) / (n_pos*n_neg)
    return float(auc)

# --------------------  Training loop  ---------------------
def get_mask_ratio(epoch: int) -> float:
    return MASK_PRETRAIN if epoch < SSL_PRETRAIN_EPOCHS else MASK_FINETUNE

def train_and_evaluate():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 1) Pair and slice the last 28 weeks ending 2025-08-27
    week_files = _pair_by_week(DATA_ROOT)
    week_files = _slice_last_weeks_pairs(week_files, last_end_iso="2025-08-27", k=28)

    # Chronological split: 6 train, 2 val, 2 test (most recent at the end)
    train_weeks = week_files[:6]
    val_weeks   = week_files[6:8]
    test_weeks  = week_files[8:10]

    # 2) Stream each split separately (no leakage across splits)
    X_tr, y_tr, feat_dim = stream_bybit(train_weeks)
    X_va, y_va, _        = stream_bybit(val_weeks)
    X_te, y_te, _        = stream_bybit(test_weeks)

    F_core = max(0, feat_dim - AUX_DIM)

    if PCA is not None:
        n_tr, L, _ = X_tr.shape
        X_core_train = X_tr[:, :, :F_core].reshape(-1, F_core)  # [n_tr*L, F_core]

        pca = PCA(n_components=PCA_VAR, svd_solver="full")
        pca.fit(X_core_train)

        def _apply_pca(arr: np.ndarray) -> np.ndarray:
            if arr.size == 0:
                return arr
            n, L_, _F = arr.shape
            core = arr[:, :, :F_core].reshape(-1, F_core)
            core_p = pca.transform(core).astype(np.float32, copy=False).reshape(n, L_, -1)
            aux    = arr[:, :, F_core:]
            return np.concatenate([core_p, aux], axis=-1)

            X_tr = _apply_pca(X_tr)
        X_va = _apply_pca(X_va)
        X_te = _apply_pca(X_te)
        feat_dim = X_tr.shape[-1]
        print(f"[PCA] kept {feat_dim - AUX_DIM} PCs (+{AUX_DIM} aux) → feat_dim={feat_dim}")

    ds_train = HFTDataset(X_tr, y_tr)
    ds_val   = HFTDataset(X_va, y_va)
    ds_test  = HFTDataset(X_te, y_te)

    n_tr = len(y_tr)
    n_va = len(y_va)
    n_te = len(y_te)
    print(f"[sizes] train={n_tr}  val={n_va}  test={n_te}")

    y_train_ret = y_tr[:, :NUM_HORIZONS].astype(np.float32)

    def _compute_trim_bounds(arr: np.ndarray) -> Tuple[float, float]:
        if arr.size == 0:
            return float("inf"), float("-inf")
        try:
            lo = float(np.quantile(arr, DIR_MASK_TAIL_FRACTION, method="linear"))
            hi = float(np.quantile(arr, 1.0 - DIR_MASK_TAIL_FRACTION, method="linear"))
        except TypeError:
            lo = float(np.quantile(arr, DIR_MASK_TAIL_FRACTION, interpolation="linear"))
            hi = float(np.quantile(arr, 1.0 - DIR_MASK_TAIL_FRACTION, interpolation="linear"))
        return lo, hi

    pos_lo_list: List[float] = []
    pos_hi_list: List[float] = []
    neg_lo_list: List[float] = []
    neg_hi_list: List[float] = []

    print("[dir-mask quantiles]")
    for idx, horizon in enumerate(HORIZONS_MS):
        horizon_returns = y_train_ret[:, idx]
        pos_returns = horizon_returns[horizon_returns > 0]
        neg_returns = horizon_returns[horizon_returns < 0]

        pos_lo, pos_hi = _compute_trim_bounds(pos_returns)
        # For negatives we operate on magnitude
        neg_lo, neg_hi = _compute_trim_bounds((-neg_returns))

        pos_lo_list.append(pos_lo)
        pos_hi_list.append(pos_hi)
        neg_lo_list.append(neg_lo)
        neg_hi_list.append(neg_hi)

        print(
            f"  {horizon}ms → pos:[{pos_lo:.3e}, {pos_hi:.3e}]  neg|mag:[{neg_lo:.3e}, {neg_hi:.3e}]"
            f" (tail frac {DIR_MASK_TAIL_FRACTION:.2%})"
        )

    pos_lo_arr = np.array(pos_lo_list, dtype=np.float32)
    pos_hi_arr = np.array(pos_hi_list, dtype=np.float32)
    neg_lo_arr = np.array(neg_lo_list, dtype=np.float32)
    neg_hi_arr = np.array(neg_hi_list, dtype=np.float32)

    pos_lo_t = torch.from_numpy(pos_lo_arr)
    pos_hi_t = torch.from_numpy(pos_hi_arr)
    neg_lo_t = torch.from_numpy(neg_lo_arr)
    neg_hi_t = torch.from_numpy(neg_hi_arr)

    def build_dir_mask(y_ret: torch.Tensor) -> torch.Tensor:
        pos = y_ret > 0
        neg = y_ret < 0

        lo_pos = pos_lo_t.to(device=y_ret.device, dtype=y_ret.dtype).view(1, -1)
        hi_pos = pos_hi_t.to(device=y_ret.device, dtype=y_ret.dtype).view(1, -1)
        lo_neg = neg_lo_t.to(device=y_ret.device, dtype=y_ret.dtype).view(1, -1)
        hi_neg = neg_hi_t.to(device=y_ret.device, dtype=y_ret.dtype).view(1, -1)

        mag_neg = (-y_ret).clamp_min(0.0)
        keep_pos = pos & (y_ret >= lo_pos) & (y_ret <= hi_pos)
        keep_neg = neg & (mag_neg >= lo_neg) & (mag_neg <= hi_neg)
        return keep_pos | keep_neg

    def _format_pos_counts(arr: np.ndarray, total: int) -> str:
        parts = []
        for horizon, count in zip(HORIZONS_MS, arr.astype(int)):
            frac = count / max(total, 1)
            parts.append(f"{horizon}ms {int(count)} ({frac:.2%})")
        return "; ".join(parts)

    num_pos_train = (y_tr[:, :NUM_HORIZONS] > 0).sum(axis=0)
    num_pos_val = (y_va[:, :NUM_HORIZONS] > 0).sum(axis=0)
    num_pos_test = (y_te[:, :NUM_HORIZONS] > 0).sum(axis=0)

    print(f"Train positive returns: {_format_pos_counts(num_pos_train, n_tr)}")
    print(f"Val positive returns: {_format_pos_counts(num_pos_val, n_va)}")
    print(f"Test positive returns: {_format_pos_counts(num_pos_test, n_te)}")

    dl_train = DataLoader(ds_train, BATCH_SIZE, shuffle=True, drop_last=True, num_workers=8, pin_memory=True, prefetch_factor=8)
    dl_val   = DataLoader(ds_val,   BATCH_SIZE, shuffle=False, num_workers=4)
    dl_test  = DataLoader(ds_test,  BATCH_SIZE, shuffle=False, num_workers=4)

    assert X_tr.shape[-1] == feat_dim, "Feature dimension mismatch"
    args = ModelArgs(DMODEL, MAMBA_LAYERS, feat_dim, LOOKBACK)
    model = SAMBA(args).to(device)

    horizon_weights = torch.tensor(HORIZON_WEIGHTS, dtype=torch.float32, device=device)
    delta_ret_tensor = torch.as_tensor(DELTA_RET, dtype=torch.float32, device=device)
    delta_logvol_tensor = torch.as_tensor(DELTA_LOGVOL, dtype=torch.float32, device=device)
    horizon_weights_cpu = horizon_weights.detach().cpu().to(torch.float64)
    horizon_weights_np = horizon_weights_cpu.numpy()

    def compute_directional_loss(logits: torch.Tensor, y_ret: torch.Tensor) -> torch.Tensor:
        mask = build_dir_mask(y_ret)
        if not mask.any():
            return torch.tensor(0.0, device=logits.device)

        y_dir = (y_ret > 0).float()
        losses = []
        weights = []
        for h_idx in range(NUM_HORIZONS):
            mask_h = mask[:, h_idx]
            if mask_h.any():
                loss_h = F.binary_cross_entropy_with_logits(
                    logits[mask_h, h_idx], y_dir[mask_h, h_idx], reduction='mean'
                )
                losses.append(loss_h)
                weights.append(horizon_weights[h_idx])

        if not losses:
            return torch.tensor(0.0, device=logits.device)

        loss_stack = torch.stack(losses)
        weight_stack = torch.stack(weights)
        return (loss_stack * weight_stack).sum() / weight_stack.sum()

    def format_metric(values: Iterable[float], fmt: str) -> str:
        formatted = []
        for horizon, value in zip(HORIZONS_MS, values):
            val = float(value)
            if math.isnan(val) or math.isinf(val):
                formatted.append(f"{horizon}ms:nan")
            else:
                formatted.append(f"{horizon}ms:{fmt.format(val)}")
        return '[' + ', '.join(formatted) + ']'

    opt = SAM(model.parameters(), torch.optim.AdamW, lr=LR, weight_decay=1e-3, rho=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt.base_optimizer, mode='min', factor=0.5, patience=7)
    torch.cuda.empty_cache()

    best = float('inf')
    no_imp = 0

    # EMA meters for loss normalization
    ema_pre = {'recon': 1.0, 'cpc': 1.0}
    ema_ft  = {'ret': 1.0, 'logvol': 1.0, 'bce': 1.0, 'recon': 1.0, 'cpc': 1.0}

    for epoch in range(EPOCHS):
        # LR warmup
        warmup_factor = min(1.0, (epoch + 1) / WARMUP_EPOCHS) if epoch < WARMUP_EPOCHS else 1.0
        for param_group in opt.base_optimizer.param_groups:
            param_group['lr'] = LR * warmup_factor

        model.train()
        total_loss = 0.0
        mratio = get_mask_ratio(epoch)
        is_ssl_pretrain = (epoch < SSL_PRETRAIN_EPOCHS)

        pbar = tqdm(dl_train, desc=f"Ep{epoch+1}/{EPOCHS} ({'SSL-Pre' if is_ssl_pretrain else 'FT'}) mask={mratio:.2f}")

        # Epoch trackers
        ep_ret = ep_logvol = ep_bce = ep_recon = ep_cpc = 0.0
        n_batches = 0

        for x, y in pbar:
            x, y = x.to(device), y.to(device)

            # ========== SAM pass #1 ==========
            opt.base_optimizer.zero_grad()

            ret_pred, vol_pred, dir_pred_logits, h_clean, h_masked, mask_idx, cpc_loss = model(x, mask_ratio=mratio)

            # Recon loss (Mamba-space distillation): target = h_clean (stop-grad)
            B = x.size(0)
            batch_idx = torch.arange(B, device=x.device).unsqueeze(1).expand(-1, mask_idx.shape[1])
            recon = F.mse_loss(h_masked[batch_idx, mask_idx], h_clean.detach()[batch_idx, mask_idx])

            if is_ssl_pretrain:
                # Pretrain: recon + CPC only (EMA-normalized)
                ema_recon = ema_update('recon', recon.item(), ema_pre)
                ema_cpc   = ema_update('cpc',   cpc_loss.item(), ema_pre)
                loss = LAMBDA_RECON_PT * (recon / (ema_recon + 1e-8)) + LAMBDA_CPC_PT * (cpc_loss / (ema_cpc + 1e-8))
                ep_recon += recon.item(); ep_cpc += cpc_loss.item()
            else:
                # Fine-tune: supervised + tiny SSL auxiliaries (EMA-normalized)
                y_ret = y[:, :NUM_HORIZONS]
                y_logvol = y[:, NUM_HORIZONS:2 * NUM_HORIZONS]
                mse_ret = huber_loss(ret_pred, y_ret, delta_ret_tensor, weights=horizon_weights)
                mse_vol = huber_loss(vol_pred, y_logvol, delta_logvol_tensor, weights=horizon_weights)
                bce_loss = compute_directional_loss(dir_pred_logits, y_ret)

                ema_ret   = ema_update('ret',    mse_ret.item(),  ema_ft)
                ema_vol   = ema_update('logvol', mse_vol.item(),  ema_ft)
                ema_bce   = ema_update('bce',    bce_loss.item(), ema_ft)
                ema_recon = ema_update('recon',  recon.item(),    ema_ft)
                ema_cpc   = ema_update('cpc',    cpc_loss.item(), ema_ft)

                loss = (mse_ret / (ema_ret + 1e-8)) + (mse_vol / (ema_vol + 1e-8)) \
                     + LAMBDA_BCE      * (bce_loss / (ema_bce + 1e-8)) \
                     + LAMBDA_RECON_FT * (recon     / (ema_recon + 1e-8)) \
                     + LAMBDA_CPC_FT   * (cpc_loss  / (ema_cpc + 1e-8))

                ep_ret += mse_ret.item(); ep_logvol += mse_vol.item(); ep_bce += bce_loss.item()
                ep_recon += recon.item(); ep_cpc += cpc_loss.item()

            loss.backward()
            opt.first_step(zero_grad=True)

            # ========== SAM pass #2 ==========
            ret_pred2, vol_pred2, dir_pred_logits2, h_clean2, h_masked2, _, cpc_loss2 = model(x, mask_ratio=mratio, mask_idx=mask_idx)

            # Use original mask_idx from pass #1 to align recon targets
            recon2 = F.mse_loss(h_masked2[batch_idx, mask_idx], h_clean2.detach()[batch_idx, mask_idx])

            if is_ssl_pretrain:
                ema_recon = ema_pre['recon']
                ema_cpc   = ema_pre['cpc']
                loss2 = LAMBDA_RECON_PT * (recon2 / (ema_recon + 1e-8)) + LAMBDA_CPC_PT * (cpc_loss2 / (ema_cpc + 1e-8))
            else:
                y_ret = y[:, :NUM_HORIZONS]
                y_logvol = y[:, NUM_HORIZONS:2 * NUM_HORIZONS]
                mse_ret2 = huber_loss(ret_pred2, y_ret, delta_ret_tensor, weights=horizon_weights)
                mse_vol2 = huber_loss(vol_pred2, y_logvol, delta_logvol_tensor, weights=horizon_weights)
                bce_loss2 = compute_directional_loss(dir_pred_logits2, y_ret)

                ema_ret   = ema_ft['ret']
                ema_vol   = ema_ft['logvol']
                ema_bce   = ema_ft['bce']
                ema_recon = ema_ft['recon']
                ema_cpc   = ema_ft['cpc']

                loss2 = (mse_ret2 / (ema_ret + 1e-8)) + (mse_vol2 / (ema_vol + 1e-8)) \
                      + LAMBDA_BCE      * (bce_loss2 / (ema_bce + 1e-8)) \
                      + LAMBDA_RECON_FT * (recon2     / (ema_recon + 1e-8)) \
                      + LAMBDA_CPC_FT   * (cpc_loss2  / (ema_cpc + 1e-8))

            loss2.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_GRAD)
            opt.second_step(zero_grad=True)

            # Update teacher after optimizer step
            model.update_teacher()

            total_loss += loss.item()
            n_batches += 1
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        # Epoch summary
        if is_ssl_pretrain:
            print(f"Ep{epoch+1} (SSL-Pre) avg: recon={ep_recon/max(1,n_batches):.4e}, cpc={ep_cpc/max(1,n_batches):.4e}")
        else:
            print(f"Ep{epoch+1} (FT) avg: ret={ep_ret/max(1,n_batches):.4e}, logvol={ep_logvol/max(1,n_batches):.4e}, "
                  f"bce={ep_bce/max(1,n_batches):.4e}, recon={ep_recon/max(1,n_batches):.4e}, cpc={ep_cpc/max(1,n_batches):.4e}")

        # =====================  Validation  =====================
        model.eval()

        if is_ssl_pretrain:
            # During SSL pretraining, only track unsupervised losses
            val_recon_sum = 0.0
            val_cpc_sum = 0.0
            val_batches = 0

            with torch.no_grad():
                for x, _ in dl_val:
                    x = x.to(device)
                    _, _, _, h_clean, h_masked, mask_idx, cpc_loss = model(x, mask_ratio=mratio)

                    B = x.size(0)
                    batch_idx = torch.arange(B, device=x.device).unsqueeze(1).expand(-1, mask_idx.shape[1])
                    recon = F.mse_loss(h_masked[batch_idx, mask_idx], h_clean.detach()[batch_idx, mask_idx])
                    val_recon_sum += recon.item()
                    val_cpc_sum += cpc_loss.item()
                    val_batches += 1

            avg_recon = val_recon_sum / max(1, val_batches)
            avg_cpc = val_cpc_sum / max(1, val_batches)
            print(f"val_recon={avg_recon:.4e}, val_cpc={avg_cpc:.4e}")

        else:
            # Full supervised validation during fine-tuning
            val_logits_all, val_ypos_all = [], []
            val_logits_masked, val_ypos_masked = [], []
            val_ret_loss_sum = np.zeros(NUM_HORIZONS, dtype=np.float64)
            val_vol_loss_sum = np.zeros(NUM_HORIZONS, dtype=np.float64)
            val_sample_total = 0
            val_acc_sum = np.zeros(NUM_HORIZONS, dtype=np.float64)
            val_total = np.zeros(NUM_HORIZONS, dtype=np.float64)
            val_bce_unmasked_sum = np.zeros(NUM_HORIZONS, dtype=np.float64)
            val_bce_unmasked_count = np.zeros(NUM_HORIZONS, dtype=np.float64)
            val_bce_masked_sum = np.zeros(NUM_HORIZONS, dtype=np.float64)
            val_bce_masked_count = np.zeros(NUM_HORIZONS, dtype=np.float64)
            val_acc_masked_sum = np.zeros(NUM_HORIZONS, dtype=np.float64)
            val_masked_total = np.zeros(NUM_HORIZONS, dtype=np.float64)

            val_logits_all = [[] for _ in range(NUM_HORIZONS)]
            val_ypos_all = [[] for _ in range(NUM_HORIZONS)]
            val_logits_masked = [[] for _ in range(NUM_HORIZONS)]
            val_ypos_masked = [[] for _ in range(NUM_HORIZONS)]

            with torch.no_grad():
                for x, y_targets in dl_val:
                    x = x.to(device)
                    y_targets = y_targets.to(device)
                    y_return = y_targets[:, :NUM_HORIZONS]
                    y_logvol = y_targets[:, NUM_HORIZONS:2 * NUM_HORIZONS]

                    ret_pred, vol_pred, dir_pred_logits, *_ = model(x, mask_ratio=0.0)

                    ret_loss_elem = huber_loss(ret_pred, y_return, delta_ret_tensor, reduction='none')
                    vol_loss_elem = huber_loss(vol_pred, y_logvol, delta_logvol_tensor, reduction='none')
                    batch_n = x.size(0)

                    val_ret_loss_sum += ret_loss_elem.sum(dim=0).detach().cpu().numpy().astype(np.float64)
                    val_vol_loss_sum += vol_loss_elem.sum(dim=0).detach().cpu().numpy().astype(np.float64)
                    val_sample_total += batch_n

                    y_dir = (y_return > 0).to(torch.float32)
                    bce_elem = F.binary_cross_entropy_with_logits(dir_pred_logits, y_dir, reduction='none')
                    val_bce_unmasked_sum += bce_elem.sum(dim=0).detach().cpu().numpy().astype(np.float64)
                    val_bce_unmasked_count += batch_n

                    pred_class = (dir_pred_logits > 0).to(torch.int32)
                    true_class = y_dir.to(torch.int32)
                    val_acc_sum += (pred_class == true_class).sum(dim=0).detach().cpu().numpy().astype(np.float64)
                    val_total += batch_n

                    for h_idx in range(NUM_HORIZONS):
                        val_logits_all[h_idx].append(dir_pred_logits[:, h_idx].detach().cpu())
                        val_ypos_all[h_idx].append(true_class[:, h_idx].detach().cpu())

                    mask = build_dir_mask(y_return)
                    for h_idx in range(NUM_HORIZONS):
                        mask_h = mask[:, h_idx]
                        if mask_h.any():
                            logits_h = dir_pred_logits[mask_h, h_idx]
                            targets_h = y_dir[mask_h, h_idx]
                            val_bce_masked_sum[h_idx] += F.binary_cross_entropy_with_logits(
                                logits_h, targets_h, reduction='sum'
                            ).item()
                            val_bce_masked_count[h_idx] += mask_h.sum().item()
                            val_logits_masked[h_idx].append(logits_h.detach().cpu())
                            val_ypos_masked[h_idx].append(targets_h.to(torch.int32).detach().cpu())
                            val_acc_masked_sum[h_idx] += ((logits_h > 0).to(torch.int32) == targets_h.to(torch.int32)).sum().item()
                            val_masked_total[h_idx] += mask_h.sum().item()

            avg_val_ret_loss_per_h = val_ret_loss_sum / max(1, val_sample_total)
            avg_val_vol_loss_per_h = val_vol_loss_sum / max(1, val_sample_total)
            val_dir_bce_per_h = np.divide(
                val_bce_unmasked_sum,
                np.maximum(val_bce_unmasked_count, 1.0)
            )
            val_accuracy_per_h = np.divide(val_acc_sum, np.maximum(val_total, 1.0))

            val_auc_per_h = []
            for h_idx in range(NUM_HORIZONS):
                if val_logits_all[h_idx]:
                    logits_cat = torch.cat(val_logits_all[h_idx])
                    ypos_cat = torch.cat(val_ypos_all[h_idx])
                    val_auc_per_h.append(binary_auc_from_logits(logits_cat, ypos_cat))
                else:
                    val_auc_per_h.append(float('nan'))

            val_dir_bce_masked_per_h = np.full(NUM_HORIZONS, np.nan, dtype=np.float64)
            val_acc_masked_per_h = np.full(NUM_HORIZONS, np.nan, dtype=np.float64)
            val_auc_masked_per_h: List[float] = []
            for h_idx in range(NUM_HORIZONS):
                if val_bce_masked_count[h_idx] > 0:
                    val_dir_bce_masked_per_h[h_idx] = val_bce_masked_sum[h_idx] / val_bce_masked_count[h_idx]
                    val_acc_masked_per_h[h_idx] = val_acc_masked_sum[h_idx] / max(val_masked_total[h_idx], 1.0)
                    logits_cat = torch.cat(val_logits_masked[h_idx])
                    ypos_cat = torch.cat(val_ypos_masked[h_idx])
                    val_auc_masked_per_h.append(binary_auc_from_logits(logits_cat, ypos_cat))
                else:
                    val_auc_masked_per_h.append(float('nan'))

            avg_val_ret_loss = float(np.dot(avg_val_ret_loss_per_h, horizon_weights_np) / max(horizon_weights_cpu.sum().item(), 1e-12))
            avg_val_vol_loss = float(np.dot(avg_val_vol_loss_per_h, horizon_weights_np) / max(horizon_weights_cpu.sum().item(), 1e-12))

            print(
                f"val_ret_huber={format_metric(avg_val_ret_loss_per_h, '{:.4e}')}, "
                f"val_logvol_huber={format_metric(avg_val_vol_loss_per_h, '{:.4e}')}, "
                f"val_dir_bce={format_metric(val_dir_bce_per_h, '{:.4e}')}, "
                f"val_acc={format_metric(val_accuracy_per_h, '{:.4f}')}, "
                f"val_auc={format_metric(val_auc_per_h, '{:.4f}')}"
            )
            print(
                f"  masked_val_dir_bce={format_metric(val_dir_bce_masked_per_h, '{:.4e}')}, "
                f"masked_val_acc={format_metric(val_acc_masked_per_h, '{:.4f}')}, "
                f"masked_val_auc={format_metric(val_auc_masked_per_h, '{:.4f}')}"
            )

            scheduler.step(avg_val_ret_loss)
            if avg_val_ret_loss < best and not is_ssl_pretrain:
                best = avg_val_ret_loss
                print(f"New best validation loss (return Huber): {best:.4e}")
                no_imp = 0
                torch.save(model.state_dict(), 'best.pth')
            else:
                no_imp += 1
                print(f"no improve {no_imp}/{PATIENCE}")
                if no_imp >= PATIENCE:
                    print("Early stopping triggered.")
                    break


    # =====================  Test  =====================
    model.eval()
    test_ret_loss_sum = np.zeros(NUM_HORIZONS, dtype=np.float64)
    test_vol_loss_sum = np.zeros(NUM_HORIZONS, dtype=np.float64)
    test_sample_total = 0
    test_acc_sum = np.zeros(NUM_HORIZONS, dtype=np.float64)
    test_total = np.zeros(NUM_HORIZONS, dtype=np.float64)
    test_bce_sum = np.zeros(NUM_HORIZONS, dtype=np.float64)
    test_bce_count = np.zeros(NUM_HORIZONS, dtype=np.float64)
    test_bce_masked_sum = np.zeros(NUM_HORIZONS, dtype=np.float64)
    test_bce_masked_count = np.zeros(NUM_HORIZONS, dtype=np.float64)
    test_acc_masked_sum = np.zeros(NUM_HORIZONS, dtype=np.float64)
    test_masked_total = np.zeros(NUM_HORIZONS, dtype=np.float64)

    test_logits_all = [[] for _ in range(NUM_HORIZONS)]
    test_ypos_all = [[] for _ in range(NUM_HORIZONS)]
    test_logits_masked = [[] for _ in range(NUM_HORIZONS)]
    test_ypos_masked = [[] for _ in range(NUM_HORIZONS)]

    with torch.no_grad():
        for x, y in dl_test:
            x = x.to(device)
            y = y.to(device)
            y_return = y[:, :NUM_HORIZONS]
            y_logvol = y[:, NUM_HORIZONS:2 * NUM_HORIZONS]

            ret_pred, vol_pred, dir_pred_logits, *_ = model(x, mask_ratio=0.0)

            ret_loss_elem = huber_loss(ret_pred, y_return, delta_ret_tensor, reduction='none')
            vol_loss_elem = huber_loss(vol_pred, y_logvol, delta_logvol_tensor, reduction='none')
            batch_n = x.size(0)

            test_ret_loss_sum += ret_loss_elem.sum(dim=0).detach().cpu().numpy().astype(np.float64)
            test_vol_loss_sum += vol_loss_elem.sum(dim=0).detach().cpu().numpy().astype(np.float64)
            test_sample_total += batch_n

            y_dir = (y_return > 0).to(torch.float32)
            bce_elem = F.binary_cross_entropy_with_logits(dir_pred_logits, y_dir, reduction='none')
            test_bce_sum += bce_elem.sum(dim=0).detach().cpu().numpy().astype(np.float64)
            test_bce_count += batch_n

            pred_class = (dir_pred_logits > 0).to(torch.int32)
            true_class = y_dir.to(torch.int32)
            test_acc_sum += (pred_class == true_class).sum(dim=0).detach().cpu().numpy().astype(np.float64)
            test_total += batch_n

            for h_idx in range(NUM_HORIZONS):
                test_logits_all[h_idx].append(dir_pred_logits[:, h_idx].detach().cpu())
                test_ypos_all[h_idx].append(true_class[:, h_idx].detach().cpu())

            mask = build_dir_mask(y_return)
            for h_idx in range(NUM_HORIZONS):
                mask_h = mask[:, h_idx]
                if mask_h.any():
                    logits_h = dir_pred_logits[mask_h, h_idx]
                    targets_h = y_dir[mask_h, h_idx]
                    test_bce_masked_sum[h_idx] += F.binary_cross_entropy_with_logits(
                        logits_h, targets_h, reduction='sum'
                    ).item()
                    test_bce_masked_count[h_idx] += mask_h.sum().item()
                    test_logits_masked[h_idx].append(logits_h.detach().cpu())
                    test_ypos_masked[h_idx].append(targets_h.to(torch.int32).detach().cpu())
                    test_acc_masked_sum[h_idx] += ((logits_h > 0).to(torch.int32) == targets_h.to(torch.int32)).sum().item()
                    test_masked_total[h_idx] += mask_h.sum().item()

    avg_test_ret_loss_per_h = test_ret_loss_sum / max(1, test_sample_total)
    avg_test_vol_loss_per_h = test_vol_loss_sum / max(1, test_sample_total)
    test_dir_bce_per_h = np.divide(test_bce_sum, np.maximum(test_bce_count, 1.0))
    test_accuracy_per_h = np.divide(test_acc_sum, np.maximum(test_total, 1.0))

    test_auc_per_h = []
    for h_idx in range(NUM_HORIZONS):
        if test_logits_all[h_idx]:
            logits_cat = torch.cat(test_logits_all[h_idx])
            ypos_cat = torch.cat(test_ypos_all[h_idx])
            test_auc_per_h.append(binary_auc_from_logits(logits_cat, ypos_cat))
        else:
            test_auc_per_h.append(float('nan'))

    test_dir_bce_masked_per_h = np.full(NUM_HORIZONS, np.nan, dtype=np.float64)
    test_acc_masked_per_h = np.full(NUM_HORIZONS, np.nan, dtype=np.float64)
    test_auc_masked_per_h: List[float] = []
    for h_idx in range(NUM_HORIZONS):
        if test_bce_masked_count[h_idx] > 0:
            test_dir_bce_masked_per_h[h_idx] = test_bce_masked_sum[h_idx] / test_bce_masked_count[h_idx]
            test_acc_masked_per_h[h_idx] = test_acc_masked_sum[h_idx] / max(test_masked_total[h_idx], 1.0)
            logits_cat = torch.cat(test_logits_masked[h_idx])
            ypos_cat = torch.cat(test_ypos_masked[h_idx])
            test_auc_masked_per_h.append(binary_auc_from_logits(logits_cat, ypos_cat))
        else:
            test_auc_masked_per_h.append(float('nan'))

    print(
        f"Test_ret_huber={format_metric(avg_test_ret_loss_per_h, '{:.4e}')}, "
        f"Test_logvol_huber={format_metric(avg_test_vol_loss_per_h, '{:.4e}')}, "
        f"Test_dir_bce={format_metric(test_dir_bce_per_h, '{:.4e}')}, "
        f"Test_acc={format_metric(test_accuracy_per_h, '{:.4f}')}, "
        f"Test_auc={format_metric(test_auc_per_h, '{:.4f}')}"
    )
    print(
        f"  masked_Test_dir_bce={format_metric(test_dir_bce_masked_per_h, '{:.4e}')}, "
        f"masked_Test_acc={format_metric(test_acc_masked_per_h, '{:.4f}')}, "
        f"masked_Test_auc={format_metric(test_auc_masked_per_h, '{:.4f}')}"
    )


if __name__ == "__main__":
    train_and_evaluate()