import os, math, copy, json, csv, zipfile, io, glob
from collections import deque
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Union, List, Dict, Tuple, Generator, Optional
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

ft_config.donated_buffer = False
torch.cuda.empty_cache()

# ==============================  MAMBA2 (unchanged)  ==============================
class Mamba2(nn.Module, PyTorchModelHubMixin):
    def __init__(
        self,
        d_model,
        d_state=128,
        d_conv=4,
        conv_init=None,
        expand=2,
        headdim=32,  # Adjusted for smaller d_model
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
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

# ---------------------------  Core hyper-params  ---------------------------
LOOKBACK        = 512  # number of tokens spanning ~20s
BATCH_SIZE      = 64
DMODEL          = 64
MAMBA_LAYERS    = 3
CONV_KERNELS    = [9,17,25,33]
DFF_CONV        = 4 * DMODEL

# Masking / SSL schedule
SSL_PRETRAIN_EPOCHS = 10     # Pretrain epochs (recon + CPC only)
MASK_PRETRAIN       = 0.50   # Pretrain mask ratio
MASK_FINETUNE       = 0.20   # Fine-tune mask ratio

TAU             = 0.1
EPOCHS          = 200
LR              = 7e-4
CLIP_GRAD       = 10000
PATIENCE        = 15
BASE_FEATURES   = [
    'mid','microprice','smartprice','spread',
    'bid_size','ask_size',
    'cum_bid_L5','cum_ask_L5','cum_bid_L10','cum_ask_L10',
    'ofi_l1','ofi_l5',
    'buy_vol_1s','sell_vol_1s','buy_count_1s','sell_count_1s',
    'buy_mean_1s','sell_mean_1s','buy_max_1s','sell_max_1s',
    'quote_count_1s','trade_count_1s',
    'std_log_mid_100ms','std_log_mid_1s',
    'ema_microprice_25ms','ema_microprice_100ms','ema_microprice_500ms',
    'ema_sp_25ms','ema_sp_100ms','ema_sp_500ms',
    'rsi_microprice_100ms','vpin','daily_rv','ewma7d','ewma30d','var10s_over_ewma7d'
]
AUX_FEATURES    = ['dt','is_trade']
FEATURES        = BASE_FEATURES + AUX_FEATURES
NUM_HEADS       = 4
WARMUP_EPOCHS   = max(1, int(EPOCHS * 0.05))  # Warmup over first 5% of epochs

# Loss mixing (fixed lambdas), with EMA normalization per loss
EMA_DECAY       = 0.99
LAMBDA_BCE      = 0.20
LAMBDA_RECON_FT = 0.05
LAMBDA_CPC_FT   = 0.02
LAMBDA_RECON_PT = 1.00
LAMBDA_CPC_PT   = 0.50

# Huber deltas (tuned conservative defaults)
DELTA_RET       = 0.005
DELTA_LOGVOL    = 0.02

# CPC settings
CPC_DELTAS_TOK  = [25, 50, 100]  # token gaps roughly matching short real-time spans
#---------------------------------------------------------------------------

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
    def __init__(self, d_model, d_ff, kernel_size=[5, 5, 9, 9, 13, 13], dropout=0.1, activation='gelu', 
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
                 enable_res_param=True, norm='batch', re_param=True, re_param_kernel=3, patch_size=16, stride=8):
        super(ConvTimeNetFeatureExtractor, self).__init__()
        self.depatch = DepatchSampling(in_feats=in_feats, seq_len=seq_len, patch_size=patch_size, stride=stride)
        self.patch_count = (seq_len - patch_size) // stride + 1
        self.patch_size = patch_size
        self.d_model_internal = d_model // in_feats
        self.output_linear = nn.Linear(patch_size, self.d_model_internal)
        self.encoder = ConvEncoder(d_model=self.d_model_internal, d_ff=d_ff, kernel_size=dw_ks, dropout=dropout, activation=act, 
                                   n_layers=n_layers, enable_res_param=enable_res_param, norm=norm, re_param=re_param, small_ks=re_param_kernel)
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
        out = out.reshape(B, self.depatch.patch_count, self.d_model_internal * out.shape[3])  # [B, patch_count, d_model]
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
    def __init__(self, d_model, d_hidden=None):
        super().__init__()
        d_hidden = d_hidden or d_model
        self.W = nn.Linear(d_model, d_hidden)
        self.u = nn.Parameter(torch.randn(d_hidden))
    def forward(self, h):  # h: [B, L, D]
        g = torch.tanh(self.W(h))        # [B, L, H]
        scores = torch.matmul(g, self.u) # [B, L]
        alpha = torch.softmax(scores, dim=1)
        z = torch.einsum('bl,bld->bd', alpha, h)
        return z

class Mamba(nn.Module):
    """Causal (forward-only) Mamba stack + channel-FFN + gated pooling."""
    def __init__(self, args: ModelArgs, ff_hid: int):
        super().__init__()
        self.args = args
        self.emb = nn.Linear(args.vocab_size, args.d_model)
        _init_small(self.emb)
        self.blocks = nn.ModuleList([ResidualBlock(args) for _ in range(args.n_layer)])
        self.ffns = nn.ModuleList([ChannelFFN(args.d_model, ff_hid) for _ in range(args.n_layer)])
        self.norm = nn.LayerNorm(args.d_model)
        self.pool = GatedPooling(args.d_model)

    def forward(self, x, embedded=False):
        if not embedded:
            x = self.emb(x)  # project features to d_model
        for i in range(self.args.n_layer):
            x = self.blocks[i](x)    # causal Mamba
            x = self.ffns[i](x)      # channel mixing (no time LN)
        h = self.norm(x)
        pooled = self.pool(h)
        return pooled, h

# -------------  SAMBA -------------
class SAMBA(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        # (3) Switch BatchNorm -> LayerNorm in ConvTimeNet
        self.depatch_proj_encoder = ConvTimeNetFeatureExtractor(
            in_feats=args.vocab_size, seq_len=args.seq_in, d_model=args.d_model, 
            dw_ks=[5, 5, 9, 9, 13, 13], n_layers=6, d_ff=256, dropout=0.1, act='gelu', 
            enable_res_param=True, norm='layer', re_param=True, re_param_kernel=3, 
            patch_size=8, stride=4
        )
        # Mamba backbone (forward-only) + pooling
        self.mamba = Mamba(args, ff_hid=DMODEL)

        # SSL bits
        self.mask_token = nn.Parameter(torch.randn(1, 1, args.d_model))
        self.cpc_deltas = CPC_DELTAS_TOK
        self.cpc_predictors = nn.ModuleDict({
            str(d): nn.Sequential(
                nn.Linear(args.d_model, args.d_model),
                nn.GELU(),
                nn.Linear(args.d_model, args.d_model)
            ) for d in self.cpc_deltas
        })
        # EMA teacher for CPC targets (teacher = EMA(student))
        self.mamba_teacher = copy.deepcopy(self.mamba)
        for p in self.mamba_teacher.parameters():
            p.requires_grad = False
        self.teacher_momentum = 0.99

        # Heads
        head_hidden_dim = args.d_model * 2
        self.return_head = nn.Sequential(
            nn.Linear(args.d_model, head_hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(head_hidden_dim, 1)
        )
        self.volatility_head = nn.Sequential(
            nn.Linear(args.d_model, head_hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(head_hidden_dim, 1)  # predicts log-vol
        )
        self.direction_head = nn.Sequential(
            nn.Linear(args.d_model, head_hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(head_hidden_dim, 1)
        )

    @torch.no_grad()
    def update_teacher(self, m: float = None):
        """EMA update for teacher parameters."""
        if m is None:
            m = self.teacher_momentum
        for p_t, p_s in zip(self.mamba_teacher.parameters(), self.mamba.parameters()):
            p_t.data.mul_(m).add_(p_s.data, alpha=(1.0 - m))

    def compute_cpc_loss(self, h_student: torch.Tensor, h_teacher: torch.Tensor) -> torch.Tensor:
        """
        BYOL/BYOLa-style CPC without negatives:
        For each Δ in self.cpc_deltas, predict teacher h_{t+Δ} from student context h_t.
        Loss = 1 - cosine(q_tΔ, z_{t+Δ}).
        """
        total = 0.0
        count = 0
        for d in self.cpc_deltas:
            if h_student.size(1) <= d:
                continue
            ctx = h_student[:, :-d, :]     # [B, L-d, D]
            tgt = h_teacher[:, d:, :].detach()  # stop-grad
            q = self.cpc_predictors[str(d)](ctx)
            cos = F.cosine_similarity(q, tgt, dim=-1)  # [B, L-d]
            total = total + (1.0 - cos).mean()
            count += 1
        if count == 0:
            return h_student.new_tensor(0.0)
        return total / count

    def forward(self, x, mask_ratio=0.0):
        """
        Training path returns: pooled, ret_pred, vol_pred, dir_logits, 
        h_clean (student), h_masked (student on masked input), mask_idx, cpc_loss.
        Eval path returns predictions only.
        """
        x_permuted = x.permute(0, 2, 1)
        h_tokens = self.depatch_proj_encoder(x_permuted)                   # [B, L, D]

        # Student (clean)
        pooled, h_clean = self.mamba(h_tokens, embedded=True)
        ret_pred = self.return_head(pooled).squeeze(-1)
        vol_pred = self.volatility_head(pooled).squeeze(-1)
        dir_pred_logits = self.direction_head(pooled).squeeze(-1)

        # Teacher (clean, no grad) for CPC
        with torch.no_grad():
            _, h_teacher_clean = self.mamba_teacher(h_tokens, embedded=True)

        # Masked pass (student) for reconstruction distillation in Mamba space
        B, L, D = h_tokens.shape
        mcnt = max(1, int(mask_ratio * L))
        mask_idx = torch.stack([torch.randperm(L, device=x.device)[:mcnt] for _ in range(B)])  # [B, mcnt]
        h_masked_input = h_tokens.clone()
        batch_idx = torch.arange(B, device=x.device).unsqueeze(1).expand(-1, mcnt)
        h_masked_input[batch_idx, mask_idx] = self.mask_token  # replace masked tokens

        _, h_masked = self.mamba(h_masked_input, embedded=True)

        # CPC loss (computed here so both SAM passes align)
        cpc_loss = self.compute_cpc_loss(h_clean, h_teacher_clean)

        return ret_pred, vol_pred, dir_pred_logits, h_clean, h_masked, mask_idx, cpc_loss

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

# --------------------  Data ingestion & merging  ---------------------

class BybitRawIter:
    """Iterate over Bybit L2 order book (.data) and trade history (.csv) files."""

    def __init__(self, ob_zip: str, th_zip: str):
        self.ob_zip = ob_zip
        self.th_zip = th_zip

    def ob_iter(self) -> Generator[Tuple[int, int, dict], None, None]:
        with zipfile.ZipFile(self.ob_zip) as z:
            name = z.namelist()[0]
            with z.open(name) as f:
                for line in f:
                    if not line:
                        continue
                    obj = json.loads(line)
                    ts = int(obj.get("ts", obj.get("cts", 0)))
                    seq = obj["data"].get("seq", 0)
                    yield ts, seq, obj

    def trade_iter(self) -> Generator[Tuple[int, int, dict], None, None]:
        with zipfile.ZipFile(self.th_zip) as z:
            name = z.namelist()[0]
            with z.open(name) as f:
                reader = csv.DictReader(io.TextIOWrapper(f))
                seq = 0
                for row in reader:
                    seq += 1
                    ts = int(float(row["timestamp"]) * 1000)
                    row["seq"] = seq
                    yield ts, seq, row


def merge_event_time(ob_iter: Generator[Tuple[int, int, dict], None, None],
                     tr_iter: Generator[Tuple[int, int, dict], None, None]) -> Generator[Tuple[str, int, int, dict], None, None]:
    """Merge OB and trade iterators by timestamp and sequence."""
    ob_item = next(ob_iter, None)
    tr_item = next(tr_iter, None)
    last_ts = -1
    while ob_item or tr_item:
        if tr_item is None or (ob_item and ob_item[0] <= tr_item[0]):
            ts, seq, data = ob_item
            ob_item = next(ob_iter, None)
            etype = "ob"
        else:
            ts, seq, data = tr_item
            tr_item = next(tr_iter, None)
            etype = "trade"
        if ts < last_ts:
            raise ValueError("Non-monotonic timestamps in event stream")
        last_ts = ts
        yield etype, ts, seq, data


# ---------------------  Rolling normalization  ---------------------

class RollingZScore:
    def __init__(self, window_ms: int = 60000):
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
    """Maintain in-memory state and compute features on each event."""

    def __init__(self, depth: int = 10):
        self.depth = depth
        self.bids: Dict[float, float] = {}
        self.asks: Dict[float, float] = {}
        self.trades = deque()
        self.quotes = deque()
        self.mid_history = deque()
        self.prev_bid_size = self.prev_ask_size = 0.0
        self.last_ts: Optional[int] = None
        self.ema_mp_25 = self.ema_mp_100 = self.ema_mp_500 = None
        self.ema_sp_25 = self.ema_sp_100 = self.ema_sp_500 = None
        self.rsi_gain = self.rsi_loss = None
        self.vpin_window = deque()
        self.daily_rv = self.ewma7d = self.ewma30d = 0.0

    def _prune(self, deq: deque, t: int, window: int):
        while deq and t - deq[0][0] > window:
            deq.popleft()

    def _book_best(self):
        bid = max(self.bids.keys()) if self.bids else 0.0
        ask = min(self.asks.keys()) if self.asks else 0.0
        bsz = self.bids.get(bid, 0.0)
        asz = self.asks.get(ask, 0.0)
        return bid, ask, bsz, asz

    def _update_book(self, data: dict):
        tp = data.get("type")
        bids = data["data"].get("b", [])
        asks = data["data"].get("a", [])
        if tp == "snapshot":
            self.bids = {float(p): float(q) for p, q in bids[:self.depth]}
            self.asks = {float(p): float(q) for p, q in asks[:self.depth]}
        else:
            for p, q in bids:
                p = float(p); q = float(q)
                if q == 0:
                    self.bids.pop(p, None)
                else:
                    self.bids[p] = q
            for p, q in asks:
                p = float(p); q = float(q)
                if q == 0:
                    self.asks.pop(p, None)
                else:
                    self.asks[p] = q

    def on_event(self, etype: str, data: dict, ts: int) -> float:
        if etype == "ob":
            self.quotes.append((ts, 1))
            self._update_book(data)
        else:
            side = data["side"].lower()
            qty = float(data["size"])
            price = float(data["price"])
            self.trades.append((ts, side, qty, price))

        bid, ask, bsz, asz = self._book_best()
        mid = (bid + ask) / 2 if bid and ask else 0.0
        mp = (ask * bsz + bid * asz) / (bsz + asz + 1e-8)
        sp = (ask / (asz + 1e-8) + bid / (bsz + 1e-8)) / (1 / (asz + 1e-8) + 1 / (bsz + 1e-8))
        spr = ask - bid

        dt = 0 if self.last_ts is None else max(1, ts - self.last_ts)
        for attr, val, halflife in [
            ("ema_mp_25", mp, 25),
            ("ema_mp_100", mp, 100),
            ("ema_mp_500", mp, 500),
            ("ema_sp_25", spr, 25),
            ("ema_sp_100", spr, 100),
            ("ema_sp_500", spr, 500),
        ]:
            cur = getattr(self, attr)
            alpha = 1 - math.exp(-dt / halflife)
            setattr(self, attr, (1 - alpha) * (cur if cur is not None else val) + alpha * val)

        delta_mp = mp - (self.mid_history[-1][1] if self.mid_history else mp)
        gain = max(delta_mp, 0.0)
        loss = max(-delta_mp, 0.0)
        alpha_rsi = 1 - math.exp(-dt / 100)
        self.rsi_gain = (1 - alpha_rsi) * (self.rsi_gain or 0.0) + alpha_rsi * gain
        self.rsi_loss = (1 - alpha_rsi) * (self.rsi_loss or 0.0) + alpha_rsi * loss

        self.mid_history.append((ts, mid))
        self.last_ts = ts
        return mid

    def snapshot(self, ts: int) -> np.ndarray:
        lookback_ms = 20000
        self._prune(self.trades, ts, lookback_ms)
        self._prune(self.quotes, ts, lookback_ms)
        self._prune(self.mid_history, ts, 20000)

        bid, ask, bsz, asz = self._book_best()
        mid = (bid + ask) / 2 if bid and ask else 0.0
        mp = (ask * bsz + bid * asz) / (bsz + asz + 1e-8)
        sp = (ask / (asz + 1e-8) + bid / (bsz + 1e-8)) / (1 / (asz + 1e-8) + 1 / (bsz + 1e-8))
        spr = ask - bid

        def cum_depth(book: Dict[float, float], reverse: bool, n: int) -> float:
            prices = sorted(book.keys(), reverse=reverse)[:n]
            return sum(book[p] for p in prices)

        cum_bid5 = cum_depth(self.bids, True, 5)
        cum_ask5 = cum_depth(self.asks, False, 5)
        cum_bid10 = cum_depth(self.bids, True, 10)
        cum_ask10 = cum_depth(self.asks, False, 10)

        ofi_l1 = (bsz - self.prev_bid_size) - (asz - self.prev_ask_size)
        ofi_l5 = (cum_bid5 - self.prev_bid_size) - (cum_ask5 - self.prev_ask_size)
        self.prev_bid_size, self.prev_ask_size = bsz, asz

        self._prune(self.trades, ts, 1000)
        buys = [q for t, s, q, p in self.trades if s == 'buy']
        sells = [q for t, s, q, p in self.trades if s == 'sell']
        buy_vol = sum(buys)
        sell_vol = sum(sells)
        buy_count = len(buys)
        sell_count = len(sells)
        buy_mean = buy_vol / buy_count if buy_count else 0.0
        sell_mean = sell_vol / sell_count if sell_count else 0.0
        buy_max = max(buys) if buys else 0.0
        sell_max = max(sells) if sells else 0.0

        self._prune(self.quotes, ts, 1000)
        quote_count = len(self.quotes)
        trade_count = len(self.trades)

        self._prune(self.mid_history, ts, 1000)
        mids_1s = [m for t, m in self.mid_history]
        logrets = np.diff(np.log(mids_1s)) if len(mids_1s) > 1 else np.array([0.0])
        std_1s = float(np.std(logrets))
        self._prune(self.mid_history, ts, 100)
        mids_100 = [m for t, m in self.mid_history]
        logrets100 = np.diff(np.log(mids_100)) if len(mids_100) > 1 else np.array([0.0])
        std_100 = float(np.std(logrets100))

        rs = self.rsi_gain / (self.rsi_loss + 1e-8)
        rsi = 100 - 100 / (1 + rs)

        self.vpin_window.append((ts, buy_vol - sell_vol))
        self._prune(self.vpin_window, ts, 1000)
        vpin = (sum(abs(v) for _, v in self.vpin_window) / (sum(abs(v) for _, v in self.vpin_window) + 1e-8))

        ret = 0.0 if len(self.mid_history) < 2 else math.log(mid / self.mid_history[-2][1])
        self.daily_rv = 0.9995 * self.daily_rv + 0.0005 * ret * ret
        self.ewma7d = 0.9999 * self.ewma7d + 0.0001 * ret * ret
        self.ewma30d = 0.99997 * self.ewma30d + 0.00003 * ret * ret
        var10s = ret * ret
        var_ratio = var10s / (self.ewma7d + 1e-8)

        feat = np.array([
            mid, mp, sp, spr,
            bsz, asz,
            cum_bid5, cum_ask5, cum_bid10, cum_ask10,
            ofi_l1, ofi_l5,
            buy_vol, sell_vol, buy_count, sell_count,
            buy_mean, sell_mean, buy_max, sell_max,
            quote_count, trade_count,
            std_100, std_1s,
            self.ema_mp_25 or mp, self.ema_mp_100 or mp, self.ema_mp_500 or mp,
            self.ema_sp_25 or spr, self.ema_sp_100 or spr, self.ema_sp_500 or spr,
            rsi, vpin, self.daily_rv, self.ewma7d, self.ewma30d, var_ratio
        ], dtype=np.float32)
        return feat


class LabelBuilder:
    def __init__(self, delta_ms: int = 5, horizon_ms: int = 1000):
        self.delta = delta_ms
        self.horizon = horizon_ms
        self.wait_delta = deque()
        self.wait_horizon = deque()

    def on_decision(self, t: int):
        self.wait_delta.append({'t': t, 't_delta': t + self.delta})

    def on_event(self, t: int, mid: float) -> List[Tuple[int, float]]:
        matured = []
        while self.wait_delta and t >= self.wait_delta[0]['t_delta']:
            item = self.wait_delta.popleft()
            item['mid0'] = mid
            item['t_mature'] = item['t_delta'] + self.horizon
            self.wait_horizon.append(item)
        while self.wait_horizon and t >= self.wait_horizon[0]['t_mature']:
            item = self.wait_horizon.popleft()
            ret = math.log(mid / item['mid0'])
            matured.append((item['t'], ret))
        return matured


class HFTDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = X.astype(np.float32)
        self.y = y.astype(np.float32)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        return torch.from_numpy(self.X[idx]), torch.from_numpy(self.y[idx])


def stream_bybit(week_files: List[Tuple[str, str]]) -> Tuple[np.ndarray, np.ndarray]:
    fe = FeatureEngine()
    normalizer = RollingZScore()
    labeler = LabelBuilder()
    tokens = deque(maxlen=LOOKBACK)
    pending = deque()
    X_list, y_list = [], []
    last_ts = None

    for ob_zip, th_zip in week_files:
        raw = BybitRawIter(ob_zip, th_zip)
        merged = merge_event_time(raw.ob_iter(), raw.trade_iter())
        for etype, ts, seq, data in merged:
            mid = fe.on_event(etype, data, ts)
            feat = fe.snapshot(ts)
            norm_feat = normalizer.update(feat, ts)
            dt = 0.0 if last_ts is None else (ts - last_ts) / 1000.0
            last_ts = ts
            aux = np.array([dt, 1.0 if etype == 'trade' else 0.0], dtype=np.float32)
            token = np.concatenate([norm_feat, aux])
            tokens.append(token)
            matured = labeler.on_event(ts, mid)
            while pending and matured and pending[0][0] <= matured[0][0]:
                t0, seq_arr = pending.popleft()
                for mt in matured:
                    if mt[0] == t0:
                        ret = mt[1]
                        X_list.append(seq_arr)
                        y_list.append([ret, math.log(abs(ret) + 1e-8)])
                        matured.remove(mt)
                        break
            if len(tokens) == LOOKBACK:
                seq_arr = np.stack(tokens, axis=0)
                labeler.on_decision(ts)
                pending.append((ts, seq_arr))
    return np.array(X_list), np.array(y_list)


# --------------------  Utils: EMA-normalized losses + Huber  ---------------------
def huber_loss(pred: torch.Tensor, target: torch.Tensor, delta: float) -> torch.Tensor:
    """Mean Huber (Smooth L1 with custom delta)."""
    diff = pred - target
    abs_diff = diff.abs()
    quadratic = torch.minimum(abs_diff, torch.tensor(delta, device=pred.device))
    linear = abs_diff - quadratic
    return (0.5 * quadratic**2 / delta + linear).mean()

def ema_update(name: str, value: float, ema_dict: Dict[str, float], decay: float = EMA_DECAY) -> float:
    old = ema_dict.get(name, 1.0)
    new = decay * old + (1.0 - decay) * value
    ema_dict[name] = new
    return new

# --------------------  Training loop  ---------------------
def get_mask_ratio(epoch: int) -> float:
    return MASK_PRETRAIN if epoch < SSL_PRETRAIN_EPOCHS else MASK_FINETUNE

def train_and_evaluate():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    ob_files = sorted(glob.glob(os.path.expanduser("~/BTCUSDT_OB_*.zip")))
    th_files = sorted(glob.glob(os.path.expanduser("~/BTCUSDT_TH_*.zip")))
    week_files = list(zip(ob_files, th_files))
    X, y = stream_bybit(week_files)
    total = len(X)
    tr, val = int(0.6 * total), int(0.8 * total)

    ds_train = HFTDataset(X[:tr], y[:tr])
    ds_val = HFTDataset(X[tr:val], y[tr:val])
    ds_test = HFTDataset(X[val:], y[val:])

    num_pos_train = int((y[:tr,0] > 0).sum())
    print(f"Train positive returns: {num_pos_train} / {tr} ({num_pos_train / max(tr,1):.2%})")
    num_pos_val = int((y[tr:val,0] > 0).sum())
    print(f"Val positive returns: {num_pos_val} / {val - tr} ({num_pos_val / max(val - tr,1):.2%})")
    num_pos_test = int((y[val:,0] > 0).sum())
    print(f"Test positive returns: {num_pos_test} / {total - val} ({num_pos_test / max(total - val,1):.2%})")

    dl_train = DataLoader(ds_train, BATCH_SIZE, shuffle=False, num_workers=8, pin_memory=True, prefetch_factor=4)
    dl_val   = DataLoader(ds_val,   BATCH_SIZE, shuffle=False, num_workers=4)
    dl_test  = DataLoader(ds_test,  BATCH_SIZE, shuffle=False, num_workers=4)

    args = ModelArgs(DMODEL, MAMBA_LAYERS, len(FEATURES), LOOKBACK)
    model = SAMBA(args).to(device)

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
                y_ret, y_logvol = y[:, 0], y[:, 1]
                mse_ret = huber_loss(ret_pred, y_ret, DELTA_RET)
                mse_vol = huber_loss(vol_pred, y_logvol, DELTA_LOGVOL)
                bce_loss = F.binary_cross_entropy_with_logits(dir_pred_logits, (y_ret > 0).float())

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
            ret_pred2, vol_pred2, dir_pred_logits2, h_clean2, h_masked2, _, cpc_loss2 = model(x, mask_ratio=mratio)

            # Use original mask_idx from pass #1 to align recon targets
            recon2 = F.mse_loss(h_masked2[batch_idx, mask_idx], h_clean2.detach()[batch_idx, mask_idx])

            if is_ssl_pretrain:
                ema_recon = ema_pre['recon']
                ema_cpc   = ema_pre['cpc']
                loss2 = LAMBDA_RECON_PT * (recon2 / (ema_recon + 1e-8)) + LAMBDA_CPC_PT * (cpc_loss2 / (ema_cpc + 1e-8))
            else:
                y_ret, y_logvol = y[:, 0], y[:, 1]
                mse_ret2 = huber_loss(ret_pred2, y_ret, DELTA_RET)
                mse_vol2 = huber_loss(vol_pred2, y_logvol, DELTA_LOGVOL)
                bce_loss2 = F.binary_cross_entropy_with_logits(dir_pred_logits2, (y_ret > 0).float())

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
        val_ret_loss_sum = 0.0
        val_vol_loss_sum = 0.0
        val_acc_sum = 0
        val_total = 0
        with torch.no_grad():
            for x, y_targets in dl_val:
                y_return, y_logvol = y_targets[:, 0].to(device), y_targets[:, 1].to(device)
                ret_pred, vol_pred, dir_pred_logits, *_ = model(x.to(device), mask_ratio=0.0)
                # Validate in same training spaces using Huber
                batch_ret_loss = huber_loss(ret_pred, y_return, DELTA_RET).item()
                batch_vol_loss = huber_loss(vol_pred, y_logvol, DELTA_LOGVOL).item()
                val_ret_loss_sum += batch_ret_loss
                val_vol_loss_sum += batch_vol_loss
                predicted_class = (dir_pred_logits > 0).float()
                true_class = (y_return > 0).float()
                val_acc_sum += (predicted_class == true_class).sum().item()
                val_total += x.size(0)

        avg_val_ret_loss = val_ret_loss_sum / len(dl_val)
        avg_val_vol_loss = val_vol_loss_sum / len(dl_val)
        val_accuracy = val_acc_sum / val_total
        print(f"val_ret_huber={avg_val_ret_loss:.4e}, val_logvol_huber={avg_val_vol_loss:.4e}, val_acc={val_accuracy:.4f}")

        # Use the return loss for scheduling and early stopping.
        scheduler.step(avg_val_ret_loss)

        if avg_val_ret_loss < best and not is_ssl_pretrain:
            best = avg_val_ret_loss
            print(f"New best validation loss (return Huber): {best:.4e}")
            no_imp = 0
            torch.save(model.state_dict(), "best.pth")
        elif not is_ssl_pretrain:
            no_imp += 1
            print(f"no improve {no_imp}/{PATIENCE}")
            if no_imp >= PATIENCE:
                print("Early stopping triggered.")
                break

    # =====================  Test  =====================
    model.eval()
    tot, acc = 0, 0
    with torch.no_grad():
        for x, y in dl_test:
            ret_pred, vol_pred, dir_pred_logits, *_ = model(x.to(device), mask_ratio=0.0)
            pred = (dir_pred_logits > 0).float()
            true = (y[:, 0].to(device) > 0).float()
            tot += y.size(0)
            acc += (pred.cpu() == true.cpu()).sum().item()
    print(f"Test acc {acc/tot:.4f}")

if __name__ == "__main__":
    train_and_evaluate()
