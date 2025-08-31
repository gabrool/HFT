import os, math, copy, torch, polars as pl, numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Union, List, Dict
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
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
LOOKBACK        = 120
BATCH_SIZE      = 256
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
FEATURES        = ['Open','High','Low','Close','Volume','VWAP','PCT', 'PastVol_5M']
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
CPC_DELTAS_TOK  = [1, 2, 3]  # horizon in tokens; on L2, scale these
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
class AAVEDataset(Dataset):
    def __init__(self, df, features, lookback: int):
        self.x = df[features].to_numpy()
        self.y_return = df['Return'].to_numpy()
        self.y_vol = df['TargetVol'].to_numpy()
        self.L = lookback
    def __len__(self):
        return len(self.x) - self.L + 1
    def __getitem__(self, idx):
        return (torch.tensor(self.x[idx:idx + self.L], dtype=torch.float32),
                torch.tensor([self.y_return[idx + self.L - 1], self.y_vol[idx + self.L - 1]], dtype=torch.float32))

def rebuild_csv():
    fmt = "%Y-%m-%d %H:%M:%S"
    df = pl.read_csv("AAVE_minute_data.csv").rename({"open": "Open", "high": "High", "low": "Low",
                                                     "close": "Close", "volume": "Volume", "vwap": "VWAP"})
    df = df.with_columns(pl.col("datetime").str.strptime(pl.Datetime, fmt))
    df = df.with_columns(pl.col("Close").pct_change().alias("PCT"))
    past_var = (pl.col("PCT").shift(1)**2 + pl.col("PCT").shift(2)**2 + pl.col("PCT").shift(3)**2 + pl.col("PCT").shift(4)**2 + pl.col("PCT").shift(5)**2) / 5
    df = df.with_columns(past_var.sqrt().alias("PastVol_5M"))
    df = df.with_columns((pl.col("Volume") / (pl.col("VWAP") + 1e-8)).map_elements(lambda v: math.log(1 + v), return_dtype=pl.Float32).alias("Volume"))
    df = df.with_columns(pl.col("VWAP").map_elements(lambda v: math.log(1 + v), return_dtype=pl.Float32).alias("VWAP"))
    df = df.with_row_index("idx")
    df = df.with_columns((pl.col("Close").shift(-5) / pl.col("Close") - 1).alias("Return"))
    future_var = (pl.col("PCT").shift(-1)**2 + pl.col("PCT").shift(-2)**2 + pl.col("PCT").shift(-3)**2 + pl.col("PCT").shift(-4)**2 + pl.col("PCT").shift(-5)**2) / 5
    df = df.with_columns(future_var.sqrt().alias("FutureVol"))
    df = df.with_columns(pl.col("FutureVol").map_elements(lambda v: math.log(v + 1e-8), return_dtype=pl.Float32).alias("TargetVol"))
    return df.filter(pl.col("PCT").is_not_null() & pl.col("Return").is_not_null() & pl.col("TargetVol").is_not_null() & pl.col("PastVol_5M").is_not_null())

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
    df = rebuild_csv()
    total = len(df)
    tr, val = int(0.7 * total), int(0.85 * total)
    scaler = StandardScaler().fit(df[:tr][FEATURES].to_numpy())
    df_train = df[:tr].with_columns(pl.DataFrame(scaler.transform(df[:tr][FEATURES].to_numpy()), schema=FEATURES))
    df_val = df[tr:val].with_columns(pl.DataFrame(scaler.transform(df[tr:val][FEATURES].to_numpy()), schema=FEATURES))
    df_test = df[val:].with_columns(pl.DataFrame(scaler.transform(df[val:][FEATURES].to_numpy()), schema=FEATURES))

    num_pos_train = (df_train['Return'] > 0).sum()
    print(f"Train positive returns: {num_pos_train} / {len(df_train)} ({num_pos_train / len(df_train):.2%})")
    num_pos_val = (df_val['Return'] > 0).sum()
    print(f"Val positive returns: {num_pos_val} / {len(df_val)} ({num_pos_val / len(df_val):.2%})")
    num_pos_test = (df_test['Return'] > 0).sum()
    print(f"Test positive returns: {num_pos_test} / {len(df_test)} ({num_pos_test / len(df_test):.2%})")

    ds_train = AAVEDataset(df_train, FEATURES, LOOKBACK)
    ds_val = AAVEDataset(df_val, FEATURES, LOOKBACK)
    ds_test = AAVEDataset(df_test, FEATURES, LOOKBACK)

    dl_train = DataLoader(ds_train, BATCH_SIZE, shuffle=True, num_workers=16, pin_memory=True, persistent_workers=True, prefetch_factor=4)
    dl_val   = DataLoader(ds_val,   BATCH_SIZE, shuffle=False, num_workers=8)
    dl_test  = DataLoader(ds_test,  BATCH_SIZE, shuffle=False, num_workers=8)

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
