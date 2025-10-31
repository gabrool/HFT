
#!/usr/bin/env python3
"""
CMSSL17_offline.py

Run CMSSL17's model *using prebuilt tokens* produced by offline_ingest.py.
This mirrors the training/eval flow in CMSSL17.py but reads dataset splits
from OUT_ROOT/meta.json and week meta files, avoiding any online feature building.

Env vars:
  BYBIT_OUT_ROOT=/path/to/offline_ingest_output_root      (REQUIRED)
  BYBIT_USE_IN_MEMORY=0|1   # 1 = load all chunks into RAM (faster but memory heavy). Default 0
  BYBIT_PCA_VAR=0|float     # 0 disables PCA; else keep variance fraction (e.g. 0.99). Default 0
  BYBIT_PCA_INCREMENTAL=0|1 # 1 = fit IncrementalPCA on training chunks (memory-safe). Default 1
  BYBIT_WORKERS=8           # dataloader workers. Default 8 for train, 4 for val/test.

Splits:
  Uses the same week ordering produced by offline_ingest (strictly increasing by end date).
  If there are >=24 weeks, it uses exactly the last 24 with a (18/3/3) chronological split.
  Otherwise it falls back to 75%/12.5%/12.5% rounded.

Files layout expected (created by offline_ingest.py):
  OUT_ROOT/
    meta.json                           # global manifest
    <WEEK_KEY>/
      meta_week.json
      Xcore_000.npy, Xaux_000.npy, y_000.npy
      Xcore_001.npy, Xaux_001.npy, y_001.npy
      ...

This script attempts to *import* model and utils from CMSSL17.py to avoid duplication.
"""

import os, sys, json, math, gc, glob, re
from dataclasses import dataclass
from typing import List, Dict, Tuple, Iterable, Optional
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# ---------------- Import from CMSSL17 ----------------
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from CMSSL17 import (  # type: ignore
    # model + args
    SAMBA, ModelArgs,
    # core hypers
    LOOKBACK, AUX_DIM, HORIZONS_MS, NUM_HORIZONS, HORIZON_WEIGHTS,
    BATCH_SIZE, EPOCHS, WARMUP_EPOCHS, LR, PATIENCE,
    # schedules / deltas / lambdas
    SSL_PRETRAIN_EPOCHS, MASK_PRETRAIN, MASK_FINETUNE, DIR_MASK_TAIL_FRACTION,
    DELTA_RET, DELTA_LOGVOL,
    EMA_DECAY, LAMBDA_BCE, LAMBDA_RECON_FT, LAMBDA_CPC_FT, LAMBDA_RECON_PT, LAMBDA_CPC_PT,
    DMODEL, MAMBA_LAYERS,
    # utils
    huber_loss, ema_update, binary_auc_from_logits,
    # optimizer
    SAM,
)

# ---------------- Config via env ----------------
OUT_ROOT = os.environ.get("BYBIT_OUT_ROOT", "").strip()
USE_IN_MEMORY = int(os.environ.get("BYBIT_USE_IN_MEMORY", "0")) == 1
PCA_VAR_ENV = float(os.environ.get("BYBIT_PCA_VAR", "0"))
PCA_INCREMENTAL = int(os.environ.get("BYBIT_PCA_INCREMENTAL", "1")) == 1
WORKERS_TRAIN = int(os.environ.get("BYBIT_WORKERS", "8"))
WORKERS_VAL   = max(1, min(4, WORKERS_TRAIN // 2))

assert OUT_ROOT, "Set BYBIT_OUT_ROOT to the root created by offline_ingest.py"

# ---------------- Helper: read meta ----------------
def read_json(path: Path) -> dict:
    with open(path, "r") as f:
        return json.load(f)

def load_global_meta(out_root: Path) -> dict:
    meta_path = out_root / "meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Not found: {meta_path}. Did you run offline_ingest.py?")
    meta = read_json(meta_path)
    # sanity
    assert isinstance(meta.get("weeks", []), list) or isinstance(meta.get("week_counts", {}), dict), "Malformed meta.json"
    return meta

def resolve_week_meta_paths(out_root: Path, meta: dict) -> List[Path]:
    w2m = meta.get("weeks_meta", {})
    weeks = meta.get("weeks", [])
    if w2m and weeks:
        # preferred path: explicit mapping and ordered list provided
        return [out_root / w2m[w] for w in weeks if w in w2m]
    # fallback: scan each week dir
    paths = []
    for w in weeks:
        p = out_root / w / "meta_week.json"
        if p.exists():
            paths.append(p)
    return paths

def choose_splits(week_meta_paths: List[Path]) -> Tuple[List[Path], List[Path], List[Path]]:
    weeks = week_meta_paths
    if len(weeks) >= 24:
        weeks = weeks[-24:]
        tr, va, te = weeks[:18], weeks[18:21], weeks[21:24]
    else:
        n = len(weeks)
        n_tr = max(1, int(round(n * 0.75)))
        n_rest = n - n_tr
        n_va = max(1, int(round(n_rest / 2)))
        n_te = max(1, n - n_tr - n_va)
        tr = weeks[:n_tr]
        va = weeks[n_tr:n_tr + n_va]
        te = weeks[n_tr + n_va:]
    return tr, va, te

# ---------------- Chunk refs ----------------
@dataclass
class ChunkRef:
    week_dir: Path
    core_file: Path
    aux_file: Path
    y_file: Path
    n: int

def build_chunk_refs(meta_week_path: Path) -> List[ChunkRef]:
    wmeta = read_json(meta_week_path)
    week_dir = meta_week_path.parent
    refs: List[ChunkRef] = []
    for ch in wmeta.get("chunks", []):
        files = ch["files"]
        refs.append(ChunkRef(
            week_dir=week_dir,
            core_file=week_dir / files["core"],
            aux_file=week_dir / files["aux"],
            y_file=week_dir / files["y"],
            n=int(ch["n"]),
        ))
    return refs

# ---------------- Dataset (streaming from .npy chunks) ----------------
class NpyChunksDataset(Dataset):
    def __init__(self, chunk_refs: List[ChunkRef], feature_dim_total: int, pca_core=None):
        """
        chunk_refs: list of chunks in chronological order (kept as given)
        feature_dim_total: F (including AUX_DIM)
        pca_core: fitted PCA/IncrementalPCA to apply on core features, or None
        """
        self.refs = list(chunk_refs)
        self.F = int(feature_dim_total)
        self.F_core = self.F - AUX_DIM
        assert self.F_core > 0
        self.pca = pca_core

        # prefix sums for O(log N) lookup
        self.starts = []
        total = 0
        for r in self.refs:
            self.starts.append(total)
            total += r.n
        self.total = total

        # small cache of currently loaded memory-mapped arrays per process
        self._cache: Dict[Tuple[str, int], Tuple[np.memmap, np.memmap, np.memmap]] = {}
        self._lru: List[Tuple[str, int]] = []
        self._cap = 8  # keep up to 8 chunks mapped

    def __len__(self):
        return self.total

    def _locate(self, idx: int) -> Tuple[int, int]:
        # binary search on starts
        lo, hi = 0, len(self.starts) - 1
        while lo <= hi:
            mid = (lo + hi) // 2
            start = self.starts[mid]
            next_start = self.starts[mid + 1] if mid + 1 < len(self.starts) else self.total
            if start <= idx < next_start:
                return mid, idx - start
            elif idx < start:
                hi = mid - 1
            else:
                lo = mid + 1
        raise IndexError(idx)

    def _load_chunk(self, i: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        ref = self.refs[i]
        key = (str(ref.week_dir), i)
        if key in self._cache:
            # move to end of LRU
            try:
                self._lru.remove(key)
            except ValueError:
                pass
            self._lru.append(key)
            return self._cache[key]
        # mmap lazy
        Xc = np.load(ref.core_file, mmap_mode='r')
        Xa = np.load(ref.aux_file,  mmap_mode='r')
        Y  = np.load(ref.y_file,    mmap_mode='r')
        self._cache[key] = (Xc, Xa, Y)
        self._lru.append(key)
        if len(self._lru) > self._cap:
            evict_key = self._lru.pop(0)
            try:
                del self._cache[evict_key]
            except KeyError:
                pass
        return Xc, Xa, Y

    def __getitem__(self, idx: int):
        ci, offset = self._locate(idx)
        Xc, Xa, Y = self._load_chunk(ci)
        core = np.asarray(Xc[offset], dtype=np.float32)   # [L, F_core]
        if self.pca is not None:
            L, F_core = core.shape
            core2d = core.reshape(-1, F_core)
            core_p = self.pca.transform(core2d).astype(np.float32, copy=False).reshape(L, -1)
            aux = np.asarray(Xa[offset], dtype=np.float32)
            x = np.concatenate([core_p, aux], axis=-1)
        else:
            aux = np.asarray(Xa[offset], dtype=np.float32)
            x = np.concatenate([core, aux], axis=-1)
        y = np.asarray(Y[offset], dtype=np.float32)
        return torch.from_numpy(x), torch.from_numpy(y)

# ---------------- In-memory convenience (optional) ----------------
def load_split_in_memory(split_week_paths: List[Path]) -> Tuple[np.ndarray, np.ndarray, int]:
    """Concatenate all chunks fully into RAM. Returns X [N, L, F], y [N, 2H], F"""
    Xs, Ys = [], []
    feat_dim = None
    for wp in split_week_paths:
        wmeta = read_json(wp)
        F_total = int(wmeta["feature_dim_total"])
        if feat_dim is None:
            feat_dim = F_total
        elif feat_dim != F_total:
            raise ValueError(f"Feature dim mismatch between weeks: {feat_dim} vs {F_total}")
        for ch in wmeta.get("chunks", []):
            d = wp.parent
            Xc = np.load(d / ch["files"]["core"])
            Xa = np.load(d / ch["files"]["aux"])
            Y  = np.load(d / ch["files"]["y"])
            Xs.append(np.concatenate([Xc, Xa], axis=-1))
            Ys.append(Y)
    if not Xs:
        return np.empty((0, LOOKBACK, feat_dim or 0), np.float32), np.empty((0, 2*NUM_HORIZONS), np.float32), (feat_dim or 0)
    X = np.concatenate(Xs, axis=0).astype(np.float32, copy=False)
    y = np.concatenate(Ys, axis=0).astype(np.float32, copy=False)
    return X, y, int(feat_dim)

# ---------------- PCA fitting (optional) ----------------
def maybe_fit_pca_core(train_week_paths: List[Path], target_variance: float):
    if target_variance <= 0:
        return None, None  # disabled
    try:
        from sklearn.decomposition import PCA, IncrementalPCA
    except Exception as e:
        print(f"[warn] sklearn not available for PCA ({e}). Proceeding without PCA.")
        return None, None

    # Discover feature dims from first week
    w0 = read_json(train_week_paths[0])
    F_total = int(w0["feature_dim_total"])
    F_core = F_total - AUX_DIM
    if F_core <= 0:
        return None, None

    if PCA_INCREMENTAL:
        ipca = IncrementalPCA(n_components=target_variance)
        total_rows = 0
        print(f"[PCA] Fitting IncrementalPCA (target var={target_variance}) on core dims ({F_core}) ...")
        for wp in train_week_paths:
            wm = read_json(wp)
            d = wp.parent
            for ch in wm.get("chunks", []):
                Xc = np.load(d / ch["files"]["core"])
                n = Xc.shape[0]
                core2d = Xc.reshape(n * Xc.shape[1], F_core)  # [n*L, F_core]
                ipca.partial_fit(core2d)
                total_rows += core2d.shape[0]
        print(f"[PCA] IncrementalPCA trained on ~{total_rows:,} rows")
        return ipca, F_total
    else:
        # Full PCA on concatenated core from all train chunks (may be RAM heavy)
        print(f"[PCA] Fitting full PCA (target var={target_variance}) on core dims ({F_core}) ...")
        core_rows = []
        for wp in train_week_paths:
            wm = read_json(wp)
            d = wp.parent
            for ch in wm.get("chunks", []):
                Xc = np.load(d / ch["files"]["core"])
                n = Xc.shape[0]
                core_rows.append(Xc.reshape(n * Xc.shape[1], F_core))
        if not core_rows:
            return None, None
        core2d = np.concatenate(core_rows, axis=0)
        pca = PCA(n_components=target_variance, svd_solver="full")
        pca.fit(core2d)
        print(f"[PCA] kept {pca.n_components_} PCs (target var={target_variance})")
        return pca, F_total

# ---------------- Directional-mask quantiles from TRAIN set ----------------
def compute_dir_mask_quantiles_from_ytrain(y_train: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    y_ret = y_train[:, :NUM_HORIZONS].astype(np.float32)
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

    pos_lo_list = []
    pos_hi_list = []
    neg_lo_list = []
    neg_hi_list = []
    print("[dir-mask quantiles]")
    for idx, horizon in enumerate(HORIZONS_MS):
        horizon_returns = y_ret[:, idx]
        pos_returns = horizon_returns[horizon_returns > 0]
        neg_returns = horizon_returns[horizon_returns < 0]
        pos_lo, pos_hi = _compute_trim_bounds(pos_returns)
        neg_lo, neg_hi = _compute_trim_bounds((-neg_returns))
        pos_lo_list.append(pos_lo); pos_hi_list.append(pos_hi)
        neg_lo_list.append(neg_lo); neg_hi_list.append(neg_hi)
        print(f"  {horizon}ms → pos:[{pos_lo:.3e}, {pos_hi:.3e}]  neg|mag:[{neg_lo:.3e}, {neg_hi:.3e}] (tail {DIR_MASK_TAIL_FRACTION:.2%})")
    return (
        np.array(pos_lo_list, dtype=np.float32),
        np.array(pos_hi_list, dtype=np.float32),
        np.array(neg_lo_list, dtype=np.float32),
        np.array(neg_hi_list, dtype=np.float32),
    )

def make_build_dir_mask_torch(pos_lo, pos_hi, neg_lo, neg_hi):
    pos_lo_t = torch.from_numpy(pos_lo)
    pos_hi_t = torch.from_numpy(pos_hi)
    neg_lo_t = torch.from_numpy(neg_lo)
    neg_hi_t = torch.from_numpy(neg_hi)

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
    return build_dir_mask

def compute_directional_loss_fn(build_dir_mask_fn, horizon_weights: torch.Tensor):
    def compute_directional_loss(logits: torch.Tensor, y_ret: torch.Tensor) -> torch.Tensor:
        mask = build_dir_mask_fn(y_ret)
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
    return compute_directional_loss

# ---------------- Train/Eval ----------------
def train_from_offline():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    out_root = Path(OUT_ROOT)
    meta = load_global_meta(out_root)
    week_meta_paths = resolve_week_meta_paths(out_root, meta)
    if not week_meta_paths:
        raise RuntimeError("No week meta files were found under OUT_ROOT")
    tr_weeks, va_weeks, te_weeks = choose_splits(week_meta_paths)
    print(f"[weeks] train={len(tr_weeks)} val={len(va_weeks)} test={len(te_weeks)}")

    # feature dim sanity
    feat_dim_total = None
    for wp in tr_weeks + va_weeks + te_weeks:
        fm = int(read_json(wp)["feature_dim_total"])
        if feat_dim_total is None:
            feat_dim_total = fm
        elif feat_dim_total != fm:
            raise ValueError(f"Feature dim mismatch: saw {feat_dim_total} then {fm}")
    F_total = int(feat_dim_total or 0)
    F_core = F_total - AUX_DIM

    # ---- build datasets or fully load ----
    if USE_IN_MEMORY:
        X_tr, y_tr, feat_dim1 = load_split_in_memory(tr_weeks)
        X_va, y_va, feat_dim2 = load_split_in_memory(va_weeks)
        X_te, y_te, feat_dim3 = load_split_in_memory(te_weeks)
        assert feat_dim1 == feat_dim2 == feat_dim3 == F_total, "feat dim mismatch"
        # optional PCA (full style, like CMSSL17.py) on *core* dims
        pca_model = None
        if PCA_VAR_ENV > 0:
            try:
                from sklearn.decomposition import PCA
                pca = PCA(n_components=PCA_VAR_ENV, svd_solver="full")
                core2d = X_tr[:, :, :F_core].reshape(-1, F_core)
                pca.fit(core2d)
                def _apply(arr):
                    n, L, _ = arr.shape
                    core = arr[:, :, :F_core].reshape(-1, F_core)
                    core_p = pca.transform(core).astype(np.float32, copy=False).reshape(n, L, -1)
                    return np.concatenate([core_p, arr[:, :, F_core:]], axis=-1)
                X_tr = _apply(X_tr); X_va = _apply(X_va); X_te = _apply(X_te)
                new_feat_dim = X_tr.shape[-1]
                print(f"[PCA] kept {new_feat_dim - AUX_DIM} PCs (+{AUX_DIM} aux) → feat_dim={new_feat_dim}")
                F_total = int(new_feat_dim)
            except Exception as e:
                print(f"[warn] PCA requested but failed ({e}); continuing without PCA.")

        ds_train = HFTDataset(X_tr, y_tr)  # reuse CMSSL17 dataset class
        ds_val   = HFTDataset(X_va, y_va)
        ds_test  = HFTDataset(X_te, y_te)
        # we will still need y_tr to build directional mask quantiles
        y_train_for_quant = y_tr
    else:
        # streaming datasets from chunk refs
        tr_refs, va_refs, te_refs = [], [], []
        for w in tr_weeks: tr_refs.extend(build_chunk_refs(w))
        for w in va_weeks: va_refs.extend(build_chunk_refs(w))
        for w in te_weeks: te_refs.extend(build_chunk_refs(w))

        # fit (optional) PCA on training core dims
        pca_model, _F_total_unused = maybe_fit_pca_core(tr_weeks, PCA_VAR_ENV)
        if pca_model is not None:
            n_components = getattr(pca_model, "n_components_", None)
            if n_components is not None:
                F_total = int(n_components + AUX_DIM)
                print(f"[PCA] Using reduced feature dim: core={n_components}, aux={AUX_DIM} → F_total={F_total}")
            else:
                print("[warn] PCA model missing n_components_; keeping original feature dim.")
        # Build datasets
        ds_train = NpyChunksDataset(tr_refs, F_total, pca_core=pca_model)
        ds_val   = NpyChunksDataset(va_refs, F_total, pca_core=pca_model)
        ds_test  = NpyChunksDataset(te_refs, F_total, pca_core=pca_model)
        # Collect y_train for mask quantiles (labels are small; fine to concat in RAM)
        y_list = []
        for r in tr_refs:
            y_list.append(np.load(r.y_file))
        y_train_for_quant = np.concatenate(y_list, axis=0).astype(np.float32, copy=False)
        del y_list; gc.collect()

    # ---------------- directional mask quantiles & loss closure ----------------
    pos_lo, pos_hi, neg_lo, neg_hi = compute_dir_mask_quantiles_from_ytrain(y_train_for_quant)
    build_dir_mask = make_build_dir_mask_torch(pos_lo, pos_hi, neg_lo, neg_hi)
    horizon_weights = torch.tensor(HORIZON_WEIGHTS, dtype=torch.float32, device=device)
    horizon_weights_cpu = horizon_weights.detach().cpu().to(torch.float64)
    horizon_weights_np = horizon_weights_cpu.numpy()
    delta_ret_tensor = torch.as_tensor(DELTA_RET, dtype=torch.float32, device=device)
    delta_logvol_tensor = torch.as_tensor(DELTA_LOGVOL, dtype=torch.float32, device=device)
    compute_directional_loss = compute_directional_loss_fn(build_dir_mask, horizon_weights)

    # ---------------- DataLoaders ----------------
    dl_train = DataLoader(ds_train, BATCH_SIZE, shuffle=True, drop_last=True,
                          num_workers=WORKERS_TRAIN, pin_memory=True, prefetch_factor=4 if WORKERS_TRAIN>0 else None)
    dl_val   = DataLoader(ds_val,   BATCH_SIZE, shuffle=False, num_workers=max(1, WORKERS_VAL))
    dl_test  = DataLoader(ds_test,  BATCH_SIZE, shuffle=False, num_workers=max(1, WORKERS_VAL))

    # ---------------- Model ----------------
    args = ModelArgs(DMODEL, MAMBA_LAYERS, F_total, LOOKBACK)
    model = SAMBA(args).to(device)
    opt = SAM(model.parameters(), torch.optim.AdamW, lr=LR, weight_decay=1e-3, rho=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt.base_optimizer, mode='min', factor=0.5, patience=7)
    torch.cuda.empty_cache()

    # ---------------- Epoch loop ----------------
    best = float('inf'); no_imp = 0
    ema_pre = {'recon': 1.0, 'cpc': 1.0}
    ema_ft  = {'ret': 1.0, 'logvol': 1.0, 'bce': 1.0, 'recon': 1.0, 'cpc': 1.0}

    for epoch in range(EPOCHS):
        early_stop_triggered = False
        # Warmup LR
        warmup_factor = min(1.0, (epoch + 1) / WARMUP_EPOCHS) if epoch < WARMUP_EPOCHS else 1.0
        for pg in opt.base_optimizer.param_groups:
            pg['lr'] = LR * warmup_factor

        model.train()
        total_loss = 0.0
        mratio = MASK_PRETRAIN if epoch < SSL_PRETRAIN_EPOCHS else MASK_FINETUNE
        is_ssl_pretrain = (epoch < SSL_PRETRAIN_EPOCHS)

        pbar = tqdm(dl_train, desc=f"Ep{epoch+1}/{EPOCHS} ({'SSL-Pre' if is_ssl_pretrain else 'FT'}) mask={mratio:.2f}")
        ep_ret = ep_logvol = ep_bce = ep_recon = ep_cpc = 0.0
        n_batches = 0

        for x, y in pbar:
            x, y = x.to(device), y.to(device)

            # ===== SAM pass #1 =====
            opt.base_optimizer.zero_grad()
            ret_pred, vol_pred, dir_pred_logits, h_clean, h_masked, mask_idx, cpc_loss = model(x, mask_ratio=mratio)

            # Recon (Mamba-space distillation): target = h_clean (stop-grad)
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

                ema_ret    = ema_update('ret',    mse_ret.item(),    ema_ft)
                ema_logvol = ema_update('logvol', mse_vol.item(),    ema_ft)
                ema_bce    = ema_update('bce',    bce_loss.item(),   ema_ft)
                ema_recon  = ema_update('recon',  recon.item(),      ema_ft)
                ema_cpc    = ema_update('cpc',    cpc_loss.item(),   ema_ft)
                loss = (mse_ret / (ema_ret + 1e-8) +
                        mse_vol / (ema_logvol + 1e-8) +
                        LAMBDA_BCE * (bce_loss / (ema_bce + 1e-8)) +
                        LAMBDA_RECON_FT * (recon / (ema_recon + 1e-8)) +
                        LAMBDA_CPC_FT   * (cpc_loss / (ema_cpc + 1e-8)))
                ep_ret += mse_ret.item(); ep_logvol += mse_vol.item(); ep_bce += bce_loss.item(); ep_recon += recon.item(); ep_cpc += cpc_loss.item()

            loss.backward()
            opt.first_step(zero_grad=True)

            # ===== SAM pass #2 =====
            ret_pred2, vol_pred2, dir_pred_logits2, h_clean2, h_masked2, _, cpc_loss2 = model(
                x, mask_ratio=mratio, mask_idx=mask_idx
            )

            # Recompute recon using original mask indices
            recon2 = F.mse_loss(h_masked2[batch_idx, mask_idx], h_clean2.detach()[batch_idx, mask_idx])
            if is_ssl_pretrain:
                # reuse same loss components
                loss2 = LAMBDA_RECON_PT * (recon2 / (ema_pre['recon'] + 1e-8)) + LAMBDA_CPC_PT * (cpc_loss2 / (ema_pre['cpc'] + 1e-8))
            else:
                y_ret = y[:, :NUM_HORIZONS]
                y_logvol = y[:, NUM_HORIZONS:2 * NUM_HORIZONS]
                mse_ret2 = huber_loss(ret_pred2, y_ret, delta_ret_tensor, weights=horizon_weights)
                mse_vol2 = huber_loss(vol_pred2, y_logvol, delta_logvol_tensor, weights=horizon_weights)
                bce_loss2 = compute_directional_loss(dir_pred_logits2, y_ret)
                loss2 = (mse_ret2 / (ema_ft['ret'] + 1e-8) +
                         mse_vol2 / (ema_ft['logvol'] + 1e-8) +
                         LAMBDA_BCE * (bce_loss2 / (ema_ft['bce'] + 1e-8)) +
                         LAMBDA_RECON_FT * (recon2 / (ema_ft['recon'] + 1e-8)) +
                         LAMBDA_CPC_FT   * (cpc_loss2 / (ema_ft['cpc'] + 1e-8)))
            loss2.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10_000)
            opt.second_step(zero_grad=True)

            total_loss += float(loss.item())
            n_batches += 1
            if is_ssl_pretrain:
                pbar.set_postfix(loss=f"{(total_loss/n_batches):.4f}", recon=f"{ep_recon/max(1,n_batches):.4f}", cpc=f"{ep_cpc/max(1,n_batches):.4f}")
            else:
                pbar.set_postfix(loss=f"{(total_loss/n_batches):.4f}", ret=f"{ep_ret/max(1,n_batches):.4f}", vol=f"{ep_logvol/max(1,n_batches):.4f}", bce=f"{ep_bce/max(1,n_batches):.4f}")

        # ---------------- Validation ----------------
        model.eval()
        with torch.no_grad():
            val_ret_loss_sum = np.zeros(NUM_HORIZONS, dtype=np.float64)
            val_vol_loss_sum = np.zeros(NUM_HORIZONS, dtype=np.float64)
            val_sample_total = 0
            val_acc_sum = np.zeros(NUM_HORIZONS, dtype=np.float64)
            val_total   = 0

            val_bce_unmasked_sum = np.zeros(NUM_HORIZONS, dtype=np.float64)
            val_bce_unmasked_count = np.zeros(NUM_HORIZONS, dtype=np.float64)
            val_bce_masked_sum = np.zeros(NUM_HORIZONS, dtype=np.float64)
            val_bce_masked_count = np.zeros(NUM_HORIZONS, dtype=np.float64)
            val_acc_masked_sum = np.zeros(NUM_HORIZONS, dtype=np.float64)
            val_masked_total = np.zeros(NUM_HORIZONS, dtype=np.float64)

            val_logits_all = [[] for _ in range(NUM_HORIZONS)]
            val_ypos_all   = [[] for _ in range(NUM_HORIZONS)]
            val_logits_masked = [[] for _ in range(NUM_HORIZONS)]
            val_ypos_masked   = [[] for _ in range(NUM_HORIZONS)]

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

            # Aggregate metrics
            avg_val_ret_loss_per_h = val_ret_loss_sum / max(val_sample_total, 1)
            avg_val_vol_loss_per_h = val_vol_loss_sum / max(val_sample_total, 1)
            avg_val_ret_loss = float(
                np.dot(avg_val_ret_loss_per_h, horizon_weights_np)
                / max(horizon_weights_cpu.sum().item(), 1e-12)
            )
            avg_val_vol_loss = float(
                np.dot(avg_val_vol_loss_per_h, horizon_weights_np)
                / max(horizon_weights_cpu.sum().item(), 1e-12)
            )

            # BCE
            val_bce_unmasked = val_bce_unmasked_sum / np.maximum(val_bce_unmasked_count, 1)
            val_bce_masked   = val_bce_masked_sum / np.maximum(val_bce_masked_count, 1)

            # Accuracy
            val_acc = val_acc_sum / np.maximum(val_total, 1)
            val_acc_masked = val_acc_masked_sum / np.maximum(val_masked_total, 1)

            # AUCs
            val_auc = np.zeros(NUM_HORIZONS, dtype=np.float64)
            val_auc_masked = np.zeros(NUM_HORIZONS, dtype=np.float64)
            for h_idx in range(NUM_HORIZONS):
                if val_logits_all[h_idx]:
                    logits_cat = torch.cat(val_logits_all[h_idx], dim=0).view(-1)
                    ypos_cat   = torch.cat(val_ypos_all[h_idx], dim=0).view(-1)
                    val_auc[h_idx] = binary_auc_from_logits(logits_cat, ypos_cat)
                else:
                    val_auc[h_idx] = float('nan')
                if val_logits_masked[h_idx]:
                    logits_cat = torch.cat(val_logits_masked[h_idx], dim=0).view(-1)
                    ypos_cat   = torch.cat(val_ypos_masked[h_idx], dim=0).view(-1)
                    val_auc_masked[h_idx] = binary_auc_from_logits(logits_cat, ypos_cat)
                else:
                    val_auc_masked[h_idx] = float('nan')

            def fmt_arr(arr, fmt="{:.5f}"):
                parts = []
                for h, v in zip(HORIZONS_MS, arr):
                    if isinstance(v, float):
                        val = v
                    else:
                        val = float(v)
                    if math.isnan(val) or math.isinf(val):
                        parts.append(f"{h}ms:nan")
                    else:
                        parts.append(f"{h}ms:{fmt.format(val)}")
                return "[" + ", ".join(parts) + "]"

            print(f"[val] ret={fmt_arr(avg_val_ret_loss_per_h)} (w_avg={avg_val_ret_loss:.4e})  "
                  f"vol={fmt_arr(avg_val_vol_loss_per_h)} (w_avg={avg_val_vol_loss:.4e})  "
                  f"BCE(all)={fmt_arr(val_bce_unmasked)}  BCE(mask)={fmt_arr(val_bce_masked)}  "
                  f"Acc(all)={fmt_arr(val_acc, '{:.3%}')}  Acc(mask)={fmt_arr(val_acc_masked, '{:.3%}')}  "
                  f"AUC(all)={fmt_arr(val_auc, '{:.3f}')}  AUC(mask)={fmt_arr(val_auc_masked, '{:.3f}')}")

            scheduler.step(avg_val_ret_loss)

            # checkpointing policy like CMSSL17: track best avg_val_ret_loss during fine-tuning
            if not is_ssl_pretrain:
                if avg_val_ret_loss < best:
                    best = float(avg_val_ret_loss)
                    no_imp = 0
                    ckpt = {
                        "epoch": epoch,
                        "model": model.state_dict(),
                        "args": {
                            "DMODEL": DMODEL, "MAMBA_LAYERS": MAMBA_LAYERS,
                            "feat_dim": F_total, "LOOKBACK": LOOKBACK,
                            "HORIZONS_MS": HORIZONS_MS,
                        },
                        "best_val_loss": best,
                    }
                    out_ckpt = out_root / "cmssl17_offline_best.pt"
                    torch.save(ckpt, out_ckpt)
                    print(f"[ckpt] saved best to {out_ckpt}")
                else:
                    no_imp += 1
                    print(f"no improve {no_imp}/{PATIENCE}")
                    if no_imp >= PATIENCE:
                        print("Early stopping triggered.")
                        early_stop_triggered = True

        if early_stop_triggered:
            break

        # (Optional) early stop on long stagnation
        # if no_imp > 50: break

    print("[done] Training complete.")

# ---------------- Lightweight HFTDataset (when loading into RAM) ----------------
class HFTDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = X.astype(np.float32, copy=False)
        self.y = y.astype(np.float32, copy=False)
    def __len__(self): return int(self.y.shape[0])
    def __getitem__(self, idx): 
        return torch.from_numpy(self.X[idx]), torch.from_numpy(self.y[idx])

# ---------------- Entry ----------------
if __name__ == "__main__":
    train_from_offline()
