import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from CMSSL17 import SAMBA, ModelArgs, DMODEL, MAMBA_LAYERS, LOOKBACK
from offline_tokens import iter_week_chunks, load_global_meta


def load_cmssl(out_root: str, ckpt_path: str, device: str = "cuda"):
    out_root = Path(out_root)
    meta = load_global_meta(out_root)
    feat_dim = int(meta["feature_dim_total"])  # includes AUX_DIM already

    args = ModelArgs(DMODEL, MAMBA_LAYERS, feat_dim, LOOKBACK)
    model = SAMBA(args).to(device)

    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, meta


@torch.no_grad()
def cmssl_predict(model, x_core, x_aux, meta, device: str = "cuda"):
    # x_core: [B, L, F_core]  x_aux: [B, L, AUX_DIM]
    x_core = torch.as_tensor(x_core, device=device)
    x_aux = torch.as_tensor(x_aux, device=device)
    x = torch.cat([x_core, x_aux], dim=-1)
    mask_idx = torch.empty((x.shape[0], 0), dtype=torch.long, device=device)
    ret_pred, vol_pred, dir_logits, *_ = model(x, mask_ratio=0.0, mask_idx=mask_idx)
    horizons = meta.get("horizons_ms", [])
    expected_h = len(horizons)
    assert expected_h > 0, "meta['horizons_ms'] must be non-empty"
    assert ret_pred.shape[-1] == expected_h, (
        f"ret_pred shape {ret_pred.shape} does not match horizons {expected_h}"
    )
    assert vol_pred.shape[-1] == expected_h, (
        f"vol_pred shape {vol_pred.shape} does not match horizons {expected_h}"
    )
    assert dir_logits.shape[-1] == expected_h, (
        f"dir_logits shape {dir_logits.shape} does not match horizons {expected_h}"
    )
    return ret_pred, vol_pred, dir_logits


def iter_chunk_batches(out_root: str):
    out_root = Path(out_root)
    meta = load_global_meta(out_root)
    for week, week_meta, week_dir in iter_week_chunks(out_root, meta=meta):
        for entry in week_meta.get("chunks", []):
            files = entry.get("files", {})
            x_core = np.load(week_dir / files["core"])
            x_aux = np.load(week_dir / files["aux"])
            y = np.load(week_dir / files["y"])
            ts = np.load(week_dir / files["ts"])
            yield week, int(entry.get("chunk", 0)), ts, x_core, x_aux, y


def _decision_ts_bounds(week_key: str, week_meta: dict) -> tuple[int, int]:
    ts_range = week_meta.get("decision_ts_range")
    assert ts_range, f"week {week_key} missing decision_ts_range in meta_week.json"
    ts_min = int(ts_range["min"])
    ts_max = int(ts_range["max"])
    assert ts_min < ts_max, f"week {week_key} has invalid decision_ts_range: {ts_range}"
    return ts_min, ts_max


def get_cmssl_splits(out_root: str) -> dict:
    out_root = Path(out_root)
    meta = load_global_meta(out_root)
    weeks = list(meta.get("weeks", []))
    assert len(weeks) == 2, f"expected exactly 2 weeks, found {len(weeks)}"

    week_meta_map = {wk: wmeta for wk, wmeta, _ in iter_week_chunks(out_root, meta=meta)}
    assert len(week_meta_map) == 2, f"expected two week metas, found {len(week_meta_map)}"
    week1_key, week2_key = weeks
    assert week1_key in week_meta_map and week2_key in week_meta_map, (
        f"week keys {weeks} do not match week metas {list(week_meta_map.keys())}"
    )

    week1_min, week1_max = _decision_ts_bounds(week1_key, week_meta_map[week1_key])
    week2_min, week2_max = _decision_ts_bounds(week2_key, week_meta_map[week2_key])

    week2_span = week2_max - week2_min
    expected_week_ms = 7 * 24 * 60 * 60 * 1000
    expected_half_ms = expected_week_ms / 2.0
    tolerance_ms = 60 * 60 * 1000
    assert abs(week2_span - expected_week_ms) <= tolerance_ms, (
        f"week2 span {week2_span:.0f}ms not ~7 days"
    )

    week2_half = week2_span / 2.0
    assert abs(week2_half - expected_half_ms) <= tolerance_ms, (
        f"week2 half span {week2_half:.0f}ms not ~3.5 days"
    )

    week2_mid = int(week2_min + week2_half)
    return {
        "train": {"week": week1_key, "start": week1_min, "end": week1_max},
        "val": {"week": week2_key, "start": week2_min, "end": week2_mid},
        "test": {"week": week2_key, "start": week2_mid, "end": week2_max},
    }


def build_two_week_time_splits(out_root: str) -> dict:
    return get_cmssl_splits(out_root)


def spread_bps_from_vol_pred(vol_pred, spread_mult: float = 1.0):
    """
    Convert model vol predictions into a spread size in basis points.

    vol_pred is trained against y_logvol (log volatility), so we recover
    sigma by exponentiating the log-vol and then scale to bps.
    If the model ever switches to predicting log-variance, use
    sigma = exp(0.5 * logvar) instead.
    """
    sigma = np.exp(vol_pred)
    sigma_bps = 1e4 * sigma
    return spread_mult * sigma_bps


def load_split_arrays(out_root: str, split: Dict[str, int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x_core_list: List[np.ndarray] = []
    x_aux_list: List[np.ndarray] = []
    y_list: List[np.ndarray] = []
    ts_list: List[np.ndarray] = []
    for week, _chunk, ts, x_core, x_aux, y in iter_chunk_batches(out_root):
        if week != split["week"]:
            continue
        mask = (ts >= split["start"]) & (ts < split["end"])
        if not np.any(mask):
            continue
        x_core_list.append(x_core[mask])
        x_aux_list.append(x_aux[mask])
        y_list.append(y[mask])
        ts_list.append(ts[mask])
    if not x_core_list:
        raise ValueError(f"No data found for split {split}")
    x_core_all = np.concatenate(x_core_list, axis=0)
    x_aux_all = np.concatenate(x_aux_list, axis=0)
    y_all = np.concatenate(y_list, axis=0)
    ts_all = np.concatenate(ts_list, axis=0)
    order = np.argsort(ts_all)
    return x_core_all[order], x_aux_all[order], y_all[order], ts_all[order]


def run_cmssl_inference(
    model,
    meta: dict,
    x_core: np.ndarray,
    x_aux: np.ndarray,
    batch_size: int = 256,
    device: str = "cuda",
) -> Dict[str, np.ndarray]:
    ret_preds: List[np.ndarray] = []
    vol_preds: List[np.ndarray] = []
    dir_logits_list: List[np.ndarray] = []
    n = x_core.shape[0]
    for i in range(0, n, batch_size):
        xc = x_core[i:i + batch_size]
        xa = x_aux[i:i + batch_size]
        ret_pred, vol_pred, dir_logits = cmssl_predict(model, xc, xa, meta, device=device)
        ret_preds.append(ret_pred.detach().cpu().numpy())
        vol_preds.append(vol_pred.detach().cpu().numpy())
        dir_logits_list.append(dir_logits.detach().cpu().numpy())
    return {
        "ret_pred": np.concatenate(ret_preds, axis=0),
        "vol_pred": np.concatenate(vol_preds, axis=0),
        "dir_logits": np.concatenate(dir_logits_list, axis=0),
    }


def _find_week_dir(out_root: Path, week_key: str) -> Path:
    meta = load_global_meta(out_root)
    for wk, _wmeta, wk_dir in iter_week_chunks(out_root, meta=meta):
        if wk == week_key:
            return wk_dir
    raise ValueError(f"Unable to locate week directory for {week_key}")


def load_raw_snapshots(out_root: str, week_key: str) -> Tuple[np.ndarray, np.ndarray]:
    week_dir = _find_week_dir(Path(out_root), week_key)
    candidates = [
        week_dir / "raw_snapshots.npz",
        week_dir / "snapshots.npz",
        week_dir / "raw_snapshots.npy",
        week_dir / "snapshots.npy",
    ]
    path = next((p for p in candidates if p.exists()), None)
    if path is None:
        raise FileNotFoundError(
            f"No raw snapshot file found in {week_dir}. Expected one of: "
            f"{', '.join(p.name for p in candidates)}"
        )

    if path.suffix == ".npz":
        data = np.load(path)
        if "ts" in data and "snapshots" in data:
            return data["ts"], data["snapshots"]
        if "timestamps" in data and "X" in data:
            return data["timestamps"], data["X"]
        raise ValueError(f"Unsupported npz layout in {path}")

    arr = np.load(path)
    if arr.dtype.names:
        if "ts" in arr.dtype.names and "snapshot" in arr.dtype.names:
            return arr["ts"], arr["snapshot"]
        if "ts" in arr.dtype.names and "snapshots" in arr.dtype.names:
            return arr["ts"], arr["snapshots"]
    ts_path = path.with_name("snapshots_ts.npy")
    if ts_path.exists():
        return np.load(ts_path), arr
    raise ValueError(f"Unsupported raw snapshot layout in {path}")


def align_snapshots_to_decisions(
    decision_ts: np.ndarray,
    snapshot_ts: np.ndarray,
    snapshots: np.ndarray,
    tolerance_ms: int = 1000,
) -> Tuple[np.ndarray, np.ndarray]:
    if snapshot_ts.ndim != 1:
        raise ValueError("snapshot_ts must be 1D")
    order = np.argsort(snapshot_ts)
    snapshot_ts = snapshot_ts[order]
    snapshots = snapshots[order]
    idx = np.searchsorted(snapshot_ts, decision_ts, side="right") - 1
    idx = np.clip(idx, 0, len(snapshot_ts) - 1)
    aligned = snapshots[idx]
    delta = np.abs(snapshot_ts[idx] - decision_ts)
    mask = delta <= tolerance_ms
    aligned = aligned.astype(np.float32)
    aligned[~mask] = np.nan
    return aligned, mask


def join_features(
    decision_ts: np.ndarray,
    y: np.ndarray,
    cmssl_out: Dict[str, np.ndarray],
    snapshots: np.ndarray,
    snapshot_mask: np.ndarray,
) -> Dict[str, np.ndarray]:
    ret_pred = cmssl_out["ret_pred"]
    vol_pred = cmssl_out["vol_pred"]
    dir_logits = cmssl_out["dir_logits"]
    spread_bps = spread_bps_from_vol_pred(vol_pred[:, 0])

    features = np.concatenate(
        [
            ret_pred,
            vol_pred,
            dir_logits,
            snapshots,
        ],
        axis=-1,
    )
    return {
        "ts": decision_ts,
        "features": features.astype(np.float32),
        "y": y.astype(np.float32),
        "spread_bps": spread_bps.astype(np.float32),
        "snapshot_mask": snapshot_mask.astype(np.bool_),
    }


def build_joined_split(
    out_root: str,
    split: Dict[str, int],
    model,
    meta: dict,
    device: str,
    batch_size: int = 256,
) -> Dict[str, np.ndarray]:
    x_core, x_aux, y, ts = load_split_arrays(out_root, split)
    cmssl_out = run_cmssl_inference(model, meta, x_core, x_aux, batch_size=batch_size, device=device)
    snapshot_ts, snapshots = load_raw_snapshots(out_root, split["week"])
    aligned_snapshots, snapshot_mask = align_snapshots_to_decisions(ts, snapshot_ts, snapshots)
    return join_features(ts, y, cmssl_out, aligned_snapshots, snapshot_mask)


def chronological_split(
    data: Dict[str, np.ndarray],
    ratios: Tuple[float, float, float] = (0.6, 0.2, 0.2),
) -> Dict[str, Dict[str, np.ndarray]]:
    assert abs(sum(ratios) - 1.0) < 1e-6
    n = len(data["ts"])
    n_train = int(n * ratios[0])
    n_val = int(n * ratios[1])
    idx_train = slice(0, n_train)
    idx_val = slice(n_train, n_train + n_val)
    idx_test = slice(n_train + n_val, n)

    def _slice(idx: slice) -> Dict[str, np.ndarray]:
        return {key: value[idx] for key, value in data.items()}

    return {
        "train": _slice(idx_train),
        "val": _slice(idx_val),
        "test": _slice(idx_test),
    }


@dataclass
class TradingBatch:
    features: np.ndarray
    returns: np.ndarray
    spread_bps: np.ndarray


class TradingEnv:
    def __init__(self, batch: TradingBatch):
        self.features = batch.features
        self.returns = batch.returns
        self.spread_bps = batch.spread_bps
        self.n = len(self.returns)
        self.idx = 0
        self.position = 0
        self.total_reward = 0.0

    def reset(self) -> np.ndarray:
        self.idx = 0
        self.position = 0
        self.total_reward = 0.0
        return self.features[self.idx]

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, float]]:
        action_map = {0: -1, 1: 0, 2: 1}
        action_sign = action_map.get(action, 0)
        ret = float(self.returns[self.idx])
        spread_cost = float(self.spread_bps[self.idx]) * 1e-4
        turnover = abs(action_sign - self.position)
        reward = action_sign * ret - turnover * spread_cost
        self.total_reward += reward
        self.position = action_sign
        self.idx += 1
        done = self.idx >= self.n
        next_obs = self.features[self.idx - 1] if done else self.features[self.idx]
        info = {
            "reward": reward,
            "total_reward": self.total_reward,
        }
        return next_obs, reward, done, info


class PolicyValueNet(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128, action_dim: int = 3):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.policy_head = nn.Linear(hidden_dim, action_dim)
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.shared(x)
        logits = self.policy_head(h)
        value = self.value_head(h).squeeze(-1)
        return logits, value


@dataclass
class PPOConfig:
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    lr: float = 3e-4
    update_epochs: int = 4
    batch_size: int = 256


def collect_rollout(env: TradingEnv, model: PolicyValueNet, device: str) -> Dict[str, torch.Tensor]:
    obs_list = []
    action_list = []
    logp_list = []
    value_list = []
    reward_list = []
    done_list = []

    obs = env.reset()
    done = False
    while not done:
        obs_t = torch.from_numpy(obs).float().to(device)
        logits, value = model(obs_t.unsqueeze(0))
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        logp = dist.log_prob(action)
        next_obs, reward, done, _info = env.step(int(action.item()))

        obs_list.append(obs_t)
        action_list.append(action)
        logp_list.append(logp)
        value_list.append(value.squeeze(0))
        reward_list.append(torch.tensor(reward, dtype=torch.float32, device=device))
        done_list.append(torch.tensor(done, dtype=torch.float32, device=device))
        obs = next_obs

    return {
        "obs": torch.stack(obs_list),
        "actions": torch.stack(action_list),
        "logp": torch.stack(logp_list),
        "values": torch.stack(value_list),
        "rewards": torch.stack(reward_list),
        "dones": torch.stack(done_list),
    }


def compute_gae(rewards: torch.Tensor, values: torch.Tensor, dones: torch.Tensor, gamma: float, lam: float):
    advantages = torch.zeros_like(rewards)
    last_gae = 0.0
    next_value = 0.0
    for t in reversed(range(len(rewards))):
        mask = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_value * mask - values[t]
        last_gae = delta + gamma * lam * mask * last_gae
        advantages[t] = last_gae
        next_value = values[t]
    returns = advantages + values
    return advantages, returns


def ppo_update(
    model: PolicyValueNet,
    optimizer: optim.Optimizer,
    rollout: Dict[str, torch.Tensor],
    config: PPOConfig,
    device: str,
):
    obs = rollout["obs"].to(device)
    actions = rollout["actions"].to(device)
    old_logp = rollout["logp"].detach().to(device)
    values = rollout["values"].detach().to(device)
    rewards = rollout["rewards"].to(device)
    dones = rollout["dones"].to(device)

    advantages, returns = compute_gae(rewards, values, dones, config.gamma, config.gae_lambda)
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    n = obs.shape[0]
    indices = torch.arange(n)
    for _ in range(config.update_epochs):
        perm = indices[torch.randperm(n)]
        for start in range(0, n, config.batch_size):
            mb_idx = perm[start:start + config.batch_size]
            logits, value = model(obs[mb_idx])
            dist = torch.distributions.Categorical(logits=logits)
            logp = dist.log_prob(actions[mb_idx])
            ratio = torch.exp(logp - old_logp[mb_idx])
            clip_adv = torch.clamp(ratio, 1.0 - config.clip_ratio, 1.0 + config.clip_ratio) * advantages[mb_idx]
            policy_loss = -(torch.min(ratio * advantages[mb_idx], clip_adv)).mean()
            value_loss = nn.functional.mse_loss(value, returns[mb_idx])
            entropy_loss = dist.entropy().mean()
            loss = policy_loss + 0.5 * value_loss - 0.01 * entropy_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


def train_ppo(env: TradingEnv, input_dim: int, device: str = "cuda", epochs: int = 10) -> PolicyValueNet:
    model = PolicyValueNet(input_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=3e-4)
    config = PPOConfig()

    for _ in range(epochs):
        rollout = collect_rollout(env, model, device)
        ppo_update(model, optimizer, rollout, config, device)
    return model


def evaluate_policy(env: TradingEnv, model: PolicyValueNet, device: str = "cuda") -> Dict[str, float]:
    obs = env.reset()
    done = False
    rewards = []
    while not done:
        obs_t = torch.from_numpy(obs).float().to(device)
        logits, _value = model(obs_t.unsqueeze(0))
        action = torch.argmax(logits, dim=-1)
        obs, reward, done, _info = env.step(int(action.item()))
        rewards.append(reward)
    rewards_arr = np.array(rewards, dtype=np.float32)
    return {
        "total_reward": float(rewards_arr.sum()),
        "mean_reward": float(rewards_arr.mean()) if rewards_arr.size else 0.0,
    }


def report_cmssl_metrics(y_true: np.ndarray, cmssl_out: Dict[str, np.ndarray]) -> Dict[str, float]:
    num_h = y_true.shape[1] // 2
    y_ret = y_true[:, :num_h]
    y_vol = y_true[:, num_h:]
    ret_pred = cmssl_out["ret_pred"]
    vol_pred = cmssl_out["vol_pred"]
    ret_mae = float(np.mean(np.abs(ret_pred - y_ret)))
    vol_mae = float(np.mean(np.abs(vol_pred - y_vol)))
    return {
        "ret_mae": ret_mae,
        "vol_mae": vol_mae,
    }


def run_pipeline(
    out_root: str,
    ckpt_path: str,
    device: str = "cuda",
    ppo_epochs: int = 10,
) -> Dict[str, Dict[str, float]]:
    meta = load_global_meta(Path(out_root))
    splits = build_two_week_time_splits(out_root)

    model, _meta = load_cmssl(out_root, ckpt_path, device=device)

    joined_train = build_joined_split(out_root, splits["train"], model, meta, device)
    joined_val = build_joined_split(out_root, splits["val"], model, meta, device)
    joined_test = build_joined_split(out_root, splits["test"], model, meta, device)

    num_h = len(meta.get("horizons_ms", []))
    cmssl_report = report_cmssl_metrics(
        joined_test["y"],
        {
            "ret_pred": joined_test["features"][:, :num_h],
            "vol_pred": joined_test["features"][:, num_h:2 * num_h],
            "dir_logits": joined_test["features"][:, 2 * num_h:3 * num_h],
        },
    )

    joined = {
        key: np.concatenate([joined_train[key], joined_val[key], joined_test[key]], axis=0)
        for key in joined_train.keys()
    }
    order = np.argsort(joined["ts"])
    joined = {key: value[order] for key, value in joined.items()}

    splits_rl = chronological_split(joined, ratios=(0.6, 0.2, 0.2))

    def _to_env(split: Dict[str, np.ndarray]) -> TradingEnv:
        returns = split["y"][:, 0]
        batch = TradingBatch(
            features=split["features"],
            returns=returns,
            spread_bps=split["spread_bps"],
        )
        return TradingEnv(batch)

    train_env = _to_env(splits_rl["train"])
    val_env = _to_env(splits_rl["val"])
    test_env = _to_env(splits_rl["test"])

    input_dim = train_env.features.shape[-1]
    ppo_model = train_ppo(train_env, input_dim, device=device, epochs=ppo_epochs)

    val_report = evaluate_policy(val_env, ppo_model, device=device)
    test_report = evaluate_policy(test_env, ppo_model, device=device)

    return {
        "cmssl_test": cmssl_report,
        "ppo_val": val_report,
        "ppo_test": test_report,
    }


if __name__ == "__main__":
    out_root = os.environ.get("BYBIT_OUT_ROOT", "").strip()
    ckpt_path = os.environ.get("BYBIT_CMSSL_CKPT", "").strip()
    device = os.environ.get("BYBIT_DEVICE", "cuda")
    ppo_epochs = int(os.environ.get("BYBIT_PPO_EPOCHS", "10"))

    if not out_root or not ckpt_path:
        raise SystemExit("Set BYBIT_OUT_ROOT and BYBIT_CMSSL_CKPT before running.")

    report = run_pipeline(out_root, ckpt_path, device=device, ppo_epochs=ppo_epochs)
    print("[cmssl test]", report["cmssl_test"])
    print("[ppo val]", report["ppo_val"])
    print("[ppo test]", report["ppo_test"])
