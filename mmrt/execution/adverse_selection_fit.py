"""Streaming baseline fitting for disk-backed adverse-selection datasets."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from mmrt.execution.adverse_selection_dataset import DiskBackedAdverseSelectionDataset


@dataclass(frozen=True, slots=True)
class AdverseBaselineFitResult:
    target_names: tuple[str, ...]
    feature_names: tuple[str, ...]
    train_rows: int
    val_rows: int
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    coefficients: np.ndarray
    intercepts: np.ndarray
    metrics: dict[str, object]


def _check_positive(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _split(n: int, train_fraction: float) -> tuple[int, int]:
    if not math.isfinite(train_fraction) or train_fraction <= 0.0 or train_fraction >= 1.0:
        raise ValueError("train_fraction must be in (0, 1)")
    if n < 2:
        return 0, n
    n_train = int(n * train_fraction)
    return min(max(n_train, 1), n - 1), n - min(max(n_train, 1), n - 1)


def _solve(lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    try:
        return np.linalg.solve(lhs, rhs)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(lhs, rhs, rcond=None)[0]


def _auc_exact(y: np.ndarray, score: np.ndarray) -> float | None:
    yb = y > 0.5
    n_pos = int(np.count_nonzero(yb)); n = len(yb); n_neg = n - n_pos
    if n_pos == 0 or n_neg == 0:
        return None
    order = np.argsort(score, kind="mergesort")
    sorted_score = score[order]
    ranks_sorted = np.empty(n, dtype=np.float64)
    i = 0
    while i < n:
        j = i + 1
        while j < n and sorted_score[j] == sorted_score[i]:
            j += 1
        ranks_sorted[i:j] = (i + 1 + j) / 2.0
        i = j
    ranks = np.empty(n, dtype=np.float64); ranks[order] = ranks_sorted
    return float((np.sum(ranks[yb]) - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))



@dataclass(slots=True)
class BinaryHistogramAUC:
    bins: int
    score_min: float | None = None
    score_max: float | None = None
    pos_hist: np.ndarray | None = None
    neg_hist: np.ndarray | None = None

    def __post_init__(self) -> None:
        self.bins = _check_positive(int(self.bins), "bins")
        if self.score_min is not None and self.score_max is not None:
            lo = float(self.score_min); hi = float(self.score_max)
            if not math.isfinite(lo) or not math.isfinite(hi):
                raise ValueError("score range must be finite")
            if hi <= lo:
                hi = lo + 1.0
            self.score_min = lo; self.score_max = hi
            self.pos_hist = np.zeros(self.bins, dtype=np.int64)
            self.neg_hist = np.zeros(self.bins, dtype=np.int64)

    def update_range(self, score: np.ndarray) -> None:
        if score.size == 0:
            return
        lo = float(np.min(score)); hi = float(np.max(score))
        self.score_min = lo if self.score_min is None else min(float(self.score_min), lo)
        self.score_max = hi if self.score_max is None else max(float(self.score_max), hi)

    def update(self, y: np.ndarray, score: np.ndarray) -> None:
        if self.pos_hist is None or self.neg_hist is None or self.score_min is None or self.score_max is None:
            self.__post_init__()
        assert self.pos_hist is not None and self.neg_hist is not None and self.score_min is not None and self.score_max is not None
        if score.size == 0:
            return
        span = max(float(self.score_max) - float(self.score_min), 1e-12)
        idx = np.floor((np.asarray(score, dtype=np.float64) - float(self.score_min)) / span * self.bins).astype(np.int64)
        idx = np.clip(idx, 0, self.bins - 1)
        pos = np.asarray(y) > 0.5
        self.pos_hist += np.bincount(idx[pos], minlength=self.bins).astype(np.int64)
        self.neg_hist += np.bincount(idx[~pos], minlength=self.bins).astype(np.int64)

    def auc(self) -> float | None:
        assert self.pos_hist is not None and self.neg_hist is not None
        n_pos = int(np.sum(self.pos_hist)); n_neg = int(np.sum(self.neg_hist))
        if n_pos == 0 or n_neg == 0:
            return None
        cum_neg = 0
        total = 0.0
        for p, n in zip(self.pos_hist, self.neg_hist):
            total += float(p) * cum_neg + 0.5 * float(p) * float(n)
            cum_neg += int(n)
        return float(total / (n_pos * n_neg))

def fit_adverse_baselines_streaming(
    dataset: DiskBackedAdverseSelectionDataset,
    *,
    target_names: tuple[str, ...],
    train_fraction: float,
    ridge_l2: float,
    min_train_samples: int,
    chunk_rows: int = 100_000,
    metrics_mode: str = "approx",
    auc_bins: int = 2000,
    exact_auc_max_rows: int = 1_000_000,
) -> AdverseBaselineFitResult:
    if not isinstance(dataset, DiskBackedAdverseSelectionDataset):
        raise ValueError("dataset must be DiskBackedAdverseSelectionDataset")
    chunk_rows = _check_positive(int(chunk_rows), "chunk_rows")
    min_train_samples = _check_positive(int(min_train_samples), "min_train_samples")
    if ridge_l2 < 0.0 or not math.isfinite(ridge_l2):
        raise ValueError("ridge_l2 must be nonnegative finite")
    if metrics_mode not in ("approx", "none", "exact"):
        raise ValueError("metrics_mode must be approx, none, or exact")
    auc_bins = _check_positive(int(auc_bins), "auc_bins")
    exact_auc_max_rows = _check_positive(int(exact_auc_max_rows), "exact_auc_max_rows")
    n = dataset.num_rows; nf = dataset.num_features
    train_rows_total, val_rows_total = _split(n, train_fraction)
    if n < 2:
        targets = {name: {"target_name": name, "train_rows": 0, "val_rows": 0, "skipped": True, "skip_reason": "not_enough_decisions"} for name in target_names}
        return AdverseBaselineFitResult((), dataset.feature_names, 0, 0, np.asarray([], dtype=np.float32), np.asarray([], dtype=np.float32), np.empty((0, nf), dtype=np.float64), np.asarray([], dtype=np.float64), {"enabled": True, "train_fraction": train_fraction, "ridge_l2": ridge_l2, "min_train_samples": min_train_samples, "train_rows_total": 0, "val_rows_total": 0, "fitted_target_count": 0, "requested_target_count": len(target_names), "targets": targets, "skipped": True, "skip_reason": "not_enough_decisions"})

    sum_x = np.zeros(nf, dtype=np.float64); sum_x2 = np.zeros(nf, dtype=np.float64)
    for start in range(0, train_rows_total, chunk_rows):
        end = min(start + chunk_rows, train_rows_total)
        X = np.asarray(dataset.arrays.features[start:end], dtype=np.float64)
        sum_x += np.sum(X, axis=0); sum_x2 += np.sum(X * X, axis=0)
    mean = sum_x / max(train_rows_total, 1)
    var = sum_x2 / max(train_rows_total, 1) - mean * mean
    scale = np.sqrt(np.maximum(var, 0.0)); scale = np.where(scale <= 1e-12, 1.0, scale)

    label_index = {name: i for i, name in enumerate(dataset.label_names)}
    targets_metrics: dict[str, object] = {}
    fitted_names: list[str] = []
    coefs: list[np.ndarray] = []
    intercepts: list[float] = []
    reg = np.eye(nf + 1, dtype=np.float64) * ridge_l2; reg[0, 0] = 0.0

    known = [name for name in target_names if name in label_index]
    for target_name in target_names:
        if target_name not in label_index:
            targets_metrics[target_name] = {"target_name": target_name, "train_rows": 0, "val_rows": 0, "skipped": True, "skip_reason": "unknown_target"}
    tidx = np.asarray([label_index[name] for name in known], dtype=np.int64)
    nt = len(known)

    # Pass over the dataset once for all targets: accumulate each target's
    # masked normal equations from the shared per-chunk design matrix.
    XtX = np.zeros((nt, nf + 1, nf + 1), dtype=np.float64)
    Xty = np.zeros((nt, nf + 1), dtype=np.float64)
    train_valid = np.zeros(nt, dtype=np.int64)
    val_valid = np.zeros(nt, dtype=np.int64)
    for start in range(0, n, chunk_rows):
        end = min(start + chunk_rows, n)
        masks = np.asarray(dataset.arrays.label_masks[start:end][:, tidx], dtype=np.bool_)
        if not masks.any():
            continue
        row_ids = np.arange(start, end)
        train_rows = row_ids < train_rows_total
        val_valid += np.count_nonzero(masks & ~train_rows[:, None], axis=0)
        train_masks = masks & train_rows[:, None]
        if not train_masks.any():
            continue
        X = np.asarray(dataset.arrays.features[start:end], dtype=np.float64)
        Xz = (X - mean) / scale
        aug = np.concatenate([np.ones((Xz.shape[0], 1), dtype=np.float64), Xz], axis=1)
        labels_chunk = np.asarray(dataset.arrays.labels[start:end][:, tidx], dtype=np.float64)
        for j in range(nt):
            m = train_masks[:, j]
            count = int(np.count_nonzero(m))
            if count == 0:
                continue
            rows = aug[m]
            XtX[j] += rows.T @ rows
            Xty[j] += rows.T @ labels_chunk[m, j]
            train_valid[j] += count

    betas = np.zeros((nt, nf + 1), dtype=np.float64)
    active = np.zeros(nt, dtype=np.bool_)
    for j, target_name in enumerate(known):
        if train_valid[j] < min_train_samples or val_valid[j] == 0:
            targets_metrics[target_name] = {"target_name": target_name, "train_rows": int(train_valid[j]), "val_rows": int(val_valid[j]), "skipped": True, "skip_reason": "not_enough_train_rows" if train_valid[j] < min_train_samples else "no_validation_rows"}
            continue
        betas[j] = _solve(XtX[j] + reg, Xty[j])
        active[j] = True

    # Metrics pass shared across targets: one z-scored design per chunk,
    # one matmul for all predictions.
    acc = {target: {"train": {"n": 0, "se": 0.0, "ae": 0.0, "sum_y": 0.0, "sum_y2": 0.0, "sum_p": 0.0},
                    "val": {"n": 0, "se": 0.0, "ae": 0.0, "sum_y": 0.0, "sum_y2": 0.0, "sum_p": 0.0}}
           for target in known}
    is_binary = {target: target.endswith("_filled") or target.endswith("_toxic_fill") for target in known}
    exact_scores = {target: {"train": {"ys": [], "ps": []}, "val": {"ys": [], "ps": []}}
                    for target in known if metrics_mode == "exact" and is_binary[target]}
    range_acc = {target: {"train": BinaryHistogramAUC(auc_bins), "val": BinaryHistogramAUC(auc_bins)}
                 for target in known if metrics_mode == "approx" and is_binary[target]}

    def _metrics_chunks():
        for start in range(0, n, chunk_rows):
            end = min(start + chunk_rows, n)
            masks = np.asarray(dataset.arrays.label_masks[start:end][:, tidx], dtype=np.bool_)
            if not masks.any():
                continue
            X = np.asarray(dataset.arrays.features[start:end], dtype=np.float64)
            preds = (np.concatenate([np.ones((X.shape[0], 1), dtype=np.float64), (X - mean) / scale], axis=1) @ betas.T)
            labels_chunk = np.asarray(dataset.arrays.labels[start:end][:, tidx], dtype=np.float64)
            row_ids = np.arange(start, end)
            yield masks, preds, labels_chunk, row_ids < train_rows_total

    for masks, preds, labels_chunk, train_rows in _metrics_chunks():
        for j, target_name in enumerate(known):
            if not active[j]:
                continue
            col_mask = masks[:, j]
            if not col_mask.any():
                continue
            for split_name, split_mask in (("train", train_rows), ("val", ~train_rows)):
                m = col_mask & split_mask
                if not m.any():
                    continue
                y = labels_chunk[m, j]; p = preds[m, j]
                a = acc[target_name][split_name]
                a["n"] += len(y); a["se"] += float(np.sum((y - p) ** 2)); a["ae"] += float(np.sum(np.abs(y - p)))
                a["sum_y"] += float(np.sum(y)); a["sum_y2"] += float(np.sum(y * y)); a["sum_p"] += float(np.sum(p))
                if target_name in exact_scores:
                    if int(a["n"]) > exact_auc_max_rows:
                        raise ValueError("metrics_mode='exact' exceeds exact_auc_max_rows; use metrics_mode='approx' or increase exact_auc_max_rows")
                    exact_scores[target_name][split_name]["ys"].append(y.astype(np.float64, copy=True)); exact_scores[target_name][split_name]["ps"].append(p.astype(np.float64, copy=True))
                if target_name in range_acc:
                    range_acc[target_name][split_name].update_range(p)

    hist_acc: dict[str, dict[str, BinaryHistogramAUC]] = {}
    if range_acc:
        hist_acc = {
            target: {split: BinaryHistogramAUC(auc_bins, score_min=accum.score_min, score_max=accum.score_max) for split, accum in splits.items()}
            for target, splits in range_acc.items()
        }
        for masks, preds, labels_chunk, train_rows in _metrics_chunks():
            for j, target_name in enumerate(known):
                if not active[j] or target_name not in hist_acc:
                    continue
                col_mask = masks[:, j]
                if not col_mask.any():
                    continue
                for split_name, split_mask in (("train", train_rows), ("val", ~train_rows)):
                    m = col_mask & split_mask
                    if m.any():
                        hist_acc[target_name][split_name].update(labels_chunk[m, j], preds[m, j])

    for j, target_name in enumerate(known):
        if not active[j]:
            continue

        def split_metrics(name: str) -> tuple[float, float, float, float, float]:
            a = acc[target_name][name]; cnt = max(int(a["n"]), 1)
            mean_y = float(a["sum_y"] / cnt); ss_tot = float(a["sum_y2"] - a["sum_y"] * a["sum_y"] / cnt)
            r2 = 0.0 if ss_tot <= 1e-12 else 1.0 - float(a["se"] / ss_tot)
            return float(np.sqrt(a["se"] / cnt)), float(a["ae"] / cnt), r2, mean_y, float(a["sum_p"] / cnt)

        tr_rmse, tr_mae, tr_r2, tr_ymean, _ = split_metrics("train")
        va_rmse, va_mae, va_r2, va_ymean, va_pmean = split_metrics("val")
        metric: dict[str, object] = {"target_name": target_name, "train_rows": int(acc[target_name]["train"]["n"]), "val_rows": int(acc[target_name]["val"]["n"]), "train_rmse": tr_rmse, "val_rmse": va_rmse, "train_mae": tr_mae, "val_mae": va_mae, "train_r2": tr_r2, "val_r2": va_r2, "label_mean_train": tr_ymean, "label_mean_val": va_ymean, "prediction_mean_val": va_pmean, "skipped": False}
        if is_binary[target_name]:
            if metrics_mode == "none":
                metric["train_auc"] = None; metric["val_auc"] = None; metric["auc_mode"] = "none"
            elif metrics_mode == "exact":
                scores = exact_scores[target_name]
                metric["train_auc"] = _auc_exact(np.concatenate(scores["train"]["ys"]), np.concatenate(scores["train"]["ps"])) if scores["train"]["ys"] else None
                metric["val_auc"] = _auc_exact(np.concatenate(scores["val"]["ys"]), np.concatenate(scores["val"]["ps"])) if scores["val"]["ys"] else None
                metric["auc_mode"] = "exact"; metric["auc_bins"] = None
            else:
                metric["train_auc"] = hist_acc[target_name]["train"].auc(); metric["val_auc"] = hist_acc[target_name]["val"].auc()
                metric["auc_mode"] = "approx_histogram"; metric["auc_bins"] = auc_bins
        targets_metrics[target_name] = metric
        fitted_names.append(target_name); coefs.append(betas[j, 1:].astype(np.float64, copy=True)); intercepts.append(float(betas[j, 0]))

    metrics: dict[str, object] = {"enabled": True, "train_fraction": train_fraction, "ridge_l2": ridge_l2, "min_train_samples": min_train_samples, "train_rows_total": train_rows_total, "val_rows_total": val_rows_total, "fitted_target_count": len(fitted_names), "requested_target_count": len(target_names), "targets": targets_metrics}
    if not fitted_names:
        metrics["skipped"] = True; metrics["skip_reason"] = "all_targets_skipped"
    return AdverseBaselineFitResult(tuple(fitted_names), dataset.feature_names, train_rows_total, val_rows_total, mean.astype(np.float32), scale.astype(np.float32), np.vstack(coefs) if coefs else np.empty((0, nf), dtype=np.float64), np.asarray(intercepts, dtype=np.float64), metrics)
