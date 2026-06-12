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

    for target_name in target_names:
        if target_name not in label_index:
            targets_metrics[target_name] = {"target_name": target_name, "train_rows": 0, "val_rows": 0, "skipped": True, "skip_reason": "unknown_target"}
            continue
        tidx = label_index[target_name]
        XtX = np.zeros((nf + 1, nf + 1), dtype=np.float64); Xty = np.zeros(nf + 1, dtype=np.float64)
        train_valid = 0; val_valid = 0
        for start in range(0, n, chunk_rows):
            end = min(start + chunk_rows, n)
            mask = np.asarray(dataset.arrays.label_masks[start:end, tidx], dtype=np.bool_)
            row_ids = np.arange(start, end)
            train_mask = mask & (row_ids < train_rows_total)
            val_valid += int(np.count_nonzero(mask & (row_ids >= train_rows_total)))
            if not np.any(train_mask):
                continue
            X = np.asarray(dataset.arrays.features[start:end][train_mask], dtype=np.float64)
            Xz = (X - mean) / scale
            aug = np.concatenate([np.ones((Xz.shape[0], 1), dtype=np.float64), Xz], axis=1)
            y = np.asarray(dataset.arrays.labels[start:end, tidx][train_mask], dtype=np.float64)
            XtX += aug.T @ aug; Xty += aug.T @ y; train_valid += len(y)
        if train_valid < min_train_samples or val_valid == 0:
            targets_metrics[target_name] = {"target_name": target_name, "train_rows": train_valid, "val_rows": val_valid, "skipped": True, "skip_reason": "not_enough_train_rows" if train_valid < min_train_samples else "no_validation_rows"}
            continue
        beta = _solve(XtX + reg, Xty)
        intercept = float(beta[0]); coef = beta[1:].astype(np.float64, copy=False)
        # Metrics streaming accumulators.
        acc = {"train": {"n": 0, "se": 0.0, "ae": 0.0, "sum_y": 0.0, "sum_y2": 0.0, "sum_p": 0.0, "ys": [], "ps": []},
               "val": {"n": 0, "se": 0.0, "ae": 0.0, "sum_y": 0.0, "sum_y2": 0.0, "sum_p": 0.0, "ys": [], "ps": []}}
        for start in range(0, n, chunk_rows):
            end = min(start + chunk_rows, n)
            masks_all = np.asarray(dataset.arrays.label_masks[start:end, tidx], dtype=np.bool_)
            if not np.any(masks_all):
                continue
            Xall = np.asarray(dataset.arrays.features[start:end], dtype=np.float64)
            pred_all = ((Xall - mean) / scale) @ coef + intercept
            yall = np.asarray(dataset.arrays.labels[start:end, tidx], dtype=np.float64)
            row_ids = np.arange(start, end)
            for split_name, split_mask in (("train", row_ids < train_rows_total), ("val", row_ids >= train_rows_total)):
                m = masks_all & split_mask
                if not np.any(m):
                    continue
                y = yall[m]; p = pred_all[m]
                a = acc[split_name]
                a["n"] += len(y); a["se"] += float(np.sum((y - p) ** 2)); a["ae"] += float(np.sum(np.abs(y - p)))
                a["sum_y"] += float(np.sum(y)); a["sum_y2"] += float(np.sum(y * y)); a["sum_p"] += float(np.sum(p))
                if metrics_mode == "exact" or (metrics_mode == "approx" and (target_name.endswith("_filled") or target_name.endswith("_toxic_fill"))):
                    a["ys"].append(y.astype(np.float64)); a["ps"].append(p.astype(np.float64))
        def split_metrics(name: str) -> tuple[float, float, float, float, float]:
            a = acc[name]; cnt = max(int(a["n"]), 1)
            mean_y = float(a["sum_y"] / cnt); ss_tot = float(a["sum_y2"] - a["sum_y"] * a["sum_y"] / cnt)
            r2 = 0.0 if ss_tot <= 1e-12 else 1.0 - float(a["se"] / ss_tot)
            return float(np.sqrt(a["se"] / cnt)), float(a["ae"] / cnt), r2, mean_y, float(a["sum_p"] / cnt)
        tr_rmse, tr_mae, tr_r2, tr_ymean, _ = split_metrics("train")
        va_rmse, va_mae, va_r2, va_ymean, va_pmean = split_metrics("val")
        metric: dict[str, object] = {"target_name": target_name, "train_rows": int(acc["train"]["n"]), "val_rows": int(acc["val"]["n"]), "train_rmse": tr_rmse, "val_rmse": va_rmse, "train_mae": tr_mae, "val_mae": va_mae, "train_r2": tr_r2, "val_r2": va_r2, "label_mean_train": tr_ymean, "label_mean_val": va_ymean, "prediction_mean_val": va_pmean, "skipped": False}
        if target_name.endswith("_filled") or target_name.endswith("_toxic_fill"):
            if metrics_mode == "none":
                metric["train_auc"] = None; metric["val_auc"] = None; metric["auc_mode"] = "none"
            else:
                metric["train_auc"] = _auc_exact(np.concatenate(acc["train"]["ys"]), np.concatenate(acc["train"]["ps"])) if acc["train"]["ys"] else None
                metric["val_auc"] = _auc_exact(np.concatenate(acc["val"]["ys"]), np.concatenate(acc["val"]["ps"])) if acc["val"]["ys"] else None
                metric["auc_mode"] = "exact" if metrics_mode == "exact" else "approx_histogram"
                metric["auc_bins"] = auc_bins
        targets_metrics[target_name] = metric
        fitted_names.append(target_name); coefs.append(coef); intercepts.append(intercept)

    metrics: dict[str, object] = {"enabled": True, "train_fraction": train_fraction, "ridge_l2": ridge_l2, "min_train_samples": min_train_samples, "train_rows_total": train_rows_total, "val_rows_total": val_rows_total, "fitted_target_count": len(fitted_names), "requested_target_count": len(target_names), "targets": targets_metrics}
    if not fitted_names:
        metrics["skipped"] = True; metrics["skip_reason"] = "all_targets_skipped"
    return AdverseBaselineFitResult(tuple(fitted_names), dataset.feature_names, train_rows_total, val_rows_total, mean.astype(np.float32), scale.astype(np.float32), np.vstack(coefs) if coefs else np.empty((0, nf), dtype=np.float64), np.asarray(intercepts, dtype=np.float64), metrics)
