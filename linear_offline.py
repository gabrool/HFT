#!/usr/bin/env python3
"""Stage 1 linear offline entrypoint using CMSSL-compatible eval machinery."""

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch

from CMSSL17 import (  # type: ignore
    LOOKBACK, WINDOW_MS, HORIZONS_MS,
    BATCH_SIZE,
    PRIMARY_METRIC, PRIMARY_METRIC_HORIZON_MS, PRIMARY_DIR_BAL_ACC_GUARD,
    LOW_ABS_TRIM_FRACTION, HIGH_ABS_TRIM_FRACTION,
    TARGET_TRANSFORM, TARGET_TASK, LABEL_TRIM_SCHEMA,
    MODEL_OUTPUT_SCHEMA,
    build_dataset_from_split,
    compute_primary_metric,
)
from CMSSL17_offline import (  # type: ignore
    require_supported_pipeline_splits,
    make_single_week_split_from_meta,
    validate_dataset_label_dim,
    validate_contract_meta,
    validate_loaded_label_array,
    compute_signed_raw_stats,
    build_signed_side_trim_masks_from_stats_np,
    compute_dir_class_weights_from_train_labels,
    compute_mag_init_targets_from_train_labels,
    load_stats_cache,
    cache_matches,
    save_stats_cache,
    CPUWindowBatchSource,
    make_train_band_eval_source,
    summarize_metrics,
    print_band_metrics_summary,
    save_band_metrics_jsonl,
    FAST_VAL_MAX_ROWS,
    BAND_DIAG,
    BAND_DIAG_TRAIN,
    BAND_DIAG_TRAIN_MAX_ROWS,
    BAND_DIAG_QUANTILES,
)
from CMSSL17_linear import (  # type: ignore
    LINEAR_CHECKPOINT_SCHEMA,
    LINEAR_MODEL_ARCH_SCHEMA,
    LinearConstantPriorModel,
    build_constant_priors_from_train_labels,
    linear_model_summary,
)


OUT_ROOT = os.environ.get("BYBIT_OUT_ROOT", "").strip()
LINEAR_OUT_DIR = os.environ.get("BYBIT_LINEAR_OUT_DIR", "").strip()
LINEAR_STAGE = os.environ.get("BYBIT_LINEAR_STAGE", "stage1").strip().lower()
LINEAR_DEVICE = os.environ.get("BYBIT_LINEAR_DEVICE", "cpu").strip().lower()
LINEAR_EVAL_BATCH_SIZE = int(os.environ.get("BYBIT_LINEAR_BATCH_SIZE", str(BATCH_SIZE)))
LINEAR_RUN_TEST = int(os.environ.get("BYBIT_LINEAR_RUN_TEST", "1")) == 1


def _resolve_device() -> torch.device:
    if LINEAR_STAGE != "stage1":
        raise ValueError(f"BYBIT_LINEAR_STAGE must be 'stage1' for this entrypoint, got {LINEAR_STAGE!r}")
    if LINEAR_DEVICE not in {"cpu", "cuda", "auto"}:
        raise ValueError("BYBIT_LINEAR_DEVICE must be one of: cpu, cuda, auto")
    if LINEAR_EVAL_BATCH_SIZE <= 0:
        raise ValueError(f"BYBIT_LINEAR_BATCH_SIZE must be > 0, got {LINEAR_EVAL_BATCH_SIZE}")
    if LINEAR_DEVICE == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if LINEAR_DEVICE == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("BYBIT_LINEAR_DEVICE=cuda was requested, but CUDA is not available")
        return torch.device("cuda:0")
    return torch.device("cpu")


def _validate_dataset_split(ds: Any, split_name: str, feature_dim_total: int) -> None:
    if feature_dim_total != int(ds.feature_dim_total):
        raise ValueError(
            f"Feature dimension mismatch for {split_name}: meta={feature_dim_total}, "
            f"dataset={int(ds.feature_dim_total)}"
        )
    if int(ds.lookback) != int(LOOKBACK):
        raise ValueError(f"LOOKBACK mismatch for {split_name}: config={LOOKBACK}, dataset={int(ds.lookback)}")
    if len(ds.stores) != 1:
        raise ValueError(f"{split_name} split must have exactly one store/week, got {len(ds.stores)}")
    if ds.week_ids.size and not np.all(ds.week_ids == 0):
        raise ValueError(f"{split_name} split week_ids must all be 0 for single-week protocol")
    if len(ds) > 0 and int(ds.row_idx.min()) < int(LOOKBACK - 1):
        raise ValueError(
            f"{split_name} split has rows without full history: "
            f"min_row_idx={int(ds.row_idx.min())}, lookback={LOOKBACK}"
        )


def _make_cache_meta(meta: Dict[str, Any], protocol: str, train_week_keys: list[str], train_split_entries: list[dict]) -> Dict[str, Any]:
    tr_start = int(min(entry["start"] for entry in train_split_entries))
    tr_end = int(max(entry["end"] for entry in train_split_entries))
    return {
        "feature_schema": meta.get("feature_schema"),
        "feature_transform": meta.get("feature_transform"),
        "feature_transform_policy": meta.get("feature_transform_policy"),
        "feature_transform_spec_hash": meta.get("feature_transform_spec_hash"),
        "feature_transform_warmup_rows": int(meta.get("feature_transform_warmup_rows", -1)),
        "feature_dim_core": int(meta.get("feature_dim_core", -1)),
        "feature_dim_total": int(meta.get("feature_dim_total", -1)),
        "feature_names_hash": meta.get("feature_names_hash"),
        "aux_dim": int(meta.get("aux_dim", -1)),
        "aux_transform": meta.get("aux_transform"),
        "label_trim_schema": LABEL_TRIM_SCHEMA,
        "low_abs_trim_fraction": float(LOW_ABS_TRIM_FRACTION),
        "high_abs_trim_fraction": float(HIGH_ABS_TRIM_FRACTION),
        "horizons_ms": [int(h) for h in HORIZONS_MS],
        "split_protocol": protocol,
        "train_week_keys": list(train_week_keys),
        "train_ts_start": tr_start,
        "train_ts_end": tr_end,
        "decision_time_basis": meta.get("decision_time_basis"),
        "trade_history_enabled": meta.get("trade_history_enabled"),
        "event_stream_mode": meta.get("event_stream_mode"),
        "target_transform": TARGET_TRANSFORM,
        "label_units": "signed_log_return_bps",
        "target_task": TARGET_TASK,
        "loss_weighting_schema": "dir_mag_signed_nonzero_side_trim_tempered_class_dir_plain_mag_q50_q85_ema_v1",
        "ranking_schema": "tie_aware_average_ranks_v1",
        "band_diag_quantiles": [float(x) for x in BAND_DIAG_QUANTILES],
        "linear_stage": "stage1",
    }


def _print_primary(tag: str, metrics: Dict[str, Any], primary_metric_value: float, primary_metric_label: str) -> None:
    print(
        f"[{tag}] rows={int(metrics.get('n_eval_rows', 0))} "
        f"primary_metric_name={primary_metric_label} value={primary_metric_value:.8g} "
        f"guard_dir_bal_acc={float(metrics.get('primary_dir_bal_acc', float('nan'))):.8g} "
        f"guard_passed={bool(metrics.get('primary_metric_guard_passed', False))}",
        flush=True,
    )


def main() -> None:
    out_root = Path(OUT_ROOT)
    linear_out_dir = Path(LINEAR_OUT_DIR) if LINEAR_OUT_DIR else out_root / "linear_stage1"
    linear_out_dir.mkdir(parents=True, exist_ok=True)
    device = _resolve_device()
    print(
        f"[linear-config] stage={LINEAR_STAGE} device={device} batch_size={LINEAR_EVAL_BATCH_SIZE} "
        f"run_test={int(LINEAR_RUN_TEST)} out_root={out_root} linear_out_dir={linear_out_dir}",
        flush=True,
    )

    meta = json.loads((out_root / "meta.json").read_text())
    validate_contract_meta(meta, "global meta.json")
    validate_dataset_label_dim(meta, "global meta.json")
    split_info = require_supported_pipeline_splits(meta, out_root)
    protocol = split_info["protocol"]
    splits = split_info["splits"]
    cmssl = splits["cmssl"]
    train_week_keys = list(cmssl["train"]["weeks"])
    cmssl_val = cmssl["val"]
    cmssl_test = cmssl.get("test")
    has_cmssl_test = cmssl_test is not None and bool(cmssl_test.get("weeks"))
    print(
        f"[split] protocol={protocol} cmssl.train={','.join(train_week_keys)} "
        f"cmssl.val={cmssl_val.get('weeks')} "
        f"cmssl.test={cmssl_test.get('weeks') if has_cmssl_test else '<missing>'}",
        flush=True,
    )

    train_split_entries = [
        make_single_week_split_from_meta(out_root=out_root, global_meta=meta, week_key=wk)
        for wk in train_week_keys
    ]
    ds_train_list = [build_dataset_from_split(str(out_root), entry) for entry in train_split_entries]
    ds_val = build_dataset_from_split(str(out_root), cmssl_val)
    ds_test = build_dataset_from_split(str(out_root), cmssl_test) if has_cmssl_test else None

    feature_dim_total = int(meta.get("feature_dim_total", 0))
    for i, ds_train in enumerate(ds_train_list):
        _validate_dataset_split(ds_train, f"train[{i}]/{train_week_keys[i]}", feature_dim_total)
    _validate_dataset_split(ds_val, "val", feature_dim_total)
    if ds_test is not None:
        _validate_dataset_split(ds_test, "test", feature_dim_total)

    y_train = np.concatenate([np.asarray(ds.y, dtype=np.float32) for ds in ds_train_list], axis=0)
    validate_loaded_label_array(y_train, "linear train labels")

    cache_path = linear_out_dir / "linear_signed_side_trim_stats_cache.npz"
    cache_meta = _make_cache_meta(meta, protocol, train_week_keys, train_split_entries)
    cached = load_stats_cache(cache_path)
    if cached and cache_matches(cached[1], cache_meta):
        stats = cached[0]
        print(f"[linear-train-stats] loaded_cache={cache_path}", flush=True)
    else:
        stats = compute_signed_raw_stats(y_train)
        save_stats_cache(cache_path, stats, cache_meta)
        print(f"[linear-train-stats] wrote_cache={cache_path}", flush=True)

    dir_pos_w, dir_neg_w = compute_dir_class_weights_from_train_labels(
        y_train,
        pos_lo=stats["pos_lo_raw_bps"],
        pos_hi=stats["pos_hi_raw_bps"],
        neg_lo=stats["neg_lo_abs_bps"],
        neg_hi=stats["neg_hi_abs_bps"],
    )
    mag_pos_init_sqrt, mag_neg_init_sqrt = compute_mag_init_targets_from_train_labels(
        y_train,
        pos_lo=stats["pos_lo_raw_bps"],
        pos_hi=stats["pos_hi_raw_bps"],
        neg_lo=stats["neg_lo_abs_bps"],
        neg_hi=stats["neg_hi_abs_bps"],
        pos_q50=stats["kept_pos_q50_abs_raw_bps"],
        neg_q50=stats["kept_neg_q50_abs_raw_bps"],
    )
    print(f"[linear-train-stats] dir_pos_w={dir_pos_w.tolist()} dir_neg_w={dir_neg_w.tolist()}", flush=True)
    print(
        f"[linear-prior-mag] pos_target_sqrt={mag_pos_init_sqrt.tolist()} "
        f"neg_target_sqrt={mag_neg_init_sqrt.tolist()}",
        flush=True,
    )

    train_keep_pos, train_keep_neg, train_keep_signed = build_signed_side_trim_masks_from_stats_np(y_train, stats)
    prior_info = build_constant_priors_from_train_labels(
        y_train=y_train,
        stats=stats,
        mag_up_sqrt_prior=mag_pos_init_sqrt,
        mag_down_sqrt_prior=mag_neg_init_sqrt,
        keep_pos=train_keep_pos,
        keep_neg=train_keep_neg,
        keep_signed=train_keep_signed,
    )
    print(
        f"[linear-prior] p_up={prior_info['p_up_prior'].tolist()} "
        f"dir_logit={prior_info['dir_logit_prior'].tolist()}",
        flush=True,
    )
    model = LinearConstantPriorModel(
        prior_info["dir_logit_prior"],
        prior_info["mag_up_sqrt_prior"],
        prior_info["mag_down_sqrt_prior"],
    ).to(device)
    model.eval()

    val_full_src = CPUWindowBatchSource(
        ds_val,
        device,
        LINEAR_EVAL_BATCH_SIZE,
        shuffle=False,
        drop_last=False,
        row_stride=1,
    )
    val_fast_src = val_full_src.make_evenly_spaced_subset(FAST_VAL_MAX_ROWS)

    train_band_metrics: Optional[Dict[str, Any]] = None
    if BAND_DIAG and BAND_DIAG_TRAIN:
        train_eval_row_stride = max(1, int(os.environ.get("BYBIT_LINEAR_TRAIN_EVAL_ROW_STRIDE", "1")))
        train_sources = [
            CPUWindowBatchSource(
                ds,
                device,
                LINEAR_EVAL_BATCH_SIZE,
                shuffle=False,
                drop_last=False,
                row_stride=train_eval_row_stride,
            )
            for ds in ds_train_list
        ]
        train_band_src = make_train_band_eval_source(train_sources, BAND_DIAG_TRAIN_MAX_ROWS)
        train_band_metrics = summarize_metrics(
            model,
            train_band_src,
            device,
            stats,
            amp_enabled=False,
            amp_dtype=torch.float32,
            primary_only=True,
            epoch=0,
            band_diag=True,
            split_name="linear_train_band",
        )
        if "band_metrics" in train_band_metrics:
            print_band_metrics_summary(train_band_metrics["band_metrics"], split_name="linear_train_band", epoch=0)
            save_band_metrics_jsonl(linear_out_dir, train_band_metrics["band_metrics"], epoch=0, split_name="linear_train_band")

    val_fast = summarize_metrics(
        model,
        val_fast_src,
        device,
        stats,
        amp_enabled=False,
        amp_dtype=torch.float32,
        primary_only=True,
        epoch=0,
        band_diag=BAND_DIAG,
        split_name="linear_val_fast",
    )
    primary_metric_value, primary_metric_label = compute_primary_metric(val_fast)
    _print_primary("linear_val_fast", val_fast, primary_metric_value, primary_metric_label)

    val_full = summarize_metrics(
        model,
        val_full_src,
        device,
        stats,
        amp_enabled=False,
        amp_dtype=torch.float32,
        primary_only=False,
        epoch=0,
        band_diag=BAND_DIAG,
        split_name="linear_val_full",
    )
    val_full_primary_value, val_full_primary_label = compute_primary_metric(val_full)
    _print_primary("linear_val_full", val_full, val_full_primary_value, val_full_primary_label)
    if BAND_DIAG and "band_metrics" in val_full:
        print_band_metrics_summary(val_full["band_metrics"], split_name="linear_val_full", epoch=0)
        save_band_metrics_jsonl(linear_out_dir, val_full["band_metrics"], epoch=0, split_name="linear_val_full")

    test_metrics: Optional[Dict[str, Any]] = None
    if LINEAR_RUN_TEST and ds_test is not None:
        test_src = CPUWindowBatchSource(
            ds_test,
            device,
            LINEAR_EVAL_BATCH_SIZE,
            shuffle=False,
            drop_last=False,
            row_stride=1,
        )
        test_metrics = summarize_metrics(
            model,
            test_src,
            device,
            stats,
            amp_enabled=False,
            amp_dtype=torch.float32,
            primary_only=False,
            epoch=0,
            band_diag=BAND_DIAG,
            split_name="linear_test",
        )
        test_primary_value, test_primary_label = compute_primary_metric(test_metrics)
        _print_primary("linear_test", test_metrics, test_primary_value, test_primary_label)
        if BAND_DIAG and "band_metrics" in test_metrics:
            print_band_metrics_summary(test_metrics["band_metrics"], split_name="linear_test", epoch=0)
            save_band_metrics_jsonl(linear_out_dir, test_metrics["band_metrics"], epoch=0, split_name="linear_test")

    metrics_payload = {
        "stage": "stage1",
        "status": "ok",
        "out_root": str(out_root),
        "linear_out_dir": str(linear_out_dir),
        "protocol": protocol,
        "train_week_keys": train_week_keys,
        "val_weeks": cmssl_val.get("weeks"),
        "test_weeks": cmssl_test.get("weeks") if has_cmssl_test else None,
        "feature_dim_total": feature_dim_total,
        "lookback": LOOKBACK,
        "horizons_ms": [int(h) for h in HORIZONS_MS],
        "target_task": TARGET_TASK,
        "target_transform": TARGET_TRANSFORM,
        "label_trim_schema": LABEL_TRIM_SCHEMA,
        "model_output_schema": MODEL_OUTPUT_SCHEMA,
        "linear_checkpoint_schema": LINEAR_CHECKPOINT_SCHEMA,
        "linear_model_arch_schema": LINEAR_MODEL_ARCH_SCHEMA,
        "prior": linear_model_summary(model),
        "primary_metric": {
            "name": primary_metric_label,
            "value": float(primary_metric_value),
            "guard_dir_bal_acc": float(val_fast.get("primary_dir_bal_acc", float("nan"))),
            "guard_passed": bool(val_fast.get("primary_metric_guard_passed", False)),
        },
        "val_fast_metrics": val_fast,
        "val_full_metrics": val_full,
        "test_metrics": test_metrics,
        "train_band_metrics": train_band_metrics,
    }
    metrics_path = linear_out_dir / "linear_stage1_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, allow_nan=True, indent=2)
    print(f"[linear_metrics] wrote {metrics_path}", flush=True)

    ckpt = {
        "state_dict": model.state_dict(),
        "args": {
            "linear_checkpoint_schema": LINEAR_CHECKPOINT_SCHEMA,
            "linear_model_arch_schema": LINEAR_MODEL_ARCH_SCHEMA,
            "model_output_schema": MODEL_OUTPUT_SCHEMA,
            "stage": "stage1",
            "feature_dim_total": feature_dim_total,
            "LOOKBACK": LOOKBACK,
            "WINDOW_MS": WINDOW_MS,
            "HORIZONS_MS": HORIZONS_MS,
            "target_task": TARGET_TASK,
            "target_transform": TARGET_TRANSFORM,
            "label_trim_schema": LABEL_TRIM_SCHEMA,
            "low_abs_trim_fraction": float(LOW_ABS_TRIM_FRACTION),
            "high_abs_trim_fraction": float(HIGH_ABS_TRIM_FRACTION),
            "primary_metric": PRIMARY_METRIC,
            "primary_metric_horizon_ms": PRIMARY_METRIC_HORIZON_MS,
            "primary_dir_bal_acc_guard": PRIMARY_DIR_BAL_ACC_GUARD,
            "split_protocol": protocol,
            "train_week_keys": train_week_keys,
        },
        "prior": linear_model_summary(model),
        "stats": stats,
        "val_fast_metrics": val_fast,
        "val_full_metrics": val_full,
    }
    ckpt_path = linear_out_dir / "linear_stage1_prior.pt"
    torch.save(ckpt, ckpt_path)
    print(f"[linear_ckpt] saved {ckpt_path}", flush=True)


if __name__ == "__main__":
    assert OUT_ROOT, "Set BYBIT_OUT_ROOT to the root created by offline_ingest.py"
    main()
