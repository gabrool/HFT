"""Baseline-only sweep / structured search for RL_exec market-making engine.

Protocol note: in v2, CMSSL test corresponds to week3 full. This sweep intentionally
evaluates RL week3 development splits only (`--eval-split val|test`), and week4 final
evaluation remains outside this sweep in the main pipeline.

Examples (RL week-3 sweep only; final evaluation stays in the main pipeline after a config/checkpoint is chosen):
    python baseline_sweep.py --out-root /path/to/out_root --ckpt-path /path/to/cmssl17_offline_best.pt \
        --device cuda --search-mode random --n-trials 40 --eval-split val --results-csv baseline_week3_val_sweep.csv

    python baseline_sweep.py --out-root /path/to/out_root --ckpt-path /path/to/cmssl17_offline_best.pt \
        --search-mode random --vary alpha_center_scale weights --anchor-spread-cap-bps 4.0 \
        --n-trials 24 --eval-split val

    python baseline_sweep.py --out-root /path/to/out_root --ckpt-path /path/to/cmssl17_offline_best.pt \
        --search-mode grid --vary vol_width_scale weights --anchor-base-half-spread-bps 0.25 \
        --anchor-spread-floor-bps 0.25 --anchor-spread-cap-bps 4.0 --eval-split val

    python baseline_sweep.py --out-root /path/to/out_root --ckpt-path /path/to/cmssl17_offline_best.pt \
        --search-mode one-factor --vary alpha_center_scale spread_cap_bps weights --anchor-base-half-spread-bps 0.25 \
        --anchor-alpha-center-scale 1.5 --anchor-spread-floor-bps 0.25 --anchor-spread-cap-bps 4.0 \
        --anchor-weight-preset blend_235 --eval-split val

    python baseline_sweep.py --out-root /path/to/out_root --ckpt-path /path/to/cmssl17_offline_best.pt \
        --search-mode one-factor --vary obs_spread_anchor_frac --anchor-obs-spread-anchor-frac 0.5 --eval-split val

    python baseline_sweep.py --out-root /path/to/out_root --ckpt-path /path/to/cmssl17_offline_best.pt \
        --search-mode grid --vary vol_width_scale obs_spread_anchor_frac --eval-split val
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import multiprocessing as mp
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np

import RL_exec

REMOVED_BASELINE_KNOBS = ("k_sigma", "vol_horizon_ms", "s_min_bps", "k_alpha", "k_inv")
REMOVED_BASELINE_ENV_VARS = {
    "k_sigma": "BYBIT_MM_K_SIGMA",
    "vol_horizon_ms": "BYBIT_MM_VOL_HORIZON_MS",
    "s_min_bps": "BYBIT_MM_S_MIN_BPS",
    "k_alpha": "BYBIT_MM_K_ALPHA",
    "k_inv": "BYBIT_MM_K_INV",
}

BASELINE_PARAM_ENV_MAP = {
    "base_half_spread_bps": "BYBIT_MM_BASE_HALF_SPREAD_BPS",
    "alpha_center_scale": "BYBIT_MM_ALPHA_CENTER_SCALE",
    "inventory_center_scale": "BYBIT_MM_INVENTORY_CENTER_SCALE",
    "vol_width_scale": "BYBIT_MM_VOL_WIDTH_SCALE",
    "uncertainty_width_scale": "BYBIT_MM_UNCERTAINTY_WIDTH_SCALE",
    "inventory_side_widen_scale": "BYBIT_MM_INVENTORY_SIDE_WIDEN_SCALE",
    "obs_spread_anchor_frac": "BYBIT_MM_OBS_SPREAD_ANCHOR_FRAC",
    "spread_floor_bps": "BYBIT_MM_SPREAD_FLOOR_BPS",
    "spread_cap_bps": "BYBIT_MM_SPREAD_CAP_BPS",
    "inv_ref_notional": "BYBIT_MM_INV_REF_NOTIONAL",
    "p250_weight": "BYBIT_MM_P250_WEIGHT",
    "p500_weight": "BYBIT_MM_P500_WEIGHT",
    "p1000_weight": "BYBIT_MM_P1000_WEIGHT",
}

DEFAULT_SEARCH_SPACE: Dict[str, Sequence[Any]] = {
    "base_half_spread_bps": [0.0, 0.25, 0.5, 1.0],
    "alpha_center_scale": [0.5, 1.0, 1.5, 2.0],
    "inventory_center_scale": [0.0, 0.5, 1.0, 2.0],
    "vol_width_scale": [0.25, 0.5, 1.0, 1.5, 2.0],
    "uncertainty_width_scale": [0.0, 0.25, 0.5, 1.0],
    "inventory_side_widen_scale": [0.0, 0.5, 1.0, 2.0],
    "obs_spread_anchor_frac": [0.5, 0.25, 0.1, 0.0],
    "spread_floor_bps": [0.0, 0.25, 0.5],
    "spread_cap_bps": [4.0, 6.0, 8.0, 10.0],
    "weights": [
        (0.0, 0.0, 1.0),
        (0.2, 0.3, 0.5),
        (0.3, 0.4, 0.3),
        (1.0, 0.0, 0.0),
    ],
    "inv_ref_notional": [1.0, 2.0, 5.0],
}

WEIGHT_PRESETS = {
    "long": DEFAULT_SEARCH_SPACE["weights"][0],
    "blend_235": DEFAULT_SEARCH_SPACE["weights"][1],
    "blend_343": DEFAULT_SEARCH_SPACE["weights"][2],
    "short": DEFAULT_SEARCH_SPACE["weights"][3],
}

TUNABLE_FACTORS = [
    "base_half_spread_bps",
    "alpha_center_scale",
    "inventory_center_scale",
    "vol_width_scale",
    "uncertainty_width_scale",
    "inventory_side_widen_scale",
    "obs_spread_anchor_frac",
    "spread_floor_bps",
    "spread_cap_bps",
    "weights",
    "inv_ref_notional",
]

RESULT_COLUMNS = [
    "timestamp_utc",
    "status",
    "trial",
    "seed",
    "search_mode",
    "eval_split",
    "baseline_eval_split",
    "run_mode",
    "score",
    "anchor_config_json",
    "varied_factors",
    "changed_factors",
    "grid_index",
    "factor_name",
    "factor_value_label",
    "base_half_spread_bps",
    "alpha_center_scale",
    "inventory_center_scale",
    "vol_width_scale",
    "uncertainty_width_scale",
    "inventory_side_widen_scale",
    "obs_spread_anchor_frac",
    "spread_floor_bps",
    "spread_cap_bps",
    "inv_ref_notional",
    "p250_weight",
    "p500_weight",
    "p1000_weight",
    "vol_bucket_edges_bps_json",
    "vol_bucket_report_json",
    "net_pnl_pct",
    "sharpe_1h",
    "sortino_1h",
    "max_dd",
    "turnover_notional",
    "turnover_qty",
    "maker_fill_rate",
    "maker_fill_count",
    "maker_opportunities",
    "maker_buy_fills",
    "maker_sell_fills",
    "inventory_mean_abs_notional",
    "inventory_max_abs_notional",
    "cmssl_test",
    "error_type",
    "error_message",
]

SCALAR_ANCHOR_ARGS = {
    "base_half_spread_bps": "anchor_base_half_spread_bps",
    "alpha_center_scale": "anchor_alpha_center_scale",
    "inventory_center_scale": "anchor_inventory_center_scale",
    "vol_width_scale": "anchor_vol_width_scale",
    "uncertainty_width_scale": "anchor_uncertainty_width_scale",
    "inventory_side_widen_scale": "anchor_inventory_side_widen_scale",
    "obs_spread_anchor_frac": "anchor_obs_spread_anchor_frac",
    "spread_floor_bps": "anchor_spread_floor_bps",
    "spread_cap_bps": "anchor_spread_cap_bps",
    "inv_ref_notional": "anchor_inv_ref_notional",
}
REMOVED_BASELINE_CLI_OPTIONS = (
    "--anchor-s-min-bps",
    "--anchor-k-alpha",
    "--anchor-k-inv",
)

WEIGHT_COMPONENT_KEYS = ("p250_weight", "p500_weight", "p1000_weight")
WEIGHT_ARG_KEYS = {
    "p250_weight": "anchor_p250_weight",
    "p500_weight": "anchor_p500_weight",
    "p1000_weight": "anchor_p1000_weight",
}

_WORKER_PREPARED_CONTEXT: Optional[RL_exec.PreparedBaselineContext] = None
_WORKER_EVAL_SPLIT: Optional[str] = None
_WORKER_FAST_MODE: bool = True
_WORKER_USE_NUMBA: Optional[bool] = None


def _raise_on_stale_deleted_knobs(payload: Any, *, context: str) -> None:
    stale_tokens = set(REMOVED_BASELINE_KNOBS) | set(REMOVED_BASELINE_ENV_VARS.values()) | set(REMOVED_BASELINE_CLI_OPTIONS)

    def _walk(value: Any, path: str) -> None:
        if isinstance(value, dict):
            for key, nested in value.items():
                key_text = str(key)
                if key_text in stale_tokens:
                    raise ValueError(f"Stale deleted baseline knob '{key_text}' found in {context} at {path}.")
                _walk(nested, f"{path}.{key_text}")
            return
        if isinstance(value, (list, tuple, set)):
            for idx, nested in enumerate(value):
                _walk(nested, f"{path}[{idx}]")
            return
        if isinstance(value, str):
            for token in stale_tokens:
                if token in value:
                    raise ValueError(f"Stale deleted baseline knob '{token}' found in {context} at {path}.")

    _walk(payload, context)


def _validate_module_examples() -> None:
    module_doc = __doc__ or ""
    _raise_on_stale_deleted_knobs(module_doc, context="baseline_sweep module doc")


def resolve_required_path(value: Optional[str], env_name: str) -> str:
    resolved = (value or os.environ.get(env_name, "")).strip()
    if not resolved:
        raise SystemExit(f"Provide --{env_name.lower().replace('_', '-')} or set {env_name}.")
    return resolved


def normalize_baseline_config(config: Dict[str, Any]) -> Dict[str, Any]:
    cfg_obj = RL_exec.resolve_baseline_quote_config_from_mapping(config)
    return RL_exec.baseline_quote_config_to_dict(cfg_obj)


def build_default_anchor_config() -> Dict[str, Any]:
    weights = DEFAULT_SEARCH_SPACE["weights"][0]
    config = {
        "base_half_spread_bps": float(DEFAULT_SEARCH_SPACE["base_half_spread_bps"][0]),
        "alpha_center_scale": float(DEFAULT_SEARCH_SPACE["alpha_center_scale"][0]),
        "inventory_center_scale": float(DEFAULT_SEARCH_SPACE["inventory_center_scale"][0]),
        "vol_width_scale": float(DEFAULT_SEARCH_SPACE["vol_width_scale"][0]),
        "uncertainty_width_scale": float(DEFAULT_SEARCH_SPACE["uncertainty_width_scale"][0]),
        "inventory_side_widen_scale": float(DEFAULT_SEARCH_SPACE["inventory_side_widen_scale"][0]),
        "obs_spread_anchor_frac": float(DEFAULT_SEARCH_SPACE["obs_spread_anchor_frac"][0]),
        "spread_floor_bps": float(DEFAULT_SEARCH_SPACE["spread_floor_bps"][0]),
        "spread_cap_bps": float(DEFAULT_SEARCH_SPACE["spread_cap_bps"][0]),
        "inv_ref_notional": float(DEFAULT_SEARCH_SPACE["inv_ref_notional"][0]),
        "p250_weight": float(weights[0]),
        "p500_weight": float(weights[1]),
        "p1000_weight": float(weights[2]),
    }
    return config


def weight_tuple_from_config(config: Dict[str, Any]) -> tuple[float, float, float]:
    return tuple(float(config[key]) for key in WEIGHT_COMPONENT_KEYS)


def apply_weight_tuple(config: Dict[str, Any], weights: Sequence[Any]) -> None:
    p250_weight, p500_weight, p1000_weight = weights
    config["p250_weight"] = float(p250_weight)
    config["p500_weight"] = float(p500_weight)
    config["p1000_weight"] = float(p1000_weight)


def resolve_anchor_config(args: argparse.Namespace) -> Dict[str, Any]:
    config = build_default_anchor_config()
    if args.anchor_weight_preset is not None:
        apply_weight_tuple(config, WEIGHT_PRESETS[args.anchor_weight_preset])

    explicit_weight_values = {key: getattr(args, arg_name) for key, arg_name in WEIGHT_ARG_KEYS.items()}
    provided_weight_keys = [key for key, value in explicit_weight_values.items() if value is not None]
    if provided_weight_keys and len(provided_weight_keys) != len(WEIGHT_COMPONENT_KEYS):
        raise ValueError(
            "Explicit anchor weight overrides require --anchor-p250-weight, --anchor-p500-weight, "
            "and --anchor-p1000-weight together."
        )

    if len(provided_weight_keys) == len(WEIGHT_COMPONENT_KEYS):
        apply_weight_tuple(config, [explicit_weight_values[key] for key in WEIGHT_COMPONENT_KEYS])

    for config_key, arg_name in SCALAR_ANCHOR_ARGS.items():
        value = getattr(args, arg_name)
        if value is None:
            continue
        config[config_key] = float(value)

    return normalize_baseline_config(config)


def validate_vary_factors(factors: Sequence[str]) -> List[str]:
    cleaned: List[str] = []
    seen = set()
    unknown = [factor for factor in factors if factor not in TUNABLE_FACTORS]
    if unknown:
        raise ValueError(f"Unknown vary factors: {unknown}. Expected subset of {TUNABLE_FACTORS}.")
    for factor in factors:
        if factor in seen:
            continue
        cleaned.append(factor)
        seen.add(factor)
    return cleaned


def resolve_vary_factors(args: argparse.Namespace) -> List[str]:
    if args.vary:
        factors = validate_vary_factors(args.vary)
    elif args.search_mode in {"random", "one-factor"}:
        factors = list(TUNABLE_FACTORS)
    else:
        raise ValueError("--vary is required when --search-mode=grid.")
    return factors


def factor_candidates(space: Dict[str, Sequence[Any]], factor: str) -> Sequence[Any]:
    return space["weights"] if factor == "weights" else space[factor]


def apply_factor_value(config: Dict[str, Any], factor: str, value: Any) -> None:
    if factor == "weights":
        apply_weight_tuple(config, value)
    else:
        config[factor] = float(value)


def factor_value_equals_anchor(anchor_config: Dict[str, Any], factor: str, candidate: Any) -> bool:
    if factor == "weights":
        return tuple(candidate) == weight_tuple_from_config(anchor_config)
    return anchor_config[factor] == candidate


def factor_value_label(factor: str, value: Any) -> str:
    if factor == "weights":
        return "weights=" + json.dumps([float(v) for v in value])
    return f"{factor}={float(value):g}"


def diff_config_vs_anchor(config: Dict[str, Any], anchor_config: Dict[str, Any]) -> List[str]:
    changed: List[str] = []
    for factor in TUNABLE_FACTORS:
        if factor == "weights":
            if weight_tuple_from_config(config) != weight_tuple_from_config(anchor_config):
                changed.append("weights")
        elif config[factor] != anchor_config[factor]:
            changed.append(factor)
    return changed


def build_trial_descriptor(
    config: Dict[str, Any],
    *,
    search_mode: str,
    varied_factors: Sequence[str],
    anchor_config: Dict[str, Any],
    grid_index: Optional[int] = None,
    factor_name: Optional[str] = None,
    factor_value: Optional[Any] = None,
) -> Dict[str, Any]:
    return {
        "config": config,
        "search_mode": search_mode,
        "varied_factors": list(varied_factors),
        "changed_factors": diff_config_vs_anchor(config, anchor_config),
        "grid_index": grid_index,
        "factor_name": factor_name,
        "factor_value_label": factor_value_label(factor_name, factor_value) if factor_name is not None else None,
    }


def sample_baseline_config(rng: np.random.Generator, space: Dict[str, Sequence[Any]]) -> Dict[str, Any]:
    sampled = generate_random_configs(
        rng,
        space=space,
        anchor_config=build_default_anchor_config(),
        vary_factors=TUNABLE_FACTORS,
        n_trials=1,
    )[0]["config"]
    return normalize_baseline_config(sampled)


def generate_random_configs(
    rng: np.random.Generator,
    *,
    space: Dict[str, Sequence[Any]],
    anchor_config: Dict[str, Any],
    vary_factors: Sequence[str],
    n_trials: int,
) -> List[Dict[str, Any]]:
    plan: List[Dict[str, Any]] = []
    for _ in range(n_trials):
        config = dict(anchor_config)
        for factor in vary_factors:
            candidates = factor_candidates(space, factor)
            chosen = candidates[int(rng.integers(len(candidates)))]
            apply_factor_value(config, factor, chosen)
        plan.append(
            build_trial_descriptor(
                config,
                search_mode="random",
                varied_factors=vary_factors,
                anchor_config=anchor_config,
            )
        )
    return plan


def grid_size_for_factors(space: Dict[str, Sequence[Any]], vary_factors: Sequence[str]) -> int:
    size = 1
    for factor in vary_factors:
        size *= len(factor_candidates(space, factor))
    return size


def generate_grid_configs(
    *,
    space: Dict[str, Sequence[Any]],
    anchor_config: Dict[str, Any],
    vary_factors: Sequence[str],
) -> List[Dict[str, Any]]:
    plan: List[Dict[str, Any]] = []
    candidate_lists = [factor_candidates(space, factor) for factor in vary_factors]
    for grid_index, values in enumerate(itertools.product(*candidate_lists)):
        config = dict(anchor_config)
        for factor, value in zip(vary_factors, values):
            apply_factor_value(config, factor, value)
        plan.append(
            build_trial_descriptor(
                config,
                search_mode="grid",
                varied_factors=vary_factors,
                anchor_config=anchor_config,
                grid_index=grid_index,
            )
        )
    return plan


def generate_one_factor_configs(
    *,
    space: Dict[str, Sequence[Any]],
    anchor_config: Dict[str, Any],
    vary_factors: Sequence[str],
    include_anchor: bool = True,
) -> List[Dict[str, Any]]:
    plan: List[Dict[str, Any]] = []
    if include_anchor:
        plan.append(
            build_trial_descriptor(
                dict(anchor_config),
                search_mode="one-factor",
                varied_factors=vary_factors,
                anchor_config=anchor_config,
                factor_name=None,
            )
        )
    for factor in vary_factors:
        for candidate in factor_candidates(space, factor):
            if factor_value_equals_anchor(anchor_config, factor, candidate):
                continue
            config = dict(anchor_config)
            apply_factor_value(config, factor, candidate)
            plan.append(
                build_trial_descriptor(
                    config,
                    search_mode="one-factor",
                    varied_factors=vary_factors,
                    anchor_config=anchor_config,
                    factor_name=factor,
                    factor_value=candidate,
                )
            )
    return plan


def generate_trial_plan(
    args: argparse.Namespace,
    *,
    rng: np.random.Generator,
    space: Dict[str, Sequence[Any]],
    anchor_config: Dict[str, Any],
) -> List[Dict[str, Any]]:
    vary_factors = resolve_vary_factors(args)
    if args.search_mode == "random":
        return generate_random_configs(
            rng,
            space=space,
            anchor_config=anchor_config,
            vary_factors=vary_factors,
            n_trials=args.n_trials,
        )
    if args.search_mode == "grid":
        grid_size = grid_size_for_factors(space, vary_factors)
        if args.max_grid_trials is not None and grid_size > args.max_grid_trials:
            raise ValueError(
                f"Planned grid has {grid_size} trials, which exceeds --max-grid-trials={args.max_grid_trials}."
            )
        return generate_grid_configs(space=space, anchor_config=anchor_config, vary_factors=vary_factors)
    if args.search_mode == "one-factor":
        return generate_one_factor_configs(
            space=space,
            anchor_config=anchor_config,
            vary_factors=vary_factors,
            include_anchor=args.include_anchor,
        )
    raise ValueError(f"Unsupported search mode: {args.search_mode}")


def initialize_results_files(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists() or path.stat().st_size == 0:
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=RESULT_COLUMNS, extrasaction="ignore")
            writer.writeheader()
    jsonl_path = path.with_suffix(".jsonl")
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    jsonl_path.touch(exist_ok=True)
    return jsonl_path


def append_row_csv(path: Path, row: Dict[str, Any]) -> None:
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_COLUMNS, extrasaction="ignore")
        writer.writerow({column: row.get(column) for column in RESULT_COLUMNS})


def append_row_jsonl(path: Path, row: Dict[str, Any]) -> None:
    _raise_on_stale_deleted_knobs(row, context="serialized results row")
    with path.open("a") as handle:
        handle.write(json.dumps(row, sort_keys=True, default=str))
        handle.write("\n")


def _init_worker(prepared_context: RL_exec.PreparedBaselineContext, eval_split: str, fast_mode: bool, use_numba: Optional[bool]) -> None:
    global _WORKER_PREPARED_CONTEXT, _WORKER_EVAL_SPLIT, _WORKER_FAST_MODE, _WORKER_USE_NUMBA
    _WORKER_PREPARED_CONTEXT = prepared_context
    _WORKER_EVAL_SPLIT = eval_split
    _WORKER_FAST_MODE = fast_mode
    _WORKER_USE_NUMBA = use_numba


def _safe_metric(metrics: Dict[str, Any], key: str) -> Any:
    value = metrics.get(key)
    if isinstance(value, (np.generic,)):
        return value.item()
    return value


def flatten_report_row(
    config: Dict[str, Any],
    report: Dict[str, Any],
    *,
    trial: int,
    seed: int,
    eval_split: str,
    search_mode: str,
    anchor_config: Dict[str, Any],
    varied_factors: Sequence[str],
    changed_factors: Sequence[str],
    grid_index: Optional[int],
    factor_name: Optional[str],
    factor_value_label_text: Optional[str],
) -> Dict[str, Any]:
    baseline = report.get("mm_baseline") or {}
    run_context = report.get("mm_run_context") or {}
    row: Dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "status": "ok",
        "trial": trial,
        "seed": seed,
        "search_mode": search_mode,
        "eval_split": eval_split,
        "baseline_eval_split": run_context.get("baseline_eval_split", eval_split),
        "run_mode": run_context.get("run_mode"),
        "score": None,
        "anchor_config_json": json.dumps(anchor_config, sort_keys=True),
        "varied_factors": json.dumps(list(varied_factors)),
        "changed_factors": json.dumps(list(changed_factors)),
        "grid_index": grid_index,
        "factor_name": factor_name,
        "factor_value_label": factor_value_label_text,
        "cmssl_test": json.dumps(report.get("cmssl_test"), sort_keys=True),
        "vol_bucket_edges_bps_json": None,
        "vol_bucket_report_json": None,
        "error_type": None,
        "error_message": None,
    }
    for key in BASELINE_PARAM_ENV_MAP:
        row[key] = config.get(key)
    for metric_key in (
        "net_pnl_pct",
        "sharpe_1h",
        "sortino_1h",
        "max_dd",
        "turnover_notional",
        "turnover_qty",
        "maker_fill_rate",
        "maker_fill_count",
        "maker_opportunities",
        "maker_buy_fills",
        "maker_sell_fills",
        "inventory_mean_abs_notional",
        "inventory_max_abs_notional",
    ):
        if metric_key == "max_dd":
            row[metric_key] = _safe_metric(baseline, "max_dd")
            if row[metric_key] is None:
                row[metric_key] = _safe_metric(baseline, "max_drawdown")
        else:
            row[metric_key] = _safe_metric(baseline, metric_key)
    if "vol_bucket_edges_bps" in baseline:
        row["vol_bucket_edges_bps_json"] = json.dumps(baseline.get("vol_bucket_edges_bps"), sort_keys=True)
    if "vol_bucket_report" in baseline:
        row["vol_bucket_report_json"] = json.dumps(baseline.get("vol_bucket_report"), sort_keys=True)
    _raise_on_stale_deleted_knobs(row, context="results row")
    return row


def score_baseline_row(
    row: Dict[str, Any],
    *,
    min_fill_rate: float,
    max_drawdown: Optional[float],
) -> float:
    fill_rate = row.get("maker_fill_rate")
    if fill_rate is None or not np.isfinite(fill_rate) or fill_rate < min_fill_rate:
        return -1e18
    drawdown = row.get("max_dd")
    if max_drawdown is not None:
        if drawdown is None or not np.isfinite(drawdown) or drawdown > max_drawdown:
            return -1e18
    net_pnl_pct = row.get("net_pnl_pct")
    sharpe_1h = row.get("sharpe_1h")
    if net_pnl_pct is None or not np.isfinite(net_pnl_pct):
        net_pnl_pct = -1e9
    if sharpe_1h is None or not np.isfinite(sharpe_1h):
        sharpe_1h = -1e6
    if drawdown is None or not np.isfinite(drawdown):
        drawdown = 1e6
    return 1000.0 * float(net_pnl_pct)


def evaluate_baseline_config(
    config: Dict[str, Any],
    *,
    prepared_context: RL_exec.PreparedBaselineContext,
    eval_split: str,
    trial: int,
    seed: int,
    search_mode: str,
    anchor_config: Dict[str, Any],
    varied_factors: Sequence[str],
    changed_factors: Sequence[str],
    grid_index: Optional[int],
    factor_name: Optional[str],
    factor_value_label_text: Optional[str],
    fast_mode: bool,
    use_numba: Optional[bool],
) -> Dict[str, Any]:
    normalized_config = normalize_baseline_config(config)
    report = RL_exec.evaluate_prepared_baseline(
        prepared_context,
        eval_split=eval_split,
        baseline_config=normalized_config,
        fast_mode=fast_mode,
        use_numba=use_numba,
    )
    return flatten_report_row(
        normalized_config,
        report,
        trial=trial,
        seed=seed,
        eval_split=eval_split,
        search_mode=search_mode,
        anchor_config=anchor_config,
        varied_factors=varied_factors,
        changed_factors=changed_factors,
        grid_index=grid_index,
        factor_name=factor_name,
        factor_value_label_text=factor_value_label_text,
    )


def _execute_trial_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    prepared_context = _WORKER_PREPARED_CONTEXT
    eval_split = _WORKER_EVAL_SPLIT
    if prepared_context is None or eval_split is None:
        raise RuntimeError("Worker prepared context not initialized")
    config = payload["config"]
    try:
        row = evaluate_baseline_config(
            config,
            prepared_context=prepared_context,
            eval_split=eval_split,
            trial=payload["trial"],
            seed=payload["seed"],
            search_mode=payload["search_mode"],
            anchor_config=payload["anchor_config"],
            varied_factors=payload["varied_factors"],
            changed_factors=payload["changed_factors"],
            grid_index=payload["grid_index"],
            factor_name=payload["factor_name"],
            factor_value_label_text=payload["factor_value_label"],
            fast_mode=_WORKER_FAST_MODE,
            use_numba=_WORKER_USE_NUMBA,
        )
    except Exception as exc:
        row = make_error_row(
            config,
            trial=payload["trial"],
            seed=payload["seed"],
            eval_split=eval_split,
            search_mode=payload["search_mode"],
            anchor_config=payload["anchor_config"],
            varied_factors=payload["varied_factors"],
            changed_factors=payload["changed_factors"],
            grid_index=payload["grid_index"],
            factor_name=payload["factor_name"],
            factor_value_label_text=payload["factor_value_label"],
            exc=exc,
        )
    return row


def make_error_row(
    config: Dict[str, Any],
    *,
    trial: int,
    seed: int,
    eval_split: str,
    search_mode: str,
    anchor_config: Dict[str, Any],
    varied_factors: Sequence[str],
    changed_factors: Sequence[str],
    grid_index: Optional[int],
    factor_name: Optional[str],
    factor_value_label_text: Optional[str],
    exc: Exception,
) -> Dict[str, Any]:
    row = {
        column: None for column in RESULT_COLUMNS
    }
    row.update(
        {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "status": "error",
            "trial": trial,
            "seed": seed,
            "search_mode": search_mode,
            "eval_split": eval_split,
            "baseline_eval_split": eval_split,
            "run_mode": "baseline",
            "score": -1e18,
            "anchor_config_json": json.dumps(anchor_config, sort_keys=True),
            "varied_factors": json.dumps(list(varied_factors)),
            "changed_factors": json.dumps(list(changed_factors)),
            "grid_index": grid_index,
            "factor_name": factor_name,
            "factor_value_label": factor_value_label_text,
            "error_type": type(exc).__name__,
            "error_message": str(exc),
        }
    )
    for key in BASELINE_PARAM_ENV_MAP:
        row[key] = config.get(key)
    _raise_on_stale_deleted_knobs(row, context="error results row")
    return row


def describe_eval_split(split: str) -> str:
    if split == "val":
        return "val (RL week-3 validation/dev)"
    if split == "test":
        return "test (RL week-3 test/dev)"
    return split


def print_leaderboard(rows: List[Dict[str, Any]], *, top_k: int, label: str) -> None:
    ranked = [row for row in rows if row.get("status") == "ok"]
    ranked.sort(key=lambda item: item.get("score", -math.inf), reverse=True)
    print(f"[{label}] top {min(top_k, len(ranked))} configs")
    for idx, row in enumerate(ranked[:top_k], start=1):
        factor_text = f" factor={row.get('factor_name')}" if row.get("factor_name") else ""
        print(
            f"  #{idx} mode={row.get('search_mode')}{factor_text} score={row.get('score'):.6f} "
            f"split={describe_eval_split(row.get('baseline_eval_split'))} pnl={row.get('net_pnl_pct')} sharpe={row.get('sharpe_1h')} "
            f"dd={row.get('max_dd')} fill_rate={row.get('maker_fill_rate')} fills={row.get('maker_fill_count')} "
            f"params={{base_half_spread_bps={row.get('base_half_spread_bps')}, alpha_center_scale={row.get('alpha_center_scale')}, "
            f"inventory_center_scale={row.get('inventory_center_scale')}, vol_width_scale={row.get('vol_width_scale')}, "
            f"uncertainty_width_scale={row.get('uncertainty_width_scale')}, inventory_side_widen_scale={row.get('inventory_side_widen_scale')}, "
            f"obs_spread_anchor_frac={row.get('obs_spread_anchor_frac')}, spread_floor_bps={row.get('spread_floor_bps')}, "
            f"spread_cap_bps={row.get('spread_cap_bps')}, inv_ref_notional={row.get('inv_ref_notional')}, "
            f"weights=({row.get('p250_weight')}, {row.get('p500_weight')}, {row.get('p1000_weight')})}}"
        )


def build_metadata(
    args: argparse.Namespace,
    *,
    out_root: str,
    ckpt_path: str,
    prepared_context: RL_exec.PreparedBaselineContext,
    anchor_config: Dict[str, Any],
    vary_factors: Sequence[str],
    planned_trials: int,
) -> Dict[str, Any]:
    metadata = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "args": vars(args),
        "seed": args.seed,
        "out_root": out_root,
        "ckpt_path": ckpt_path,
        "prepared_context_reuse": True,
        "prepared_joined_rows": prepared_context.joined_rows,
        "prepared_val_rows": int(prepared_context.mm_val_batch.features.shape[0]),
        "prepared_test_rows": int(prepared_context.mm_test_batch.features.shape[0]),
        "prepared_cmssl_batch_size": prepared_context.cmssl_batch_size,
        "search_mode": args.search_mode,
        "anchor_config": anchor_config,
        "vary_factors": list(vary_factors),
        "planned_trials": planned_trials,
        "n_trials_ignored": args.search_mode in {"grid", "one-factor"},
        "fast_baseline_mode": not args.disable_fast_baseline,
        "fast_baseline_numba": not args.disable_numba,
        "workers": args.workers,
        "worker_chunk_size": args.worker_chunk_size,
        "start_method": args.start_method,
        "append_results_mode": True,
    }
    if args.search_mode == "grid":
        metadata["grid_size"] = planned_trials
    if args.search_mode == "one-factor":
        metadata["one_factor_include_anchor"] = args.include_anchor
    _raise_on_stale_deleted_knobs(metadata, context="metadata")
    return metadata


def write_metadata(path: Path, payload: Dict[str, Any]) -> None:
    _raise_on_stale_deleted_knobs(payload, context="serialized metadata payload")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))


def parse_args() -> argparse.Namespace:
    for raw_arg in os.sys.argv[1:]:
        option = raw_arg.split("=", 1)[0]
        for knob in REMOVED_BASELINE_KNOBS:
            deleted_opt = f"--{knob.replace('_', '-')}"
            if option == deleted_opt:
                raise SystemExit(
                    f"{deleted_opt} has been removed from baseline_sweep.py; delete stale references to {knob}."
                )
        for deleted_opt in REMOVED_BASELINE_CLI_OPTIONS:
            if option == deleted_opt:
                raise SystemExit(
                    f"{deleted_opt} has been removed from baseline_sweep.py; use the renamed anchor baseline quote controls instead."
                )

    parser = argparse.ArgumentParser(
        description=(
            "Baseline-only RL week-3 sweep / structured search for RL_exec. "
            "In v2 protocol, CMSSL test corresponds to week3 full; this sweep still uses only RL week-3 "
            "development splits (val/test), and week4 final evaluation stays in the main pipeline."
        )
    )
    parser.add_argument("--out-root", default=None)
    parser.add_argument("--ckpt-path", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--n-trials", type=int, default=40)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument(
        "--eval-split",
        choices=("val", "test"),
        default="val",
        help=(
            "Sweep split (unchanged): val = RL week-3 validation/dev; test = RL week-3 test/dev. "
            "CMSSL test maps to week3 full in v2 protocol, but this sweep remains RL week-3 val/test only."
        ),
    )
    parser.add_argument(
        "--results-csv",
        default="baseline_sweep_results.csv",
        help="Output CSV for RL week-3 val/test sweep results (not week4 final evaluation).",
    )
    parser.add_argument("--top-k", type=int, default=5, help="Number of top RL week-3 sweep candidates to summarize.")
    parser.add_argument(
        "--retest-topk-on-test",
        action="store_true",
        help=(
            "After ranking candidates on the chosen RL week-3 sweep split, retest top-k on RL week-3 test/dev. "
            "This is still within sweep development evaluation; week4 final evaluation remains in the main pipeline."
        ),
    )
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--search-mode", choices=("random", "grid", "one-factor"), default="random")
    parser.add_argument("--vary", nargs="+", default=None)
    parser.add_argument("--include-anchor", dest="include_anchor", action="store_true", default=True)
    parser.add_argument("--exclude-anchor", dest="include_anchor", action="store_false")
    parser.add_argument("--max-grid-trials", type=int, default=None)
    parser.add_argument("--anchor-base-half-spread-bps", type=float, default=None)
    parser.add_argument("--anchor-alpha-center-scale", type=float, default=None)
    parser.add_argument("--anchor-inventory-center-scale", type=float, default=None)
    parser.add_argument("--anchor-vol-width-scale", type=float, default=None)
    parser.add_argument("--anchor-uncertainty-width-scale", type=float, default=None)
    parser.add_argument("--anchor-inventory-side-widen-scale", type=float, default=None)
    parser.add_argument("--anchor-obs-spread-anchor-frac", type=float, default=None)
    parser.add_argument("--anchor-spread-floor-bps", type=float, default=None)
    parser.add_argument("--anchor-spread-cap-bps", type=float, default=None)
    parser.add_argument("--anchor-inv-ref-notional", type=float, default=None)
    parser.add_argument("--anchor-p250-weight", type=float, default=None)
    parser.add_argument("--anchor-p500-weight", type=float, default=None)
    parser.add_argument("--anchor-p1000-weight", type=float, default=None)
    parser.add_argument("--anchor-weight-preset", choices=tuple(WEIGHT_PRESETS), default=None)
    parser.add_argument("--min-fill-rate", type=float, default=0.002)
    parser.add_argument("--max-drawdown", type=float, default=None)
    parser.add_argument("--workers", type=int, default=1, help="CPU workers for RL week-3 sweep trials; start with 4-8 on large hosts.")
    parser.add_argument("--worker-chunk-size", type=int, default=1)
    parser.add_argument("--start-method", choices=("auto", "fork", "spawn"), default="auto")
    parser.add_argument("--disable-fast-baseline", action="store_true")
    parser.add_argument("--disable-numba", action="store_true")

    help_text = parser.format_help()
    _raise_on_stale_deleted_knobs(help_text, context="CLI help text")

    args = parser.parse_args()
    _raise_on_stale_deleted_knobs(vars(args), context="parsed CLI args")
    return args


def build_retest_row_metadata(row: Dict[str, Any], anchor_config: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "anchor_config": json.loads(row.get("anchor_config_json") or json.dumps(anchor_config, sort_keys=True)),
        "varied_factors": json.loads(row.get("varied_factors") or "[]"),
        "changed_factors": json.loads(row.get("changed_factors") or "[]"),
        "grid_index": row.get("grid_index"),
        "factor_name": row.get("factor_name"),
        "factor_value_label": row.get("factor_value_label"),
        "search_mode": row.get("search_mode") or "random",
    }


def main() -> None:
    _validate_module_examples()
    args = parse_args()
    out_root = resolve_required_path(args.out_root, "BYBIT_OUT_ROOT")
    ckpt_path = resolve_required_path(args.ckpt_path, "BYBIT_CMSSL_CKPT")

    # Imported usage skips RL_exec.__main__, so run the shared setup hooks here.
    RL_exec._set_seed_from_env()
    RL_exec._configure_tf32_from_env()

    rng = np.random.default_rng(args.seed)
    results_csv = Path(args.results_csv)
    results_jsonl = initialize_results_files(results_csv)
    metadata_json = results_csv.with_suffix(".metadata.json")

    rows: List[Dict[str, Any]] = []
    prepared_context = RL_exec.prepare_baseline_context(out_root, ckpt_path, device=args.device)
    fast_mode = not args.disable_fast_baseline
    use_numba = None if not args.disable_fast_baseline and not args.disable_numba else False
    anchor_config = resolve_anchor_config(args)
    vary_factors = resolve_vary_factors(args)
    trial_plan = generate_trial_plan(args, rng=rng, space=DEFAULT_SEARCH_SPACE, anchor_config=anchor_config)

    if args.search_mode == "grid":
        print(f"[info] mode=grid planned RL week-3 grid size={len(trial_plan)} on {describe_eval_split(args.eval_split)}; --n-trials={args.n_trials} is ignored.")
    elif args.search_mode == "one-factor":
        print(f"[info] mode=one-factor uses {len(trial_plan)} planned RL week-3 trials on {describe_eval_split(args.eval_split)}; --n-trials={args.n_trials} is ignored.")

    write_metadata(
        metadata_json,
        build_metadata(
            args,
            out_root=out_root,
            ckpt_path=ckpt_path,
            prepared_context=prepared_context,
            anchor_config=anchor_config,
            vary_factors=vary_factors,
            planned_trials=len(trial_plan),
        ),
    )

    total_trials = len(trial_plan)
    payloads = [
        {
            "trial": trial,
            "seed": args.seed,
            "config": plan_entry["config"],
            "search_mode": plan_entry["search_mode"],
            "anchor_config": anchor_config,
            "varied_factors": plan_entry["varied_factors"],
            "changed_factors": plan_entry["changed_factors"],
            "grid_index": plan_entry["grid_index"],
            "factor_name": plan_entry["factor_name"],
            "factor_value_label": plan_entry["factor_value_label"],
        }
        for trial, plan_entry in enumerate(trial_plan)
    ]
    start_method = args.start_method
    if start_method == "auto":
        start_method = "fork" if os.name == "posix" else "spawn"
    if args.workers <= 1:
        _init_worker(prepared_context, args.eval_split, fast_mode, use_numba)
        trial_iter: Iterable[Dict[str, Any]] = (_execute_trial_payload(payload) for payload in payloads)
    else:
        ctx = mp.get_context(start_method)
        pool = ctx.Pool(
            processes=args.workers,
            initializer=_init_worker,
            initargs=(prepared_context, args.eval_split, fast_mode, use_numba),
        )
        trial_iter = pool.imap(_execute_trial_payload, payloads, chunksize=max(1, args.worker_chunk_size))
    try:
        for row in trial_iter:
            row["score"] = score_baseline_row(
                row,
                min_fill_rate=args.min_fill_rate,
                max_drawdown=args.max_drawdown,
            )
            rows.append(row)
            append_row_csv(results_csv, row)
            append_row_jsonl(results_jsonl, row)
            if args.verbose:
                print(
                    f"[trial {row['trial'] + 1}/{total_trials}] status={row['status']} score={row.get('score')} "
                    f"split={describe_eval_split(row.get('baseline_eval_split'))} config={{base_half_spread_bps={row.get('base_half_spread_bps')}, alpha_center_scale={row.get('alpha_center_scale')}, vol_width_scale={row.get('vol_width_scale')}, obs_spread_anchor_frac={row.get('obs_spread_anchor_frac')}, spread_cap_bps={row.get('spread_cap_bps')}}}"
                )
    finally:
        if args.workers > 1:
            pool.close()
            pool.join()

    print_leaderboard(rows, top_k=args.top_k, label=f"baseline sweep {describe_eval_split(args.eval_split)}")

    best_rows = [row for row in rows if row.get("status") == "ok"]
    best_rows.sort(key=lambda item: item.get("score", -math.inf), reverse=True)
    if best_rows:
        best = best_rows[0]
        best_env = {
            BASELINE_PARAM_ENV_MAP[key]: str(best[key])
            for key in BASELINE_PARAM_ENV_MAP
        }
        _raise_on_stale_deleted_knobs(best_env, context="best config env payload")
        print(f"[best config env from {describe_eval_split(args.eval_split)}]", json.dumps(best_env, sort_keys=True))

    if args.retest_topk_on_test:
        print(f"[retest] retesting top {min(args.top_k, len(best_rows))} RL week-3 candidates from {describe_eval_split(args.eval_split)} on {describe_eval_split('test')}; no separate final evaluation week is run here.")
        top_rows = [row for row in best_rows[: args.top_k]]
        retest_rows: List[Dict[str, Any]] = []
        retest_csv = results_csv.with_name(f"{results_csv.stem}_topk_test.csv")
        retest_jsonl = initialize_results_files(retest_csv)
        for rank, row in enumerate(top_rows, start=1):
            config = {key: row[key] for key in BASELINE_PARAM_ENV_MAP}
            row_metadata = build_retest_row_metadata(row, anchor_config)
            try:
                retest_row = evaluate_baseline_config(
                    config,
                    prepared_context=prepared_context,
                    eval_split="test",
                    trial=rank,
                    seed=args.seed,
                    search_mode=row_metadata["search_mode"],
                    anchor_config=row_metadata["anchor_config"],
                    varied_factors=row_metadata["varied_factors"],
                    changed_factors=row_metadata["changed_factors"],
                    grid_index=row_metadata["grid_index"],
                    factor_name=row_metadata["factor_name"],
                    factor_value_label_text=row_metadata["factor_value_label"],
                    fast_mode=fast_mode,
                    use_numba=use_numba,
                )
                retest_row["score"] = score_baseline_row(
                    retest_row,
                    min_fill_rate=args.min_fill_rate,
                    max_drawdown=args.max_drawdown,
                )
            except Exception as exc:
                retest_row = make_error_row(
                    config,
                    trial=rank,
                    seed=args.seed,
                    eval_split="test",
                    search_mode=row_metadata["search_mode"],
                    anchor_config=row_metadata["anchor_config"],
                    varied_factors=row_metadata["varied_factors"],
                    changed_factors=row_metadata["changed_factors"],
                    grid_index=row_metadata["grid_index"],
                    factor_name=row_metadata["factor_name"],
                    factor_value_label_text=row_metadata["factor_value_label"],
                    exc=exc,
                )
                print(f"[retest rank {rank} on {describe_eval_split('test')}] error {type(exc).__name__}: {exc}")
            retest_rows.append(retest_row)
            append_row_csv(retest_csv, retest_row)
            append_row_jsonl(retest_jsonl, retest_row)
        print_leaderboard(retest_rows, top_k=args.top_k, label=f"baseline sweep retest on {describe_eval_split('test')}")


if __name__ == "__main__":
    main()
