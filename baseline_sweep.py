"""Baseline-only parameter sweep for RL_exec market-making engine.

Example:
    python baseline_sweep.py --out-root /path/to/out_root --ckpt-path /path/to/cmssl17_offline_best.pt \
        --device cuda --n-trials 40 --eval-split val --results-csv baseline_val_sweep.csv \
        --top-k 5 --retest-topk-on-test
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence

import numpy as np

import RL_exec

BASELINE_PARAM_ENV_MAP = {
    "s_min_bps": "BYBIT_MM_S_MIN_BPS",
    "k_sigma": "BYBIT_MM_K_SIGMA",
    "k_inv": "BYBIT_MM_K_INV",
    "k_alpha": "BYBIT_MM_K_ALPHA",
    "spread_floor_bps": "BYBIT_MM_SPREAD_FLOOR_BPS",
    "spread_cap_bps": "BYBIT_MM_SPREAD_CAP_BPS",
    "inv_ref_notional": "BYBIT_MM_INV_REF_NOTIONAL",
    "vol_horizon_ms": "BYBIT_MM_VOL_HORIZON_MS",
    "p250_weight": "BYBIT_MM_P250_WEIGHT",
    "p500_weight": "BYBIT_MM_P500_WEIGHT",
    "p1000_weight": "BYBIT_MM_P1000_WEIGHT",
}

DEFAULT_SEARCH_SPACE: Dict[str, Sequence[Any]] = {
    "s_min_bps": [0.0, 0.25, 0.5],
    "k_sigma": [0.05, 0.10, 0.15, 0.25, 0.40],
    "k_alpha": [0.5, 1.0, 1.5, 2.0, 3.0],
    "spread_floor_bps": [0.0, 0.25, 0.5],
    "spread_cap_bps": [2.0, 3.0, 4.0, 6.0, 8.0],
    "vol_horizon_ms": [500, 1000],
    "weights": [
        (0.0, 0.0, 1.0),
        (0.2, 0.3, 0.5),
        (0.3, 0.4, 0.3),
        (0.5, 0.5, 0.0),
    ],
    "k_inv": [0.0],
    "inv_ref_notional": [1.0],
}

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
    "s_min_bps",
    "k_sigma",
    "k_inv",
    "k_alpha",
    "spread_floor_bps",
    "spread_cap_bps",
    "inv_ref_notional",
    "vol_horizon_ms",
    "p250_weight",
    "p500_weight",
    "p1000_weight",
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


@contextmanager
def temporary_env(overrides: Dict[str, str]) -> Iterator[None]:
    previous = {key: os.environ.get(key) for key in overrides}
    try:
        for key, value in overrides.items():
            os.environ[key] = value
        yield
    finally:
        for key, old_value in previous.items():
            if old_value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = old_value


def resolve_required_path(value: Optional[str], env_name: str) -> str:
    resolved = (value or os.environ.get(env_name, "")).strip()
    if not resolved:
        raise SystemExit(f"Provide --{env_name.lower().replace('_', '-')} or set {env_name}.")
    return resolved


def validate_baseline_config(config: Dict[str, Any], *, tol: float = 1e-6) -> None:
    weight_sum = float(config["p250_weight"]) + float(config["p500_weight"]) + float(config["p1000_weight"])
    if abs(weight_sum - 1.0) > tol:
        raise ValueError(
            "Baseline horizon weights must sum to 1.0 within tolerance; "
            f"got {weight_sum:.12f} for {config}"
        )


def sample_baseline_config(rng: np.random.Generator, space: Dict[str, Sequence[Any]]) -> Dict[str, Any]:
    config = {
        "s_min_bps": float(rng.choice(space["s_min_bps"])),
        "k_sigma": float(rng.choice(space["k_sigma"])),
        "k_alpha": float(rng.choice(space["k_alpha"])),
        "spread_floor_bps": float(rng.choice(space["spread_floor_bps"])),
        "spread_cap_bps": float(rng.choice(space["spread_cap_bps"])),
        "vol_horizon_ms": int(rng.choice(space["vol_horizon_ms"])),
        "k_inv": float(rng.choice(space["k_inv"])),
        "inv_ref_notional": float(rng.choice(space["inv_ref_notional"])),
    }
    weights = rng.choice(len(space["weights"]))
    p250_weight, p500_weight, p1000_weight = space["weights"][int(weights)]
    config["p250_weight"] = float(p250_weight)
    config["p500_weight"] = float(p500_weight)
    config["p1000_weight"] = float(p1000_weight)
    validate_baseline_config(config)
    return config


def config_to_env_overrides(config: Dict[str, Any], eval_split: str) -> Dict[str, str]:
    validate_baseline_config(config)
    overrides = {
        "BYBIT_MM_RUN_MODE": "baseline",
        "BYBIT_MM_BASELINE_EVAL_SPLIT": eval_split,
    }
    for key, env_name in BASELINE_PARAM_ENV_MAP.items():
        overrides[env_name] = str(config[key])
    return overrides


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
        "cmssl_test": json.dumps(report.get("cmssl_test"), sort_keys=True),
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
        row[metric_key] = _safe_metric(baseline, metric_key)
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
) -> Dict[str, Any]:
    overrides = config_to_env_overrides(config, eval_split)
    with temporary_env(overrides):
        report = RL_exec.evaluate_prepared_baseline(
            prepared_context,
            eval_split=eval_split,
        )
    return flatten_report_row(
        config,
        report,
        trial=trial,
        seed=seed,
        eval_split=eval_split,
        search_mode=search_mode,
    )


def make_error_row(
    config: Dict[str, Any],
    *,
    trial: int,
    seed: int,
    eval_split: str,
    search_mode: str,
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
            "error_type": type(exc).__name__,
            "error_message": str(exc),
        }
    )
    for key in BASELINE_PARAM_ENV_MAP:
        row[key] = config.get(key)
    return row


def write_rows_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column) for column in RESULT_COLUMNS})


def write_rows_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, default=str))
            handle.write("\n")


def print_leaderboard(rows: List[Dict[str, Any]], *, top_k: int, label: str) -> None:
    ranked = [row for row in rows if row.get("status") == "ok"]
    ranked.sort(key=lambda item: item.get("score", -math.inf), reverse=True)
    print(f"[{label}] top {min(top_k, len(ranked))} configs")
    for idx, row in enumerate(ranked[:top_k], start=1):
        print(
            f"  #{idx} score={row.get('score'):.6f} split={row.get('baseline_eval_split')} "
            f"pnl={row.get('net_pnl_pct')} sharpe={row.get('sharpe_1h')} dd={row.get('max_dd')} "
            f"fill_rate={row.get('maker_fill_rate')} fills={row.get('maker_fill_count')} "
            f"params={{s_min_bps={row.get('s_min_bps')}, k_sigma={row.get('k_sigma')}, "
            f"k_alpha={row.get('k_alpha')}, spread_floor_bps={row.get('spread_floor_bps')}, "
            f"spread_cap_bps={row.get('spread_cap_bps')}, vol_horizon_ms={row.get('vol_horizon_ms')}, "
            f"weights=({row.get('p250_weight')}, {row.get('p500_weight')}, {row.get('p1000_weight')})}}"
        )


def build_metadata(
    args: argparse.Namespace,
    *,
    out_root: str,
    ckpt_path: str,
    prepared_context: RL_exec.PreparedBaselineContext,
) -> Dict[str, Any]:
    return {
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
    }


def write_metadata(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Random baseline-only sweep for RL_exec.")
    parser.add_argument("--out-root", default=None)
    parser.add_argument("--ckpt-path", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--n-trials", type=int, default=40)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--eval-split", choices=("val", "test"), default="val")
    parser.add_argument("--results-csv", default="baseline_sweep_results.csv")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--retest-topk-on-test", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--search-mode", choices=("random",), default="random")
    parser.add_argument("--min-fill-rate", type=float, default=0.002)
    parser.add_argument("--max-drawdown", type=float, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_root = resolve_required_path(args.out_root, "BYBIT_OUT_ROOT")
    ckpt_path = resolve_required_path(args.ckpt_path, "BYBIT_CMSSL_CKPT")
    if args.search_mode != "random":
        raise ValueError(f"Unsupported search mode: {args.search_mode}")

    # Imported usage skips RL_exec.__main__, so run the shared setup hooks here.
    RL_exec._set_seed_from_env()
    RL_exec._configure_tf32_from_env()

    rng = np.random.default_rng(args.seed)
    results_csv = Path(args.results_csv)
    results_jsonl = results_csv.with_suffix(".jsonl")
    metadata_json = results_csv.with_suffix(".metadata.json")

    rows: List[Dict[str, Any]] = []
    prepared_context = RL_exec.prepare_baseline_context(out_root, ckpt_path, device=args.device)
    write_metadata(
        metadata_json,
        build_metadata(
            args,
            out_root=out_root,
            ckpt_path=ckpt_path,
            prepared_context=prepared_context,
        ),
    )

    for trial in range(args.n_trials):
        config = sample_baseline_config(rng, DEFAULT_SEARCH_SPACE)
        try:
            row = evaluate_baseline_config(
                config,
                prepared_context=prepared_context,
                eval_split=args.eval_split,
                trial=trial,
                seed=args.seed,
                search_mode=args.search_mode,
            )
            row["score"] = score_baseline_row(
                row,
                min_fill_rate=args.min_fill_rate,
                max_drawdown=args.max_drawdown,
            )
            if args.verbose:
                print(
                    f"[trial {trial}] ok score={row['score']:.6f} split={row['baseline_eval_split']} "
                    f"pnl={row.get('net_pnl_pct')} fill_rate={row.get('maker_fill_rate')} config={config}"
                )
        except Exception as exc:
            row = make_error_row(
                config,
                trial=trial,
                seed=args.seed,
                eval_split=args.eval_split,
                search_mode=args.search_mode,
                exc=exc,
            )
            print(f"[trial {trial}] error {type(exc).__name__}: {exc}")
        rows.append(row)
        write_rows_csv(results_csv, rows)
        write_rows_jsonl(results_jsonl, rows)

    print_leaderboard(rows, top_k=args.top_k, label=f"baseline sweep {args.eval_split}")

    best_rows = [row for row in rows if row.get("status") == "ok"]
    best_rows.sort(key=lambda item: item.get("score", -math.inf), reverse=True)
    if best_rows:
        best = best_rows[0]
        best_env = {
            BASELINE_PARAM_ENV_MAP[key]: str(best[key])
            for key in BASELINE_PARAM_ENV_MAP
        }
        print("[best config env]", json.dumps(best_env, sort_keys=True))

    if args.retest_topk_on_test:
        top_rows = [row for row in best_rows[: args.top_k]]
        retest_rows: List[Dict[str, Any]] = []
        retest_csv = results_csv.with_name(f"{results_csv.stem}_topk_test.csv")
        retest_jsonl = retest_csv.with_suffix(".jsonl")
        for rank, row in enumerate(top_rows, start=1):
            config = {key: row[key] for key in BASELINE_PARAM_ENV_MAP}
            try:
                retest_row = evaluate_baseline_config(
                    config,
                    prepared_context=prepared_context,
                    eval_split="test",
                    trial=rank,
                    seed=args.seed,
                    search_mode=args.search_mode,
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
                    search_mode=args.search_mode,
                    exc=exc,
                )
                print(f"[retest rank {rank}] error {type(exc).__name__}: {exc}")
            retest_rows.append(retest_row)
            write_rows_csv(retest_csv, retest_rows)
            write_rows_jsonl(retest_jsonl, retest_rows)
        print_leaderboard(retest_rows, top_k=args.top_k, label="baseline sweep test retest")


if __name__ == "__main__":
    main()
