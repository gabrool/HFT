"""Deterministic performance harness for MMRT pipeline hot paths.

The harness intentionally uses tiny synthetic fixtures. It is not a market
quality benchmark; it is a stable safety rail that later optimization PRs can
use to prove they still produce the same artifacts while reducing runtime.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import time
from typing import Any, Callable

import numpy as np

from mmrt.contracts import AggressorSide, AsOfPolicy, LabelSpec, PriceReference
from mmrt.execution.contracts import BookLevelSnapshot, BookTop, SymbolSpec, TradePrint
from mmrt.execution.decision_grid import load_decision_grid, validate_decision_grid_for_execution_tape
from mmrt.execution.event_merge import iter_merged_execution_events, merge_execution_events
from mmrt.execution.execution_tape import build_execution_tape, save_execution_tape
from mmrt.execution.execution_tape_writer import StreamingExecutionTapeWriter, StreamingExecutionTapeWriterConfig
from mmrt.execution.l2_reconstructor import ReconstructedL2Event
from mmrt.features.labels import build_labels_from_price_event_arrays
from mmrt.cli.build_decision_grid import BuildDecisionGridConfig, build_decision_grid_from_config
from mmrt.linear.models import DirectionLinearHead, LinearModelConfig
from mmrt.metadata.symbol_rules import ExchangeSymbolRules, SymbolRuleMode

PERFORMANCE_BASELINE_SCHEMA = "mmrt_performance_baseline_v1"
DEFAULT_HEAVY_FILE_MIN_BYTES = 13_000


@dataclass(frozen=True, slots=True)
class PipelineBenchmarkConfig:
    iterations: int = 3
    include_optional: bool = True
    inventory_min_bytes: int = DEFAULT_HEAVY_FILE_MIN_BYTES
    work_root: str | None = None

    def __post_init__(self) -> None:
        if isinstance(self.iterations, bool) or not isinstance(self.iterations, int) or self.iterations <= 0:
            raise ValueError("iterations must be a positive int")
        if not isinstance(self.include_optional, bool):
            raise ValueError("include_optional must be bool")
        if isinstance(self.inventory_min_bytes, bool) or not isinstance(self.inventory_min_bytes, int) or self.inventory_min_bytes <= 0:
            raise ValueError("inventory_min_bytes must be a positive int")
        if self.work_root is not None and (not isinstance(self.work_root, str) or not self.work_root.strip()):
            raise ValueError("work_root must be None or a non-empty str")


def heavy_pipeline_file_inventory(
    repo_root: str | Path = ".",
    *,
    min_bytes: int = DEFAULT_HEAVY_FILE_MIN_BYTES,
) -> list[dict[str, object]]:
    if isinstance(min_bytes, bool) or not isinstance(min_bytes, int) or min_bytes <= 0:
        raise ValueError("min_bytes must be a positive int")
    root = Path(repo_root)
    mmrt_root = root / "mmrt"
    if not mmrt_root.exists():
        raise FileNotFoundError(f"missing mmrt package root: {mmrt_root}")
    rows: list[dict[str, object]] = []
    for path in mmrt_root.rglob("*.py"):
        rel = path.relative_to(root).as_posix()
        size = path.stat().st_size
        if size < min_bytes:
            continue
        rows.append(
            {
                "path": rel,
                "bytes": int(size),
                "subsystem": rel.split("/")[1] if "/" in rel else "root",
            }
        )
    rows.sort(key=lambda row: (-int(row["bytes"]), str(row["path"])))
    return rows


def build_pipeline_golden_outputs(work_root: str | Path) -> dict[str, object]:
    root = Path(work_root)
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)
    tape = _fixture_tape()
    tape_root = root / "tape"
    save_execution_tape(tape, tape_root, overwrite=True)
    grid_summary = build_decision_grid_from_config(
        BuildDecisionGridConfig(
            tape_root=str(tape_root),
            output_grid=str(root / "decision_grid"),
            output_json=str(root / "decision_grid_summary.json"),
            min_decision_interval_us=100,
            max_decision_interval_us=500,
            l1_size_change_fraction=0.25,
            chunk_rows=2,
            overwrite=True,
        )
    )
    grid = load_decision_grid(root / "decision_grid")
    validate_decision_grid_for_execution_tape(grid, tape)
    labels, mask = _fixture_labels()
    return {
        "schema": PERFORMANCE_BASELINE_SCHEMA,
        "fixture": {
            "events": int(tape.manifest.num_events),
            "l2_batches": int(tape.manifest.num_l2_batches),
            "trades": int(tape.manifest.num_trades),
            "decision_grid_rows": int(grid.n_rows),
        },
        "hashes": {
            "events": _array_digest(tape.arrays.events),
            "l2_events": _array_digest(tape.arrays.l2_events),
            "trades": _array_digest(tape.arrays.trades),
            "book_bid_ticks": _array_digest(tape.arrays.book_bid_ticks),
            "book_ask_ticks": _array_digest(tape.arrays.book_ask_ticks),
            "decision_event_index": _array_digest(grid.decision_event_index),
            "decision_local_ts_us": _array_digest(grid.decision_local_ts_us),
            "labels": _array_digest(labels),
            "label_mask": _array_digest(mask),
        },
        "decision_grid_hash": grid.decision_grid_hash,
        "reason_counts": grid_summary["reason_counts"],
    }


def run_pipeline_benchmarks(
    config: PipelineBenchmarkConfig | None = None,
    *,
    repo_root: str | Path = ".",
) -> dict[str, object]:
    cfg = config or PipelineBenchmarkConfig()
    inventory = heavy_pipeline_file_inventory(repo_root, min_bytes=cfg.inventory_min_bytes)
    with _benchmark_workspace(cfg.work_root) as work_root:
        cases = _benchmark_cases(Path(work_root), include_optional=cfg.include_optional)
        results = []
        for name, fn in cases:
            timings_ns: list[int] = []
            last_result: dict[str, object] | None = None
            for iteration in range(cfg.iterations):
                iter_root = Path(work_root) / name / str(iteration)
                if iter_root.exists():
                    shutil.rmtree(iter_root)
                iter_root.mkdir(parents=True, exist_ok=True)
                start_ns = time.perf_counter_ns()
                last_result = fn(iter_root)
                timings_ns.append(time.perf_counter_ns() - start_ns)
            results.append(_case_summary(name, timings_ns, last_result or {}))
        return {
            "schema": PERFORMANCE_BASELINE_SCHEMA,
            "config": {
                "iterations": cfg.iterations,
                "include_optional": cfg.include_optional,
                "inventory_min_bytes": cfg.inventory_min_bytes,
            },
            "heavy_file_inventory": inventory,
            "golden_outputs": build_pipeline_golden_outputs(Path(work_root) / "golden"),
            "benchmarks": results,
        }


def _benchmark_cases(
    work_root: Path,
    *,
    include_optional: bool,
) -> list[tuple[str, Callable[[Path], dict[str, object]]]]:
    cases: list[tuple[str, Callable[[Path], dict[str, object]]]] = [
        ("execution_tape_materialized", _benchmark_execution_tape_materialized),
        ("execution_tape_streaming_writer", _benchmark_execution_tape_streaming_writer),
        ("decision_grid_build", _benchmark_decision_grid_build),
        ("labels_batch", _benchmark_labels_batch),
        ("linear_direction_update", _benchmark_linear_direction_update),
    ]
    if include_optional:
        cases.append(("ppo_update", _benchmark_ppo_update))
    return cases


def _benchmark_execution_tape_materialized(_: Path) -> dict[str, object]:
    tape = _fixture_tape()
    return {"events": int(tape.manifest.num_events), "l2_batches": int(tape.manifest.num_l2_batches)}


def _benchmark_execution_tape_streaming_writer(root: Path) -> dict[str, object]:
    l2_events, trades = _fixture_events()
    writer = StreamingExecutionTapeWriter(
        StreamingExecutionTapeWriterConfig(
            output_root=str(root / "streamed_tape"),
            symbol_spec=_symbol_spec(),
            symbol_rules=_symbol_rules(),
            book_depth=2,
            chunk_rows=2,
            overwrite=True,
        )
    )
    for event in iter_merged_execution_events(l2_events, trades):
        writer.append(event)
    result = writer.finalize()
    return {
        "events": int(result.tape.manifest.num_events),
        "chunks_cleaned": bool(result.chunk_summary["chunks_cleaned"]),
    }


def _benchmark_decision_grid_build(root: Path) -> dict[str, object]:
    tape = _fixture_tape()
    tape_root = root / "tape"
    save_execution_tape(tape, tape_root, overwrite=True)
    summary = build_decision_grid_from_config(
        BuildDecisionGridConfig(
            tape_root=str(tape_root),
            output_grid=str(root / "decision_grid"),
            output_json=str(root / "summary.json"),
            min_decision_interval_us=100,
            max_decision_interval_us=500,
            l1_size_change_fraction=0.25,
            chunk_rows=2,
            overwrite=True,
        )
    )
    return {"decision_grid_rows": int(summary["counters"]["decision_grid_rows"])}


def _benchmark_labels_batch(_: Path) -> dict[str, object]:
    labels, mask = _fixture_labels()
    return {"rows": int(labels.shape[0]), "valid_rows": int(mask.sum())}


def _benchmark_linear_direction_update(_: Path) -> dict[str, object]:
    rng = np.random.default_rng(20260614)
    x = rng.normal(size=(128, 4)).astype(np.float64)
    y = (x[:, 0] + 0.5 * x[:, 1] > 0.0).astype(np.int8)
    head = DirectionLinearHead(
        ("f0", "f1", "f2", "f3"),
        LinearModelConfig(learning_rate=0.05, l2=1e-4, max_grad_norm=10.0),
    )
    for _ in range(8):
        head.partial_fit(x, y)
    return {"updates": int(head.n_updates), "rows_seen": int(head.n_rows_seen)}


def _benchmark_ppo_update(_: Path) -> dict[str, object]:
    try:
        import torch
        from mmrt.rl.ppo import PPOConfig, update_ppo
        from mmrt.rl.rollout import RolloutBatch
        from mmrt.rl.torch_networks import EXECUTION_ACTION_DIM, ActorCriticConfig, ActorCriticNetwork
    except Exception as exc:  # pragma: no cover - depends on optional torch installs.
        return {"skipped": True, "reason": type(exc).__name__}

    torch.manual_seed(20260614)
    steps = 8
    obs_dim = 4
    action_dim = EXECUTION_ACTION_DIM
    batch = RolloutBatch(
        observations=torch.randn(steps, obs_dim),
        actions=torch.cat((torch.randint(0, 2, (steps, 2), dtype=torch.float32), torch.randn(steps, action_dim - 2)), dim=-1),
        log_probs=torch.zeros(steps),
        values=torch.zeros(steps),
        rewards=torch.randn(steps),
        dones=torch.zeros(steps, dtype=torch.bool),
        terminated=torch.zeros(steps, dtype=torch.bool),
        truncated=torch.zeros(steps, dtype=torch.bool),
        advantages=torch.arange(steps, dtype=torch.float32),
        returns=torch.randn(steps),
        entropies=torch.zeros(steps),
        episode_count=0,
    )
    policy = ActorCriticNetwork(obs_dim=obs_dim, config=ActorCriticConfig(hidden_sizes=(8,)))
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
    metrics = update_ppo(policy, optimizer, batch, config=PPOConfig(update_epochs=1, minibatch_size=4))
    return {
        "updates": 1,
        "loss": float(metrics.loss),
        "minibatches_processed": int(metrics.minibatches_processed),
    }


def _case_summary(name: str, timings_ns: list[int], result: dict[str, object]) -> dict[str, object]:
    arr = np.asarray(timings_ns, dtype=np.float64)
    return {
        "name": name,
        "iterations": int(arr.size),
        "min_ms": float(arr.min() / 1_000_000.0),
        "mean_ms": float(arr.mean() / 1_000_000.0),
        "max_ms": float(arr.max() / 1_000_000.0),
        "last_result": result,
    }


def _fixture_events() -> tuple[list[ReconstructedL2Event], list[TradePrint]]:
    l2_events = [
        _l2(seq=0, local_ts_us=100),
        _l2(seq=1, local_ts_us=200),
        _l2(seq=2, local_ts_us=250, bid_size=2.0),
        _l2(seq=3, local_ts_us=300, bid_size=2.0),
        _l2(seq=4, local_ts_us=800, bid_size=2.0),
    ]
    trades = [
        TradePrint(
            local_ts_us=150,
            ts_us=149,
            side=AggressorSide.BUY,
            price_tick=1002,
            amount=0.01,
            trade_id="0",
            source_row=0,
        )
    ]
    return l2_events, trades


def _fixture_tape():
    l2_events, trades = _fixture_events()
    return build_execution_tape(
        symbol_spec=_symbol_spec(),
        symbol_rules=_symbol_rules(),
        l2_events=l2_events,
        trades=trades,
        merged_events=merge_execution_events(l2_events, trades).events,
        book_depth=2,
        created_at_utc="2026-01-01T00:00:00Z",
    )


def _fixture_labels() -> tuple[np.ndarray, np.ndarray]:
    spec = LabelSpec(
        horizons_us=(100_000, 200_000),
        entry_delay_us=0,
        price_reference=PriceReference.MID,
        asof_policy=AsOfPolicy.LAST_OBSERVATION,
    )
    price_ts = np.array([1_000_000, 1_100_000, 1_200_000, 1_300_000], dtype=np.int64)
    price_seq = np.arange(price_ts.shape[0], dtype=np.int64)
    prices = np.array([100.0, 101.0, 102.0, 103.0], dtype=np.float64)
    decision_ts = np.array([1_000_000, 1_100_000], dtype=np.int64)
    decision_seq = np.array([0, 1], dtype=np.int64)
    return build_labels_from_price_event_arrays(decision_ts, decision_seq, price_ts, price_seq, prices, spec)


def _l2(seq: int, local_ts_us: int, *, bid_size: float = 1.0) -> ReconstructedL2Event:
    return ReconstructedL2Event(
        batch_seq=seq,
        local_ts_us=local_ts_us,
        min_ts_us=local_ts_us,
        max_ts_us=local_ts_us,
        num_updates=1,
        is_snapshot_batch=(seq == 0),
        book_top=BookTop(local_ts_us, 1000, 1002, bid_size, 1.2),
        bid_depth=2,
        ask_depth=2,
        book_snapshot=BookLevelSnapshot(
            local_ts_us,
            (1000, 999),
            (bid_size, 2.0),
            (1002, 1003),
            (1.2, 2.2),
        ),
    )


def _symbol_spec() -> SymbolSpec:
    return SymbolSpec("binance-futures", "BTCUSDT", 0.1, 0.001, 0.001, 100.0, 5.0)


def _symbol_rules() -> ExchangeSymbolRules:
    return ExchangeSymbolRules(
        exchange="binance-futures",
        symbol="BTCUSDT",
        mode=SymbolRuleMode.CURRENT_RULES_REPLAY,
        base_asset="BTC",
        quote_asset="USDT",
        margin_asset="USDT",
        contract_type="PERPETUAL",
        status="TRADING",
        tick_size=Decimal("0.1"),
        min_price=Decimal("0.1"),
        max_price=Decimal("1000000"),
        step_size=Decimal("0.001"),
        min_qty=Decimal("0.001"),
        max_qty=Decimal("100"),
        min_notional=Decimal("5"),
        allowed_order_types=("LIMIT",),
        allowed_time_in_force=("GTC", "GTX"),
    )


def _array_digest(array: np.ndarray) -> str:
    arr = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(arr.dtype).encode("utf-8"))
    digest.update(json.dumps(arr.shape).encode("utf-8"))
    digest.update(arr.view(np.uint8))
    return digest.hexdigest()


class _benchmark_workspace:
    def __init__(self, work_root: str | None) -> None:
        self._requested = Path(work_root) if work_root is not None else None
        self._tmp: tempfile.TemporaryDirectory[str] | None = None
        self.path: Path | None = None

    def __enter__(self) -> Path:
        if self._requested is None:
            self._tmp = tempfile.TemporaryDirectory(prefix="mmrt-perf-")
            self.path = Path(self._tmp.name)
        else:
            self.path = self._requested
            if self.path.exists():
                shutil.rmtree(self.path)
            self.path.mkdir(parents=True, exist_ok=True)
        return self.path

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._tmp is not None:
            self._tmp.cleanup()


__all__ = [
    "PERFORMANCE_BASELINE_SCHEMA",
    "DEFAULT_HEAVY_FILE_MIN_BYTES",
    "PipelineBenchmarkConfig",
    "heavy_pipeline_file_inventory",
    "build_pipeline_golden_outputs",
    "run_pipeline_benchmarks",
]
