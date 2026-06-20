"""Empirical alpha actionability diagnostics for execution profiling."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from mmrt.execution.adverse_selection import (
    candidate_price_tick,
    quote_candidate_configs_from_names,
)
from mmrt.execution.adverse_selection_dataset import (
    ADVERSE_SELECTION_DATASET_SCHEMA,
    DiskBackedAdverseSelectionDataset,
    load_adverse_selection_dataset,
)
from mmrt.execution.contracts import OrderSide
from mmrt.execution.decision_grid import DecisionGrid
from mmrt.execution.execution_tape import ExecutionTape
from mmrt.execution.linear_signal import LinearSignalArtifact
from mmrt.execution.split_contract import DecisionSplitRange, ranges_for_split, validate_split_contract_payload


ALPHA_ACTIONABILITY_SOURCE = "empirical_adverse_dataset_labels"
DEFAULT_ALPHA_ACTIONABILITY_PERCENTILES = (10, 20)
DEFAULT_ALPHA_ACTIONABILITY_MAX_ROWS = 1_000_000
DEFAULT_ALPHA_ACTIONABILITY_RANDOM_SEED = 123
DEFAULT_ALPHA_ACTIONABILITY_DECISION_HORIZON_US = 1_000_000
DEFAULT_ALPHA_ACTIONABILITY_FILL_PLUS_HORIZON_US = 1_000_000
DEFAULT_ALPHA_ACTIONABILITY_CORRECTNESS_DEADBAND_BPS = 0.0

_AXIS_DESCRIPTIONS = {
    "direction_score": "2*p_up_given_move - 1; primary direction-only alpha axis",
    "signed_move_prob": "p_up_move - p_down_move; no-move-gated directional alpha axis",
    "expected_return_bps": "expected_up_bps - expected_down_bps; economic combined alpha axis",
}

_CANDIDATES = ("touch", "inside_1", "away_1")
_REQUIRED_CANDIDATES = ("touch",)
_SIDES = ("bid", "ask")
_LABEL_SUFFIXES = (
    "filled",
    "fill_latency_us",
    "adverse_bps",
    "toxic_fill",
    "toxic_cost_bps",
)


@dataclass(frozen=True, slots=True)
class QuoteLabelSpec:
    output_name: str
    prefix: str
    filled: str
    fill_latency_us: str
    adverse_bps: str
    toxic_fill: str
    toxic_cost_bps: str


@dataclass(frozen=True, slots=True)
class MarkoutContext:
    decision_local_ts_us: np.ndarray
    decision_mid_ticks: np.ndarray
    decision_horizon_mid_ticks: np.ndarray
    candidate_price_ticks: Mapping[str, np.ndarray]


@dataclass(frozen=True, slots=True)
class DirectionMasks:
    available: np.ndarray
    correct: np.ndarray
    wrong: np.ndarray
    no_move: np.ndarray
    metrics: Mapping[str, object]


class ExecutionTapeMarkoutIndex:
    """Tape-backed lookup for decision books and future L2 mids."""

    def __init__(self, *, tape: ExecutionTape, decision_grid: DecisionGrid, post_only_gap_ticks: int = 1) -> None:
        if not isinstance(tape, ExecutionTape):
            raise ValueError("tape must be ExecutionTape")
        if not isinstance(decision_grid, DecisionGrid):
            raise ValueError("decision_grid must be DecisionGrid")
        if isinstance(post_only_gap_ticks, bool) or int(post_only_gap_ticks) < 0:
            raise ValueError("post_only_gap_ticks must be a nonnegative int")
        self.tape = tape
        self.decision_grid = decision_grid
        self.post_only_gap_ticks = int(post_only_gap_ticks)
        self._l2_ts = np.asarray(tape.arrays.l2_events["local_ts_us"], dtype=np.int64)
        self._l2_bid = np.asarray(tape.arrays.l2_events["best_bid_tick"], dtype=np.int64)
        self._l2_ask = np.asarray(tape.arrays.l2_events["best_ask_tick"], dtype=np.int64)

    def _validate_rows(self, linear_rows: np.ndarray) -> np.ndarray:
        rows = np.asarray(linear_rows, dtype=np.int64)
        if rows.ndim != 1:
            raise ValueError("sampled linear rows must be a rank-1 array")
        if rows.size and (int(np.min(rows)) < 0 or int(np.max(rows)) >= self.decision_grid.n_rows):
            raise ValueError("sampled linear rows are outside decision grid bounds")
        return rows

    def decision_local_ts_us(self, linear_rows: np.ndarray) -> np.ndarray:
        rows = self._validate_rows(linear_rows)
        return np.asarray(self.decision_grid.decision_local_ts_us[rows], dtype=np.int64)

    def decision_mid_ticks(self, linear_rows: np.ndarray) -> np.ndarray:
        rows = self._validate_rows(linear_rows)
        return self._mid_ticks_for_book_ptrs(np.asarray(self.decision_grid.book_ptr[rows], dtype=np.int64))

    def future_mid_ticks_at_or_after(self, target_local_ts_us: np.ndarray) -> np.ndarray:
        targets = np.asarray(target_local_ts_us, dtype=np.int64)
        out = np.full(targets.shape, np.nan, dtype=np.float64)
        if targets.size == 0 or self._l2_ts.size == 0:
            return out
        left = np.searchsorted(self._l2_ts, targets, side="left")
        valid = left < self._l2_ts.size
        if not np.any(valid):
            return out
        idx = left.copy()
        valid_left = left[valid]
        exact = self._l2_ts[valid_left] == targets[valid]
        if np.any(exact):
            right = np.searchsorted(self._l2_ts, targets[valid][exact], side="right")
            valid_indices = np.flatnonzero(valid)
            idx[valid_indices[exact]] = right - 1
        out[valid] = self._mid_ticks_for_l2_ptrs(idx[valid])
        return out

    def candidate_price_ticks(self, linear_rows: np.ndarray, quote_name: str) -> np.ndarray:
        rows = self._validate_rows(linear_rows)
        side_name, candidate_name = _split_quote_name(quote_name)
        side = OrderSide.BUY if side_name == "bid" else OrderSide.SELL
        candidate = quote_candidate_configs_from_names((candidate_name,))[0]
        book_ptrs = np.asarray(self.decision_grid.book_ptr[rows], dtype=np.int64)
        best_bid = self._l2_bid[book_ptrs]
        best_ask = self._l2_ask[book_ptrs]
        out = np.full(rows.shape, np.nan, dtype=np.float64)
        for idx, (bid, ask) in enumerate(zip(best_bid, best_ask)):
            price = candidate_price_tick(
                candidate=candidate,
                side=side,
                best_bid=int(bid),
                best_ask=int(ask),
                post_only_gap_ticks=self.post_only_gap_ticks,
            )
            if price is not None and price > 0:
                out[idx] = float(price)
        return out

    def _mid_ticks_for_book_ptrs(self, book_ptrs: np.ndarray) -> np.ndarray:
        out = np.full(book_ptrs.shape, np.nan, dtype=np.float64)
        valid = (
            (book_ptrs >= 0)
            & (book_ptrs < self._l2_bid.shape[0])
            & (self._l2_bid[book_ptrs] > 0)
            & (self._l2_ask[book_ptrs] > self._l2_bid[book_ptrs])
        )
        out[valid] = (self._l2_bid[book_ptrs[valid]].astype(np.float64) + self._l2_ask[book_ptrs[valid]].astype(np.float64)) * 0.5
        return out

    def _mid_ticks_for_l2_ptrs(self, l2_ptrs: np.ndarray) -> np.ndarray:
        out = np.full(l2_ptrs.shape, np.nan, dtype=np.float64)
        valid = (
            (l2_ptrs >= 0)
            & (l2_ptrs < self._l2_bid.shape[0])
            & (self._l2_bid[l2_ptrs] > 0)
            & (self._l2_ask[l2_ptrs] > self._l2_bid[l2_ptrs])
        )
        out[valid] = (self._l2_bid[l2_ptrs[valid]].astype(np.float64) + self._l2_ask[l2_ptrs[valid]].astype(np.float64)) * 0.5
        return out


def parse_alpha_actionability_percentiles(value: str | Sequence[int]) -> tuple[int, ...]:
    """Parse symmetric tail percentiles, requiring values in (0, 50)."""

    if isinstance(value, str):
        parts = [part.strip() for part in value.split(",") if part.strip()]
        if not parts:
            raise ValueError("alpha_actionability_percentiles must contain at least one percentile")
        raw_values: list[int] = []
        for part in parts:
            try:
                raw_values.append(int(part))
            except ValueError as exc:
                raise ValueError(f"invalid alpha_actionability_percentile {part!r}") from exc
    else:
        raw_values = [int(item) for item in value]
    percentiles = tuple(sorted(set(raw_values)))
    if not percentiles:
        raise ValueError("alpha_actionability_percentiles must contain at least one percentile")
    invalid = [item for item in percentiles if item <= 0 or item >= 50]
    if invalid:
        raise ValueError("alpha_actionability_percentiles must be > 0 and < 50")
    return percentiles


def compute_alpha_actionability_summary(
    *,
    adverse_dataset_root: str | Path,
    split: str,
    split_contract: Mapping[str, object],
    decision_grid_hash: str,
    decision_grid_n_rows: int,
    linear_signals: LinearSignalArtifact,
    execution_tape: ExecutionTape | None = None,
    decision_grid: DecisionGrid | None = None,
    markout_index: ExecutionTapeMarkoutIndex | None = None,
    max_rows: int = DEFAULT_ALPHA_ACTIONABILITY_MAX_ROWS,
    percentiles: str | Sequence[int] = DEFAULT_ALPHA_ACTIONABILITY_PERCENTILES,
    seed: int = DEFAULT_ALPHA_ACTIONABILITY_RANDOM_SEED,
    decision_horizon_us: int = DEFAULT_ALPHA_ACTIONABILITY_DECISION_HORIZON_US,
    fill_plus_horizon_us: int = DEFAULT_ALPHA_ACTIONABILITY_FILL_PLUS_HORIZON_US,
    correctness_deadband_bps: float = DEFAULT_ALPHA_ACTIONABILITY_CORRECTNESS_DEADBAND_BPS,
    maker_fee_bps: float = 0.0,
    post_only_gap_ticks: int = 1,
    chunk_rows: int = 100_000,
) -> dict[str, object]:
    """Compute empirical maker-fill actionability by alpha tail buckets."""

    if not isinstance(linear_signals, LinearSignalArtifact):
        raise ValueError("linear_signals must be LinearSignalArtifact")
    max_rows = _positive_int(max_rows, "alpha_actionability_max_rows")
    seed = _nonnegative_int(seed, "alpha_actionability_random_seed")
    chunk_rows = _positive_int(chunk_rows, "chunk_rows")
    decision_horizon_us = _positive_int(decision_horizon_us, "alpha_actionability_decision_horizon_us")
    fill_plus_horizon_us = _positive_int(fill_plus_horizon_us, "alpha_actionability_fill_plus_horizon_us")
    correctness_deadband_bps = _nonnegative_finite_float(
        correctness_deadband_bps,
        "alpha_actionability_correctness_deadband_bps",
    )
    maker_fee_bps = _finite_float(maker_fee_bps, "maker_fee_bps")
    post_only_gap_ticks = _nonnegative_int(post_only_gap_ticks, "post_only_gap_ticks")
    percentiles_tuple = parse_alpha_actionability_percentiles(percentiles)
    if markout_index is None and execution_tape is not None and decision_grid is not None:
        markout_index = ExecutionTapeMarkoutIndex(
            tape=execution_tape,
            decision_grid=decision_grid,
            post_only_gap_ticks=post_only_gap_ticks,
        )
    dataset = load_adverse_selection_dataset(adverse_dataset_root, mmap_mode="r")
    _validate_lineage(
        dataset=dataset,
        split_contract=split_contract,
        decision_grid_hash=decision_grid_hash,
        decision_grid_n_rows=decision_grid_n_rows,
        linear_signals=linear_signals,
    )
    quote_specs = _available_quote_specs(dataset.label_names)
    label_names_used = tuple(dict.fromkeys(label for spec in quote_specs for label in _labels_for_spec(spec)))
    split_ranges = ranges_for_split(split_contract, split)
    sampled_dataset_rows, sampled_linear_rows, rows_available, sampling = _sample_rows_for_split(
        dataset=dataset,
        linear_signals=linear_signals,
        split=split,
        split_ranges=split_ranges,
        max_rows=max_rows,
        seed=seed,
        chunk_rows=chunk_rows,
    )
    if sampled_dataset_rows.size == 0:
        raise ValueError(f"selected split {split!r} has no adverse dataset rows")

    labels, masks = _load_label_arrays(dataset, sampled_dataset_rows, label_names_used)
    axes = _axis_arrays(linear_signals, sampled_linear_rows)
    markout = _markout_context(
        markout_index=markout_index,
        sampled_linear_rows=sampled_linear_rows,
        quote_specs=quote_specs,
        decision_horizon_us=decision_horizon_us,
    )
    unconditional = _unconditional_quote_metrics(quote_specs, labels, masks)
    axis_summaries = {
        axis_name: _axis_summary(
            axis_name=axis_name,
            axis_values=axis_values,
            percentiles=percentiles_tuple,
            quote_specs=quote_specs,
            labels=labels,
            masks=masks,
            markout=markout,
            markout_index=markout_index,
            decision_horizon_us=decision_horizon_us,
            fill_plus_horizon_us=fill_plus_horizon_us,
            correctness_deadband_bps=correctness_deadband_bps,
            maker_fee_bps=maker_fee_bps,
            unconditional=unconditional,
        )
        for axis_name, axis_values in axes.items()
    }
    warnings = _alpha_warnings(
        axes=axis_summaries,
        rows_sampled=int(sampled_dataset_rows.size),
        percentiles=percentiles_tuple,
        unconditional=unconditional,
    )
    compact = _compact_summary(axis_summaries, percentiles_tuple)
    compact["warning_count"] = len(warnings)
    compact["warnings"] = warnings
    return {
        "enabled": True,
        "source": ALPHA_ACTIONABILITY_SOURCE,
        "adverse_dataset_root": str(adverse_dataset_root),
        "selected_split": split,
        "sample": {
            "rows_available_in_split": rows_available,
            "rows_sampled": int(sampled_dataset_rows.size),
            "max_rows": max_rows,
            "seed": seed,
            "sampling": sampling,
        },
        "lineage": {
            "decision_grid_hash": decision_grid_hash,
            "decision_grid_n_rows": int(decision_grid_n_rows),
            "adverse_dataset_schema": dataset.manifest.schema,
            "adverse_dataset_num_rows": dataset.num_rows,
            "label_names_used": list(label_names_used),
        },
        "markout_config": {
            "decision_horizon_us": decision_horizon_us,
            "fill_plus_horizon_us": fill_plus_horizon_us,
            "correctness_deadband_bps": correctness_deadband_bps,
            "maker_fee_bps": maker_fee_bps,
            "future_mid_source": "execution_tape_l2_mid_at_or_after",
        },
        "axes": axis_summaries,
        "compact": compact,
    }


def _positive_int(value: int, name: str) -> int:
    if isinstance(value, bool) or int(value) <= 0:
        raise ValueError(f"{name} must be a positive int")
    return int(value)


def _nonnegative_int(value: int, name: str) -> int:
    if isinstance(value, bool) or int(value) < 0:
        raise ValueError(f"{name} must be a nonnegative int")
    return int(value)


def _finite_float(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite float")
    out = float(value)
    if not np.isfinite(out):
        raise ValueError(f"{name} must be a finite float")
    return out


def _nonnegative_finite_float(value: float, name: str) -> float:
    out = _finite_float(value, name)
    if out < 0.0:
        raise ValueError(f"{name} must be finite and >= 0")
    return out


def _labels_for_prefix(prefix: str) -> tuple[str, ...]:
    return tuple(f"{prefix}_{suffix}" for suffix in _LABEL_SUFFIXES)


def _labels_for_spec(spec: QuoteLabelSpec) -> tuple[str, ...]:
    return (
        spec.filled,
        spec.fill_latency_us,
        spec.adverse_bps,
        spec.toxic_fill,
        spec.toxic_cost_bps,
    )


def _split_quote_name(quote_name: str) -> tuple[str, str]:
    if quote_name.startswith("bid_"):
        return "bid", quote_name[len("bid_"):]
    if quote_name.startswith("ask_"):
        return "ask", quote_name[len("ask_"):]
    raise ValueError(f"invalid quote name {quote_name!r}")


def _available_quote_specs(label_names: Sequence[str]) -> tuple[QuoteLabelSpec, ...]:
    available = set(label_names)
    missing_required: list[str] = []
    specs: list[QuoteLabelSpec] = []
    for candidate in _CANDIDATES:
        candidate_specs: list[QuoteLabelSpec] = []
        candidate_missing: list[str] = []
        for side in _SIDES:
            prefix = f"{side}_{candidate}"
            labels = _labels_for_prefix(prefix)
            missing = [label for label in labels if label not in available]
            candidate_missing.extend(missing)
            if not missing:
                candidate_specs.append(
                    QuoteLabelSpec(
                        output_name=prefix,
                        prefix=prefix,
                        filled=labels[0],
                        fill_latency_us=labels[1],
                        adverse_bps=labels[2],
                        toxic_fill=labels[3],
                        toxic_cost_bps=labels[4],
                    )
                )
        if candidate in _REQUIRED_CANDIDATES and candidate_missing:
            missing_required.extend(candidate_missing)
        elif len(candidate_specs) == len(_SIDES):
            specs.extend(candidate_specs)
    if missing_required:
        raise ValueError(f"adverse dataset missing required touch labels: {sorted(set(missing_required))}")
    return tuple(specs)


def _split_contract_view(contract: Mapping[str, object]) -> dict[str, object]:
    view = validate_split_contract_payload(contract)
    view.pop("adverse_row_counts", None)
    view.pop("adverse_dataset_rows_total", None)
    return view


def _validate_lineage(
    *,
    dataset: DiskBackedAdverseSelectionDataset,
    split_contract: Mapping[str, object],
    decision_grid_hash: str,
    decision_grid_n_rows: int,
    linear_signals: LinearSignalArtifact,
) -> None:
    manifest = dataset.manifest
    if manifest.schema != ADVERSE_SELECTION_DATASET_SCHEMA:
        raise ValueError("adverse dataset schema must be mmrt_adverse_selection_dataset_grid_v1")
    if manifest.decision_grid_hash != decision_grid_hash:
        raise ValueError("adverse dataset decision_grid_hash does not match loaded decision grid")
    if manifest.decision_grid_n_rows != int(decision_grid_n_rows):
        raise ValueError("adverse dataset decision_grid_n_rows does not match loaded decision grid")
    if linear_signals.metadata.decision_grid_hash != decision_grid_hash:
        raise ValueError("linear_signals decision_grid_hash does not match loaded decision grid")
    if linear_signals.metadata.decision_grid_n_rows != int(decision_grid_n_rows):
        raise ValueError("linear_signals decision_grid_n_rows does not match loaded decision grid")
    if _split_contract_view(manifest.split_contract) != _split_contract_view(split_contract):
        raise ValueError("adverse dataset split_contract does not match profile split source")
    dataset_total = manifest.split_contract.get("adverse_dataset_rows_total")
    if dataset_total is not None and int(dataset_total) != dataset.num_rows:
        raise ValueError("adverse dataset row count lineage does not match manifest num_rows")
    adverse_counts = manifest.split_contract.get("adverse_row_counts")
    if isinstance(adverse_counts, Mapping):
        counted_rows = sum(int(adverse_counts.get(role, 0)) for role in ("train", "val", "test", "out_of_split"))
        if counted_rows != dataset.num_rows:
            raise ValueError("adverse dataset split row count lineage does not match manifest num_rows")


def _sample_rows_for_split(
    *,
    dataset: DiskBackedAdverseSelectionDataset,
    linear_signals: LinearSignalArtifact,
    split: str,
    split_ranges: Sequence[DecisionSplitRange],
    max_rows: int,
    seed: int,
    chunk_rows: int,
) -> tuple[np.ndarray, np.ndarray, int, str]:
    rows_available = _count_split_rows(
        dataset=dataset,
        linear_signals=linear_signals,
        split_ranges=split_ranges,
        chunk_rows=chunk_rows,
    )
    adverse_counts = dataset.manifest.split_contract.get("adverse_row_counts")
    if isinstance(adverse_counts, Mapping):
        expected_rows = int(adverse_counts.get(split, 0))
        if rows_available != expected_rows:
            raise ValueError("adverse dataset selected split row count lineage does not match mapped rows")
    sample_count = min(rows_available, max_rows)
    if rows_available <= max_rows:
        target_ordinals = None
        sampling = "all"
    else:
        rng = np.random.default_rng(seed)
        target_ordinals = np.sort(rng.choice(rows_available, size=sample_count, replace=False).astype(np.int64))
        sampling = "deterministic_random_without_replacement"
    sampled_dataset_rows, sampled_linear_rows = _collect_split_rows(
        dataset=dataset,
        linear_signals=linear_signals,
        split_ranges=split_ranges,
        target_ordinals=target_ordinals,
        chunk_rows=chunk_rows,
    )
    return sampled_dataset_rows, sampled_linear_rows, rows_available, sampling


def _count_split_rows(
    *,
    dataset: DiskBackedAdverseSelectionDataset,
    linear_signals: LinearSignalArtifact,
    split_ranges: Sequence[DecisionSplitRange],
    chunk_rows: int,
) -> int:
    rows_available = 0
    for start in range(0, dataset.num_rows, chunk_rows):
        end = min(start + chunk_rows, dataset.num_rows)
        linear_rows = _map_dataset_rows_to_linear_rows(dataset, linear_signals, start, end)
        rows_available += int(np.count_nonzero(_rows_in_ranges(linear_rows, split_ranges)))
    return rows_available


def _collect_split_rows(
    *,
    dataset: DiskBackedAdverseSelectionDataset,
    linear_signals: LinearSignalArtifact,
    split_ranges: Sequence[DecisionSplitRange],
    target_ordinals: np.ndarray | None,
    chunk_rows: int,
) -> tuple[np.ndarray, np.ndarray]:
    dataset_parts: list[np.ndarray] = []
    linear_parts: list[np.ndarray] = []
    selected_seen = 0
    for start in range(0, dataset.num_rows, chunk_rows):
        end = min(start + chunk_rows, dataset.num_rows)
        linear_rows = _map_dataset_rows_to_linear_rows(dataset, linear_signals, start, end)
        split_mask = _rows_in_ranges(linear_rows, split_ranges)
        selected_offsets = np.flatnonzero(split_mask)
        selected_count = int(selected_offsets.size)
        if selected_count == 0:
            continue
        if target_ordinals is None:
            take_offsets = selected_offsets
        else:
            left = int(np.searchsorted(target_ordinals, selected_seen, side="left"))
            right = int(np.searchsorted(target_ordinals, selected_seen + selected_count, side="left"))
            local_ordinals = target_ordinals[left:right] - selected_seen
            take_offsets = selected_offsets[local_ordinals]
        if take_offsets.size:
            dataset_parts.append((start + take_offsets).astype(np.int64, copy=False))
            linear_parts.append(linear_rows[take_offsets].astype(np.int64, copy=False))
        selected_seen += selected_count
    if not dataset_parts:
        empty = np.empty(0, dtype=np.int64)
        return empty, empty
    return np.concatenate(dataset_parts), np.concatenate(linear_parts)


def _map_dataset_rows_to_linear_rows(
    dataset: DiskBackedAdverseSelectionDataset,
    linear_signals: LinearSignalArtifact,
    start: int,
    end: int,
) -> np.ndarray:
    dataset_event = np.asarray(dataset.arrays.decision_event_index[start:end], dtype=np.int64)
    positions = np.searchsorted(linear_signals.decision_event_index, dataset_event)
    in_bounds = positions < linear_signals.n_rows
    valid_positions = np.where(in_bounds, positions, 0)
    mismatch = ~in_bounds
    mismatch |= linear_signals.decision_event_index[valid_positions] != dataset_event
    mismatch |= linear_signals.decision_local_ts_us[valid_positions] != np.asarray(
        dataset.arrays.decision_local_ts_us[start:end],
        dtype=np.int64,
    )
    mismatch |= linear_signals.decision_event_seq[valid_positions] != np.asarray(
        dataset.arrays.decision_event_seq[start:end],
        dtype=np.int64,
    )
    if np.any(mismatch):
        first = int(start + np.flatnonzero(mismatch)[0])
        raise ValueError(f"adverse dataset decision row mapping mismatch at dataset row {first}")
    return positions.astype(np.int64, copy=False)


def _rows_in_ranges(rows: np.ndarray, ranges: Sequence[DecisionSplitRange]) -> np.ndarray:
    if not ranges:
        return np.zeros(rows.shape, dtype=np.bool_)
    starts = np.asarray([item.start_decision_row for item in ranges], dtype=np.int64)
    ends = np.asarray([item.end_decision_row for item in ranges], dtype=np.int64)
    range_idx = np.searchsorted(starts, rows, side="right") - 1
    valid = range_idx >= 0
    out = np.zeros(rows.shape, dtype=np.bool_)
    if np.any(valid):
        valid_idx = range_idx[valid]
        out[valid] = rows[valid] < ends[valid_idx]
    return out


def _load_label_arrays(
    dataset: DiskBackedAdverseSelectionDataset,
    sampled_dataset_rows: np.ndarray,
    label_names: Sequence[str],
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    label_index = {name: idx for idx, name in enumerate(dataset.label_names)}
    labels: dict[str, np.ndarray] = {}
    masks: dict[str, np.ndarray] = {}
    for name in label_names:
        idx = label_index[name]
        labels[name] = np.asarray(dataset.arrays.labels[sampled_dataset_rows, idx], dtype=np.float64)
        masks[name] = np.asarray(dataset.arrays.label_masks[sampled_dataset_rows, idx], dtype=np.bool_)
    return labels, masks


def _axis_arrays(linear_signals: LinearSignalArtifact, sampled_linear_rows: np.ndarray) -> dict[str, np.ndarray]:
    arrays = linear_signals.arrays
    p_move = np.asarray(arrays.p_move[sampled_linear_rows], dtype=np.float64)
    signed_move_prob = np.asarray(arrays.signed_move_prob[sampled_linear_rows], dtype=np.float64)
    direction_score = np.clip(signed_move_prob / np.maximum(p_move, 1e-12), -1.0, 1.0)
    expected_return_bps = np.asarray(arrays.expected_return_bps[sampled_linear_rows], dtype=np.float64)
    return {
        "direction_score": direction_score,
        "signed_move_prob": signed_move_prob,
        "expected_return_bps": expected_return_bps,
    }


def _markout_context(
    *,
    markout_index: ExecutionTapeMarkoutIndex | None,
    sampled_linear_rows: np.ndarray,
    quote_specs: Sequence[QuoteLabelSpec],
    decision_horizon_us: int,
) -> MarkoutContext:
    n = int(sampled_linear_rows.size)
    if markout_index is None:
        return MarkoutContext(
            decision_local_ts_us=np.zeros(n, dtype=np.int64),
            decision_mid_ticks=np.full(n, np.nan, dtype=np.float64),
            decision_horizon_mid_ticks=np.full(n, np.nan, dtype=np.float64),
            candidate_price_ticks={
                spec.output_name: np.full(n, np.nan, dtype=np.float64)
                for spec in quote_specs
            },
        )
    decision_ts = markout_index.decision_local_ts_us(sampled_linear_rows)
    decision_mid = markout_index.decision_mid_ticks(sampled_linear_rows)
    decision_future_mid = markout_index.future_mid_ticks_at_or_after(decision_ts + int(decision_horizon_us))
    candidate_prices = {
        spec.output_name: markout_index.candidate_price_ticks(sampled_linear_rows, spec.output_name)
        for spec in quote_specs
    }
    return MarkoutContext(
        decision_local_ts_us=decision_ts,
        decision_mid_ticks=decision_mid,
        decision_horizon_mid_ticks=decision_future_mid,
        candidate_price_ticks=candidate_prices,
    )


def _unconditional_quote_metrics(
    quote_specs: Sequence[QuoteLabelSpec],
    labels: Mapping[str, np.ndarray],
    masks: Mapping[str, np.ndarray],
) -> dict[str, dict[str, object]]:
    out: dict[str, dict[str, object]] = {}
    for spec in quote_specs:
        fill_rate = _mean_or_none(labels[spec.filled], masks[spec.filled])
        toxic_cost = _mean_or_none(labels[spec.toxic_cost_bps], masks[spec.toxic_cost_bps])
        out[spec.output_name] = {
            "fill_rate": fill_rate,
            "toxic_cost_bps_mean": toxic_cost,
        }
    return out


def _axis_summary(
    *,
    axis_name: str,
    axis_values: np.ndarray,
    percentiles: Sequence[int],
    quote_specs: Sequence[QuoteLabelSpec],
    labels: Mapping[str, np.ndarray],
    masks: Mapping[str, np.ndarray],
    markout: MarkoutContext,
    markout_index: ExecutionTapeMarkoutIndex | None,
    decision_horizon_us: int,
    fill_plus_horizon_us: int,
    correctness_deadband_bps: float,
    maker_fee_bps: float,
    unconditional: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    thresholds: dict[str, float] = {}
    for percentile in percentiles:
        thresholds[f"bottom_{percentile}"] = float(np.percentile(axis_values, percentile))
    for percentile in reversed(percentiles):
        thresholds[f"top_{percentile}"] = float(np.percentile(axis_values, 100 - percentile))

    buckets: dict[str, object] = {}
    for bucket_name, threshold in thresholds.items():
        if bucket_name.startswith("bottom_"):
            bucket_mask = axis_values <= threshold
        else:
            bucket_mask = axis_values >= threshold
        bucket_axis = axis_values[bucket_mask]
        direction_masks = _direction_masks(
            bucket_name=bucket_name,
            bucket_mask=bucket_mask,
            markout=markout,
            correctness_deadband_bps=correctness_deadband_bps,
        )
        quotes = {
            spec.output_name: _quote_metrics(
                spec=spec,
                bucket_mask=bucket_mask,
                labels=labels,
                masks=masks,
                markout=markout,
                markout_index=markout_index,
                direction_masks=direction_masks,
                decision_horizon_us=decision_horizon_us,
                fill_plus_horizon_us=fill_plus_horizon_us,
                maker_fee_bps=maker_fee_bps,
                unconditional=unconditional[spec.output_name],
            )
            for spec in quote_specs
        }
        buckets[bucket_name] = {
            "count": int(bucket_axis.size),
            "fraction": float(bucket_axis.size / max(axis_values.size, 1)),
            "axis_min": _min_or_none(bucket_axis),
            "axis_mean": _mean_or_none(bucket_axis),
            "axis_max": _max_or_none(bucket_axis),
            "realized_direction": dict(direction_masks.metrics),
            "quotes": quotes,
        }

    return {
        "description": _AXIS_DESCRIPTIONS[axis_name],
        "thresholds": thresholds,
        "unconditional": _axis_distribution(axis_values),
        "buckets": buckets,
        "directional_summary": {
            name: _directional_summary_for_bucket(name, bucket)
            for name, bucket in buckets.items()
        },
    }


def _direction_masks(
    *,
    bucket_name: str,
    bucket_mask: np.ndarray,
    markout: MarkoutContext,
    correctness_deadband_bps: float,
) -> DirectionMasks:
    realized = _realized_return_bps(markout)
    available = bucket_mask & np.isfinite(realized)
    if bucket_name.startswith("top_"):
        correct = available & (realized > correctness_deadband_bps)
        wrong = available & (realized < -correctness_deadband_bps)
    else:
        correct = available & (realized < -correctness_deadband_bps)
        wrong = available & (realized > correctness_deadband_bps)
    no_move = available & ~(correct | wrong)
    available_count = int(np.count_nonzero(available))
    bucket_count = int(np.count_nonzero(bucket_mask))
    metrics = {
        "realized_return_bps_mean": _mean_or_none(realized, available),
        "realized_return_bps_p10": _percentile_or_none(realized, available, 10),
        "realized_return_bps_p50": _percentile_or_none(realized, available, 50),
        "realized_return_bps_p90": _percentile_or_none(realized, available, 90),
        "correct_count": int(np.count_nonzero(correct)),
        "wrong_count": int(np.count_nonzero(wrong)),
        "no_move_count": int(np.count_nonzero(no_move)),
        "correct_rate": _rate(int(np.count_nonzero(correct)), available_count),
        "wrong_rate": _rate(int(np.count_nonzero(wrong)), available_count),
        "no_move_rate": _rate(int(np.count_nonzero(no_move)), available_count),
        "available_count": available_count,
        "available_rate": _rate(available_count, bucket_count),
    }
    return DirectionMasks(
        available=available,
        correct=correct,
        wrong=wrong,
        no_move=no_move,
        metrics=metrics,
    )


def _realized_return_bps(markout: MarkoutContext) -> np.ndarray:
    out = np.full(markout.decision_mid_ticks.shape, np.nan, dtype=np.float64)
    valid = (
        np.isfinite(markout.decision_mid_ticks)
        & (markout.decision_mid_ticks > 0.0)
        & np.isfinite(markout.decision_horizon_mid_ticks)
    )
    out[valid] = (
        (markout.decision_horizon_mid_ticks[valid] - markout.decision_mid_ticks[valid])
        / markout.decision_mid_ticks[valid]
        * 10_000.0
    )
    return out


def _quote_metrics(
    *,
    spec: QuoteLabelSpec,
    bucket_mask: np.ndarray,
    labels: Mapping[str, np.ndarray],
    masks: Mapping[str, np.ndarray],
    markout: MarkoutContext,
    markout_index: ExecutionTapeMarkoutIndex | None,
    direction_masks: DirectionMasks,
    decision_horizon_us: int,
    fill_plus_horizon_us: int,
    maker_fee_bps: float,
    unconditional: Mapping[str, object],
) -> dict[str, object]:
    count = int(np.count_nonzero(bucket_mask))
    if count == 0:
        return {
            "count": 0,
            "label_valid_count": 0,
            "fill_count": 0,
            "fill_rate": None,
            "fill_rate_lift_vs_unconditional": None,
            "fill_latency_us_mean": None,
            "fill_latency_us_p50": None,
            "fill_latency_us_p90": None,
            "adverse_bps_mean": None,
            "adverse_bps_p50": None,
            "adverse_bps_p90": None,
            "toxic_fill_rate": None,
            "toxic_cost_bps_mean": None,
            "unconditional_toxic_cost_bps_mean": None,
            "fill_selection": _empty_fill_selection_metrics(),
            "decision_horizon_markout": _empty_decision_horizon_markout_metrics(),
            "fill_plus_horizon_markout": _empty_fill_plus_markout_metrics(),
        }

    filled_values = labels[spec.filled]
    filled_valid = bucket_mask & masks[spec.filled] & np.isfinite(filled_values)
    fill_count = int(np.count_nonzero((filled_values >= 0.5) & filled_valid))
    fill_rate = _mean_or_none(filled_values, filled_valid)
    unconditional_fill = unconditional.get("fill_rate")
    lift = None if fill_rate is None or unconditional_fill is None else float(fill_rate - float(unconditional_fill))

    filled_rows = bucket_mask & masks[spec.filled] & (filled_values >= 0.5)
    latency_values = labels[spec.fill_latency_us]
    latency_mask = filled_rows & masks[spec.fill_latency_us] & np.isfinite(latency_values) & (latency_values > 0.0)
    adverse_values = labels[spec.adverse_bps]
    adverse_mask = filled_rows & masks[spec.adverse_bps] & np.isfinite(adverse_values)
    toxic_values = labels[spec.toxic_fill]
    toxic_mask = bucket_mask & masks[spec.toxic_fill] & np.isfinite(toxic_values)
    toxic_cost_values = labels[spec.toxic_cost_bps]
    toxic_cost_mask = bucket_mask & masks[spec.toxic_cost_bps] & np.isfinite(toxic_cost_values)
    label_valid_count = int(np.count_nonzero(filled_valid))
    fill_selection = _fill_selection_metrics(
        filled_values=filled_values,
        filled_valid=filled_valid,
        filled_rows=filled_rows,
        direction_masks=direction_masks,
    )
    decision_markout = _decision_horizon_markout_metrics(
        spec=spec,
        labels=labels,
        masks=masks,
        filled_rows=filled_rows,
        filled_valid=filled_valid,
        fill_count=fill_count,
        label_valid_count=label_valid_count,
        markout=markout,
        decision_horizon_us=decision_horizon_us,
        maker_fee_bps=maker_fee_bps,
    )
    fill_plus_markout = _fill_plus_horizon_markout_metrics(
        spec=spec,
        labels=labels,
        masks=masks,
        filled_rows=filled_rows,
        filled_valid=filled_valid,
        fill_count=fill_count,
        label_valid_count=label_valid_count,
        markout=markout,
        markout_index=markout_index,
        fill_plus_horizon_us=fill_plus_horizon_us,
        maker_fee_bps=maker_fee_bps,
    )
    return {
        "count": count,
        "label_valid_count": label_valid_count,
        "fill_count": fill_count,
        "fill_rate": fill_rate,
        "fill_rate_lift_vs_unconditional": lift,
        "fill_latency_us_mean": _mean_or_none(latency_values, latency_mask),
        "fill_latency_us_p50": _percentile_or_none(latency_values, latency_mask, 50),
        "fill_latency_us_p90": _percentile_or_none(latency_values, latency_mask, 90),
        "adverse_bps_mean": _mean_or_none(adverse_values, adverse_mask),
        "adverse_bps_p50": _percentile_or_none(adverse_values, adverse_mask, 50),
        "adverse_bps_p90": _percentile_or_none(adverse_values, adverse_mask, 90),
        "toxic_fill_rate": _mean_or_none(toxic_values, toxic_mask),
        "toxic_cost_bps_mean": _mean_or_none(toxic_cost_values, toxic_cost_mask),
        "unconditional_toxic_cost_bps_mean": unconditional.get("toxic_cost_bps_mean"),
        "fill_selection": fill_selection,
        "decision_horizon_markout": decision_markout,
        "fill_plus_horizon_markout": fill_plus_markout,
    }


def _empty_fill_selection_metrics() -> dict[str, object]:
    return {
        "fill_rate_given_correct": None,
        "fill_rate_given_wrong": None,
        "fill_rate_given_no_move": None,
        "correct_rate_given_fill": None,
        "wrong_rate_given_fill": None,
        "no_move_rate_given_fill": None,
        "correct_selection_lift": None,
        "wrong_selection_lift": None,
    }


def _fill_selection_metrics(
    *,
    filled_values: np.ndarray,
    filled_valid: np.ndarray,
    filled_rows: np.ndarray,
    direction_masks: DirectionMasks,
) -> dict[str, object]:
    correct_den = int(np.count_nonzero(direction_masks.correct & filled_valid))
    wrong_den = int(np.count_nonzero(direction_masks.wrong & filled_valid))
    no_move_den = int(np.count_nonzero(direction_masks.no_move & filled_valid))
    available_fills = filled_rows & direction_masks.available
    fill_den = int(np.count_nonzero(available_fills))
    correct_fill_count = int(np.count_nonzero(filled_rows & direction_masks.correct))
    wrong_fill_count = int(np.count_nonzero(filled_rows & direction_masks.wrong))
    no_move_fill_count = int(np.count_nonzero(filled_rows & direction_masks.no_move))
    correct_rate_given_fill = _rate(correct_fill_count, fill_den)
    wrong_rate_given_fill = _rate(wrong_fill_count, fill_den)
    no_move_rate_given_fill = _rate(no_move_fill_count, fill_den)
    bucket_correct_rate = direction_masks.metrics.get("correct_rate")
    bucket_wrong_rate = direction_masks.metrics.get("wrong_rate")
    return {
        "fill_rate_given_correct": _rate(correct_fill_count, correct_den),
        "fill_rate_given_wrong": _rate(wrong_fill_count, wrong_den),
        "fill_rate_given_no_move": _rate(no_move_fill_count, no_move_den),
        "correct_rate_given_fill": correct_rate_given_fill,
        "wrong_rate_given_fill": wrong_rate_given_fill,
        "no_move_rate_given_fill": no_move_rate_given_fill,
        "correct_selection_lift": (
            None
            if correct_rate_given_fill is None or bucket_correct_rate is None
            else float(correct_rate_given_fill - float(bucket_correct_rate))
        ),
        "wrong_selection_lift": (
            None
            if wrong_rate_given_fill is None or bucket_wrong_rate is None
            else float(wrong_rate_given_fill - float(bucket_wrong_rate))
        ),
    }


def _empty_decision_horizon_markout_metrics() -> dict[str, object]:
    return {
        "filled_with_markout_count": 0,
        "fill_after_decision_horizon_count": 0,
        "markout_available_rate_on_fills": None,
        "signed_markout_bps_mean": None,
        "signed_markout_bps_p10": None,
        "signed_markout_bps_p50": None,
        "signed_markout_bps_p90": None,
        "net_markout_bps_mean": None,
        "net_markout_bps_p10": None,
        "net_markout_bps_p50": None,
        "net_markout_bps_p90": None,
        "attempt_net_markout_bps_mean": None,
    }


def _empty_fill_plus_markout_metrics() -> dict[str, object]:
    return {
        "filled_with_markout_count": 0,
        "markout_available_rate_on_fills": None,
        "signed_markout_bps_mean": None,
        "signed_markout_bps_p10": None,
        "signed_markout_bps_p50": None,
        "signed_markout_bps_p90": None,
        "net_markout_bps_mean": None,
        "net_markout_bps_p10": None,
        "net_markout_bps_p50": None,
        "net_markout_bps_p90": None,
        "attempt_net_markout_bps_mean": None,
    }


def _decision_horizon_markout_metrics(
    *,
    spec: QuoteLabelSpec,
    labels: Mapping[str, np.ndarray],
    masks: Mapping[str, np.ndarray],
    filled_rows: np.ndarray,
    filled_valid: np.ndarray,
    fill_count: int,
    label_valid_count: int,
    markout: MarkoutContext,
    decision_horizon_us: int,
    maker_fee_bps: float,
) -> dict[str, object]:
    if label_valid_count == 0:
        return _empty_decision_horizon_markout_metrics()
    latency = labels[spec.fill_latency_us]
    latency_valid = filled_rows & masks[spec.fill_latency_us] & np.isfinite(latency) & (latency >= 0.0)
    after_horizon = latency_valid & (latency > float(decision_horizon_us))
    price = np.asarray(markout.candidate_price_ticks.get(spec.output_name), dtype=np.float64)
    future_mid = markout.decision_horizon_mid_ticks
    valid = (
        latency_valid
        & ~after_horizon
        & np.isfinite(price)
        & (price > 0.0)
        & np.isfinite(future_mid)
    )
    signed = _signed_markout_bps(spec.output_name, future_mid, price)
    return _markout_distribution(
        signed=signed,
        valid=valid,
        fill_count=fill_count,
        label_valid_count=label_valid_count,
        maker_fee_bps=maker_fee_bps,
        fill_after_decision_horizon_count=int(np.count_nonzero(after_horizon)),
    )


def _fill_plus_horizon_markout_metrics(
    *,
    spec: QuoteLabelSpec,
    labels: Mapping[str, np.ndarray],
    masks: Mapping[str, np.ndarray],
    filled_rows: np.ndarray,
    filled_valid: np.ndarray,
    fill_count: int,
    label_valid_count: int,
    markout: MarkoutContext,
    markout_index: ExecutionTapeMarkoutIndex | None,
    fill_plus_horizon_us: int,
    maker_fee_bps: float,
) -> dict[str, object]:
    if label_valid_count == 0:
        return _empty_fill_plus_markout_metrics()
    latency = labels[spec.fill_latency_us]
    latency_valid = filled_rows & masks[spec.fill_latency_us] & np.isfinite(latency) & (latency >= 0.0)
    future_mid = np.full(latency.shape, np.nan, dtype=np.float64)
    if markout_index is not None and np.any(latency_valid):
        idx = np.flatnonzero(latency_valid)
        targets = (
            markout.decision_local_ts_us[idx].astype(np.float64)
            + latency[idx].astype(np.float64)
            + float(fill_plus_horizon_us)
        ).astype(np.int64)
        future_mid[idx] = markout_index.future_mid_ticks_at_or_after(targets)
    price = np.asarray(markout.candidate_price_ticks.get(spec.output_name), dtype=np.float64)
    valid = latency_valid & np.isfinite(price) & (price > 0.0) & np.isfinite(future_mid)
    signed = _signed_markout_bps(spec.output_name, future_mid, price)
    summary = _markout_distribution(
        signed=signed,
        valid=valid,
        fill_count=fill_count,
        label_valid_count=label_valid_count,
        maker_fee_bps=maker_fee_bps,
    )
    summary.pop("fill_after_decision_horizon_count", None)
    return summary


def _signed_markout_bps(quote_name: str, future_mid_ticks: np.ndarray, price_ticks: np.ndarray) -> np.ndarray:
    out = np.full(price_ticks.shape, np.nan, dtype=np.float64)
    valid = np.isfinite(future_mid_ticks) & np.isfinite(price_ticks) & (price_ticks > 0.0)
    if quote_name.startswith("bid_"):
        out[valid] = (future_mid_ticks[valid] - price_ticks[valid]) / price_ticks[valid] * 10_000.0
    elif quote_name.startswith("ask_"):
        out[valid] = (price_ticks[valid] - future_mid_ticks[valid]) / price_ticks[valid] * 10_000.0
    else:
        raise ValueError(f"invalid quote name {quote_name!r}")
    return out


def _markout_distribution(
    *,
    signed: np.ndarray,
    valid: np.ndarray,
    fill_count: int,
    label_valid_count: int,
    maker_fee_bps: float,
    fill_after_decision_horizon_count: int | None = None,
) -> dict[str, object]:
    net = signed - maker_fee_bps
    valid_count = int(np.count_nonzero(valid))
    selected_signed = signed[valid]
    selected_net = net[valid]
    out = {
        "filled_with_markout_count": valid_count,
        "markout_available_rate_on_fills": _rate(valid_count, fill_count),
        "signed_markout_bps_mean": _mean_or_none(selected_signed),
        "signed_markout_bps_p10": _percentile_or_none(selected_signed, None, 10),
        "signed_markout_bps_p50": _percentile_or_none(selected_signed, None, 50),
        "signed_markout_bps_p90": _percentile_or_none(selected_signed, None, 90),
        "net_markout_bps_mean": _mean_or_none(selected_net),
        "net_markout_bps_p10": _percentile_or_none(selected_net, None, 10),
        "net_markout_bps_p50": _percentile_or_none(selected_net, None, 50),
        "net_markout_bps_p90": _percentile_or_none(selected_net, None, 90),
        "attempt_net_markout_bps_mean": (
            None
            if label_valid_count <= 0
            else float(np.sum(selected_net, dtype=np.float64) / float(label_valid_count))
        ),
    }
    if fill_after_decision_horizon_count is not None:
        out["fill_after_decision_horizon_count"] = fill_after_decision_horizon_count
    return out


def _axis_distribution(values: np.ndarray) -> dict[str, object]:
    finite = values[np.isfinite(values)]
    return {
        "count": int(values.size),
        "axis_mean": _mean_or_none(finite),
        "axis_std": _std_or_none(finite),
        "axis_p01": _percentile_or_none(finite, None, 1),
        "axis_p10": _percentile_or_none(finite, None, 10),
        "axis_p50": _percentile_or_none(finite, None, 50),
        "axis_p90": _percentile_or_none(finite, None, 90),
        "axis_p99": _percentile_or_none(finite, None, 99),
    }


def _directional_summary_for_bucket(bucket_name: str, bucket: Mapping[str, object]) -> dict[str, object]:
    quotes = bucket["quotes"]
    if not isinstance(quotes, Mapping):
        return {}
    bid_touch = quotes.get("bid_touch", {})
    ask_touch = quotes.get("ask_touch", {})
    if not isinstance(bid_touch, Mapping):
        bid_touch = {}
    if not isinstance(ask_touch, Mapping):
        ask_touch = {}
    realized = bucket.get("realized_direction")
    if not isinstance(realized, Mapping):
        realized = {}
    if bucket_name.startswith("top_"):
        summary = {
            "bucket_role": "bullish",
            "bullish_bid_touch_fill_rate": bid_touch.get("fill_rate"),
            "bullish_bid_touch_fill_lift": bid_touch.get("fill_rate_lift_vs_unconditional"),
            "bullish_bid_touch_toxic_cost_bps_mean": bid_touch.get("toxic_cost_bps_mean"),
            "bullish_bid_touch_adverse_bps_mean": bid_touch.get("adverse_bps_mean"),
            "opposite_side_check": {
                "bullish_ask_touch_fill_rate": ask_touch.get("fill_rate"),
            },
        }
        summary.update(_directional_summary_additions(realized, desired=bid_touch, opposite=ask_touch, desired_name="bid_touch", opposite_name="ask_touch"))
        return summary
    summary = {
        "bucket_role": "bearish",
        "bearish_ask_touch_fill_rate": ask_touch.get("fill_rate"),
        "bearish_ask_touch_fill_lift": ask_touch.get("fill_rate_lift_vs_unconditional"),
        "bearish_ask_touch_toxic_cost_bps_mean": ask_touch.get("toxic_cost_bps_mean"),
        "bearish_ask_touch_adverse_bps_mean": ask_touch.get("adverse_bps_mean"),
        "opposite_side_check": {
            "bearish_bid_touch_fill_rate": bid_touch.get("fill_rate"),
        },
    }
    summary.update(_directional_summary_additions(realized, desired=ask_touch, opposite=bid_touch, desired_name="ask_touch", opposite_name="bid_touch"))
    return summary


def _directional_summary_additions(
    realized: Mapping[str, object],
    *,
    desired: Mapping[str, object],
    opposite: Mapping[str, object],
    desired_name: str,
    opposite_name: str,
) -> dict[str, object]:
    desired_selection = _mapping(desired.get("fill_selection"))
    opposite_selection = _mapping(opposite.get("fill_selection"))
    desired_decision = _mapping(desired.get("decision_horizon_markout"))
    desired_fill_plus = _mapping(desired.get("fill_plus_horizon_markout"))
    opposite_decision = _mapping(opposite.get("decision_horizon_markout"))
    opposite_fill_plus = _mapping(opposite.get("fill_plus_horizon_markout"))
    return {
        "prediction_correct_rate": realized.get("correct_rate"),
        "prediction_wrong_rate": realized.get("wrong_rate"),
        "prediction_no_move_rate": realized.get("no_move_rate"),
        "desired_touch": {
            "quote_name": desired_name,
            "fill_rate": desired.get("fill_rate"),
            "fill_lift_vs_unconditional": desired.get("fill_rate_lift_vs_unconditional"),
            "fill_rate_given_correct": desired_selection.get("fill_rate_given_correct"),
            "fill_rate_given_wrong": desired_selection.get("fill_rate_given_wrong"),
            "fill_rate_given_no_move": desired_selection.get("fill_rate_given_no_move"),
            "correct_rate_given_fill": desired_selection.get("correct_rate_given_fill"),
            "wrong_rate_given_fill": desired_selection.get("wrong_rate_given_fill"),
            "no_move_rate_given_fill": desired_selection.get("no_move_rate_given_fill"),
            "correct_selection_lift": desired_selection.get("correct_selection_lift"),
            "wrong_selection_lift": desired_selection.get("wrong_selection_lift"),
            "decision_horizon_attempt_net_bps_mean": desired_decision.get("attempt_net_markout_bps_mean"),
            "decision_horizon_net_bps_mean_given_fill": desired_decision.get("net_markout_bps_mean"),
            "fill_plus_horizon_attempt_net_bps_mean": desired_fill_plus.get("attempt_net_markout_bps_mean"),
            "fill_plus_horizon_net_bps_mean_given_fill": desired_fill_plus.get("net_markout_bps_mean"),
        },
        "opposite_touch": {
            "quote_name": opposite_name,
            "fill_rate": opposite.get("fill_rate"),
            "fill_lift_vs_unconditional": opposite.get("fill_rate_lift_vs_unconditional"),
            "correct_rate_given_fill": opposite_selection.get("correct_rate_given_fill"),
            "wrong_rate_given_fill": opposite_selection.get("wrong_rate_given_fill"),
            "no_move_rate_given_fill": opposite_selection.get("no_move_rate_given_fill"),
            "decision_horizon_attempt_net_bps_mean": opposite_decision.get("attempt_net_markout_bps_mean"),
            "fill_plus_horizon_attempt_net_bps_mean": opposite_fill_plus.get("attempt_net_markout_bps_mean"),
        },
    }


def _mapping(value: object) -> Mapping[str, object]:
    return value if isinstance(value, Mapping) else {}


def _compact_summary(
    axes: Mapping[str, Mapping[str, object]],
    percentiles: Sequence[int],
) -> dict[str, object]:
    compact: dict[str, object] = {}
    for axis_name, axis_summary in axes.items():
        buckets = axis_summary["buckets"]
        if not isinstance(buckets, Mapping):
            continue
        for percentile in percentiles:
            for side_name, bucket_name, quote_name in (
                ("top", f"top_{percentile}", "bid_touch"),
                ("bottom", f"bottom_{percentile}", "ask_touch"),
            ):
                bucket = buckets.get(bucket_name)
                if not isinstance(bucket, Mapping):
                    continue
                quotes = bucket.get("quotes")
                if not isinstance(quotes, Mapping):
                    continue
                quote = quotes.get(quote_name)
                if not isinstance(quote, Mapping):
                    continue
                prefix = f"{axis_name}_{side_name}{percentile}_{quote_name}"
                compact[f"{prefix}_fill_rate"] = quote.get("fill_rate")
                compact[f"{prefix}_fill_lift"] = quote.get("fill_rate_lift_vs_unconditional")
                selection = _mapping(quote.get("fill_selection"))
                decision = _mapping(quote.get("decision_horizon_markout"))
                fill_plus = _mapping(quote.get("fill_plus_horizon_markout"))
                compact[f"{prefix}_wrong_rate_given_fill"] = selection.get("wrong_rate_given_fill")
                compact[f"{prefix}_wrong_selection_lift"] = selection.get("wrong_selection_lift")
                compact[f"{prefix}_decision_horizon_attempt_net_bps_mean"] = decision.get("attempt_net_markout_bps_mean")
                compact[f"{prefix}_fill_plus_horizon_attempt_net_bps_mean"] = fill_plus.get("attempt_net_markout_bps_mean")
    return compact


def _alpha_warnings(
    *,
    axes: Mapping[str, Mapping[str, object]],
    rows_sampled: int,
    percentiles: Sequence[int],
    unconditional: Mapping[str, Mapping[str, object]],
) -> list[str]:
    warnings: list[str] = []
    if rows_sampled < 10_000:
        warnings.append(f"rows_sampled below 10000: {rows_sampled}")
    for axis_name, axis_summary in axes.items():
        buckets = axis_summary.get("buckets")
        if not isinstance(buckets, Mapping):
            continue
        counts = [
            int(bucket.get("count", 0))
            for bucket in buckets.values()
            if isinstance(bucket, Mapping)
        ]
        if counts and min(counts) < 1_000:
            warnings.append(f"{axis_name} has alpha bucket count below 1000: min_count={min(counts)}")

    direction = axes.get("direction_score", {})
    expected = axes.get("expected_return_bps", {})
    top10_direction_bid = _bucket_quote_metric(direction, "top_10", "bid_touch", "fill_rate_lift_vs_unconditional")
    bottom10_direction_ask = _bucket_quote_metric(direction, "bottom_10", "ask_touch", "fill_rate_lift_vs_unconditional")
    if top10_direction_bid is not None and top10_direction_bid <= 0.0:
        warnings.append("direction_score top_10 bid_touch fill lift is nonpositive")
    if bottom10_direction_ask is not None and bottom10_direction_ask <= 0.0:
        warnings.append("direction_score bottom_10 ask_touch fill lift is nonpositive")

    top10_expected_bid = _bucket_quote_metric(expected, "top_10", "bid_touch", "fill_rate_lift_vs_unconditional")
    bottom10_expected_ask = _bucket_quote_metric(expected, "bottom_10", "ask_touch", "fill_rate_lift_vs_unconditional")
    if _sign_disagrees(top10_direction_bid, top10_expected_bid):
        warnings.append("expected_return_bps top_10 bid_touch actionability disagrees with direction_score")
    if _sign_disagrees(bottom10_direction_ask, bottom10_expected_ask):
        warnings.append("expected_return_bps bottom_10 ask_touch actionability disagrees with direction_score")

    bid_touch_fill = unconditional.get("bid_touch", {}).get("fill_rate")
    ask_touch_fill = unconditional.get("ask_touch", {}).get("fill_rate")
    if (
        bid_touch_fill is not None
        and ask_touch_fill is not None
        and float(bid_touch_fill) < 0.02
        and float(ask_touch_fill) < 0.02
    ):
        warnings.append("bid_touch and ask_touch unconditional fill rates are below 0.02")

    signed = axes.get("signed_move_prob", {})
    for bucket_name, quote_name, label in (
        ("top_10", "bid_touch", "signed_move_prob top_10 bid_touch"),
        ("bottom_10", "ask_touch", "signed_move_prob bottom_10 ask_touch"),
    ):
        wrong_lift = _bucket_quote_nested_metric(signed, bucket_name, quote_name, "fill_selection", "wrong_selection_lift")
        if wrong_lift is not None and wrong_lift > 0.05:
            warnings.append(f"{label} fills are selected toward wrong predictions: wrong_selection_lift={wrong_lift:.6g}")
        decision_attempt = _bucket_quote_nested_metric(
            signed,
            bucket_name,
            quote_name,
            "decision_horizon_markout",
            "attempt_net_markout_bps_mean",
        )
        if decision_attempt is not None and decision_attempt <= 0.0:
            warnings.append(f"{label} decision-horizon attempt net markout is nonpositive: {decision_attempt:.6g}")
        decision_available = _bucket_quote_nested_metric(
            signed,
            bucket_name,
            quote_name,
            "decision_horizon_markout",
            "markout_available_rate_on_fills",
        )
        if decision_available is not None and decision_available < 0.8:
            warnings.append(f"{label} decision-horizon markout available rate on fills is below 0.8: {decision_available:.6g}")
        fill_plus_attempt = _bucket_quote_nested_metric(
            signed,
            bucket_name,
            quote_name,
            "fill_plus_horizon_markout",
            "attempt_net_markout_bps_mean",
        )
        if decision_attempt is not None and fill_plus_attempt is not None and decision_attempt > 0.0 and fill_plus_attempt <= 0.0:
            warnings.append(f"{label} fill-plus attempt net markout disagrees with positive decision-horizon markout")
    return warnings


def _bucket_quote_metric(
    axis_summary: Mapping[str, object],
    bucket_name: str,
    quote_name: str,
    metric_name: str,
) -> float | None:
    buckets = axis_summary.get("buckets")
    if not isinstance(buckets, Mapping):
        return None
    bucket = buckets.get(bucket_name)
    if not isinstance(bucket, Mapping):
        return None
    quotes = bucket.get("quotes")
    if not isinstance(quotes, Mapping):
        return None
    quote = quotes.get(quote_name)
    if not isinstance(quote, Mapping):
        return None
    value = quote.get(metric_name)
    return None if value is None else float(value)


def _bucket_quote_nested_metric(
    axis_summary: Mapping[str, object],
    bucket_name: str,
    quote_name: str,
    container_name: str,
    metric_name: str,
) -> float | None:
    buckets = axis_summary.get("buckets")
    if not isinstance(buckets, Mapping):
        return None
    bucket = buckets.get(bucket_name)
    if not isinstance(bucket, Mapping):
        return None
    quotes = bucket.get("quotes")
    if not isinstance(quotes, Mapping):
        return None
    quote = quotes.get(quote_name)
    if not isinstance(quote, Mapping):
        return None
    container = quote.get(container_name)
    if not isinstance(container, Mapping):
        return None
    value = container.get(metric_name)
    return None if value is None else float(value)


def _sign_disagrees(left: float | None, right: float | None) -> bool:
    if left is None or right is None:
        return False
    return (left > 0.0) != (right > 0.0)


def _rate(numerator: int, denominator: int) -> float | None:
    if denominator <= 0:
        return None
    return float(numerator / denominator)


def _mean_or_none(values: np.ndarray, mask: np.ndarray | None = None) -> float | None:
    selected = values if mask is None else values[mask]
    selected = selected[np.isfinite(selected)]
    if selected.size == 0:
        return None
    return float(np.mean(selected))


def _std_or_none(values: np.ndarray) -> float | None:
    selected = values[np.isfinite(values)]
    if selected.size == 0:
        return None
    return float(np.std(selected))


def _min_or_none(values: np.ndarray) -> float | None:
    selected = values[np.isfinite(values)]
    if selected.size == 0:
        return None
    return float(np.min(selected))


def _max_or_none(values: np.ndarray) -> float | None:
    selected = values[np.isfinite(values)]
    if selected.size == 0:
        return None
    return float(np.max(selected))


def _percentile_or_none(values: np.ndarray, mask: np.ndarray | None, percentile: float) -> float | None:
    selected = values if mask is None else values[mask]
    selected = selected[np.isfinite(selected)]
    if selected.size == 0:
        return None
    return float(np.percentile(selected, percentile))
