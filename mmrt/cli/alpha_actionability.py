"""Empirical alpha actionability diagnostics for execution profiling."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from mmrt.execution.adverse_selection_dataset import (
    ADVERSE_SELECTION_DATASET_SCHEMA,
    DiskBackedAdverseSelectionDataset,
    load_adverse_selection_dataset,
)
from mmrt.execution.linear_signal import LinearSignalArtifact
from mmrt.execution.split_contract import DecisionSplitRange, ranges_for_split, validate_split_contract_payload


ALPHA_ACTIONABILITY_SOURCE = "empirical_adverse_dataset_labels"
DEFAULT_ALPHA_ACTIONABILITY_PERCENTILES = (10, 20)
DEFAULT_ALPHA_ACTIONABILITY_MAX_ROWS = 1_000_000
DEFAULT_ALPHA_ACTIONABILITY_RANDOM_SEED = 123

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
    max_rows: int = DEFAULT_ALPHA_ACTIONABILITY_MAX_ROWS,
    percentiles: str | Sequence[int] = DEFAULT_ALPHA_ACTIONABILITY_PERCENTILES,
    seed: int = DEFAULT_ALPHA_ACTIONABILITY_RANDOM_SEED,
    chunk_rows: int = 100_000,
) -> dict[str, object]:
    """Compute empirical maker-fill actionability by alpha tail buckets."""

    if not isinstance(linear_signals, LinearSignalArtifact):
        raise ValueError("linear_signals must be LinearSignalArtifact")
    max_rows = _positive_int(max_rows, "alpha_actionability_max_rows")
    seed = _nonnegative_int(seed, "alpha_actionability_random_seed")
    chunk_rows = _positive_int(chunk_rows, "chunk_rows")
    percentiles_tuple = parse_alpha_actionability_percentiles(percentiles)
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
    unconditional = _unconditional_quote_metrics(quote_specs, labels, masks)
    axis_summaries = {
        axis_name: _axis_summary(
            axis_name=axis_name,
            axis_values=axis_values,
            percentiles=percentiles_tuple,
            quote_specs=quote_specs,
            labels=labels,
            masks=masks,
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
        quotes = {
            spec.output_name: _quote_metrics(
                spec=spec,
                bucket_mask=bucket_mask,
                labels=labels,
                masks=masks,
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


def _quote_metrics(
    *,
    spec: QuoteLabelSpec,
    bucket_mask: np.ndarray,
    labels: Mapping[str, np.ndarray],
    masks: Mapping[str, np.ndarray],
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
    return {
        "count": count,
        "label_valid_count": int(np.count_nonzero(filled_valid)),
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
    }


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
    if bucket_name.startswith("top_"):
        return {
            "bucket_role": "bullish",
            "bullish_bid_touch_fill_rate": bid_touch.get("fill_rate"),
            "bullish_bid_touch_fill_lift": bid_touch.get("fill_rate_lift_vs_unconditional"),
            "bullish_bid_touch_toxic_cost_bps_mean": bid_touch.get("toxic_cost_bps_mean"),
            "bullish_bid_touch_adverse_bps_mean": bid_touch.get("adverse_bps_mean"),
            "opposite_side_check": {
                "bullish_ask_touch_fill_rate": ask_touch.get("fill_rate"),
            },
        }
    return {
        "bucket_role": "bearish",
        "bearish_ask_touch_fill_rate": ask_touch.get("fill_rate"),
        "bearish_ask_touch_fill_lift": ask_touch.get("fill_rate_lift_vs_unconditional"),
        "bearish_ask_touch_toxic_cost_bps_mean": ask_touch.get("toxic_cost_bps_mean"),
        "bearish_ask_touch_adverse_bps_mean": ask_touch.get("adverse_bps_mean"),
        "opposite_side_check": {
            "bearish_bid_touch_fill_rate": bid_touch.get("fill_rate"),
        },
    }


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


def _sign_disagrees(left: float | None, right: float | None) -> bool:
    if left is None or right is None:
        return False
    return (left > 0.0) != (right > 0.0)


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
