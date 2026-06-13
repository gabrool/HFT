"""Build aligned no-move-gated linear signal artifacts from an execution tape."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np

from mmrt.execution.execution_tape import ExecutionTapeValidationMode, load_execution_tape
from mmrt.execution.linear_signal import (
    LINEAR_SIGNALS_FILENAME,
    MAGNITUDE_INPUT_LOG1P_BPS,
    MAGNITUDE_INPUT_MODES,
    LINEAR_SIGNAL_ARTIFACT_SCHEMA,
    LinearSignalArtifactMetadata,
    LinearSignalConfig,
    linear_signal_array_fields,
    save_linear_signal_artifact_arrays,
)
from mmrt.execution.linear_signal_builder import (
    ExecutionLinearFeatureDataset,
    build_linear_signal_build_result,
    execution_linear_feature_names,
    iter_execution_linear_feature_chunks,
    schedule_config_from_train_result,
    transform_config_from_train_result,
)
from mmrt.linear.train import LINEAR_TRAINING_RESULT_SCHEMA, load_linear_train_result

__all__ = [
    "BuildLinearSignalsConfig",
    "build_linear_signals_from_config",
    "build_arg_parser",
    "main",
]


def _require_nonempty_str(value: str, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()


def _require_bool(value: bool, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be bool")
    return value


def _require_positive_int(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive int")
    return value


def _require_optional_nonnegative_int(value: int | None, name: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be None or a nonnegative int")
    return value


def _require_optional_positive_int(value: int | None, name: str) -> int | None:
    if value is None:
        return None
    return _require_positive_int(value, name)


@dataclass(frozen=True, slots=True)
class BuildLinearSignalsConfig:
    tape_root: str
    linear_train_result_json: str
    output_npz: str | None = None
    output_json: str | None = None
    overwrite: bool = False
    mmap_mode: str | None = "r"
    start_event_index: int | None = None
    max_decisions: int | None = None
    output_dtype: str = "float32"
    magnitude_input: str = MAGNITUDE_INPUT_LOG1P_BPS
    chunk_rows: int = 100_000

    def __post_init__(self) -> None:
        object.__setattr__(self, "tape_root", _require_nonempty_str(self.tape_root, "tape_root"))
        object.__setattr__(self, "linear_train_result_json", _require_nonempty_str(self.linear_train_result_json, "linear_train_result_json"))
        if self.output_npz is not None:
            object.__setattr__(self, "output_npz", _require_nonempty_str(self.output_npz, "output_npz"))
        if self.output_json is not None:
            object.__setattr__(self, "output_json", _require_nonempty_str(self.output_json, "output_json"))
        object.__setattr__(self, "overwrite", _require_bool(self.overwrite, "overwrite"))
        if self.mmap_mode not in (None, "r"):
            raise ValueError("mmap_mode must be None or 'r'")
        object.__setattr__(self, "start_event_index", _require_optional_nonnegative_int(self.start_event_index, "start_event_index"))
        object.__setattr__(self, "max_decisions", _require_optional_positive_int(self.max_decisions, "max_decisions"))
        if self.output_dtype not in ("float32", "float64"):
            raise ValueError("output_dtype must be 'float32' or 'float64'")
        if self.magnitude_input not in MAGNITUDE_INPUT_MODES:
            raise ValueError("magnitude_input must be a supported magnitude input mode")
        object.__setattr__(self, "chunk_rows", _require_positive_int(self.chunk_rows, "chunk_rows"))

    def as_dict(self) -> dict[str, object]:
        return {
            "tape_root": self.tape_root,
            "linear_train_result_json": self.linear_train_result_json,
            "output_npz": self.output_npz,
            "output_json": self.output_json,
            "overwrite": self.overwrite,
            "mmap_mode": self.mmap_mode,
            "start_event_index": self.start_event_index,
            "max_decisions": self.max_decisions,
            "output_dtype": self.output_dtype,
            "magnitude_input": self.magnitude_input,
            "chunk_rows": self.chunk_rows,
        }


def _default_output_npz(tape_root: str) -> Path:
    return Path(tape_root) / LINEAR_SIGNALS_FILENAME


def _default_output_json(tape_root: str) -> Path:
    return Path(tape_root) / "linear_signals_summary.json"


def _write_json_atomic(path: Path, payload: dict[str, object], *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"output_json already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


@dataclass(frozen=True, slots=True)
class _FeatureScan:
    num_decisions: int
    feature_names: tuple[str, ...]
    first_decision_event_index: int
    last_decision_event_index: int
    first_decision_local_ts_us: int
    last_decision_local_ts_us: int
    replay_start_event_index: int
    start_event_index: int
    decision_schedule: dict[str, object]
    transform_config: dict[str, object]

    def feature_dataset_summary(self) -> dict[str, object]:
        return {
            "num_decisions": self.num_decisions,
            "num_features": len(self.feature_names),
            "feature_names": list(self.feature_names),
            "first_decision_event_index": self.first_decision_event_index,
            "last_decision_event_index": self.last_decision_event_index,
            "first_decision_local_ts_us": self.first_decision_local_ts_us,
            "last_decision_local_ts_us": self.last_decision_local_ts_us,
            "decision_schedule": dict(self.decision_schedule),
            "replay_start_event_index": self.replay_start_event_index,
            "start_event_index": self.start_event_index,
            "transform_config": dict(self.transform_config),
        }


@dataclass(slots=True)
class _SignalWriters:
    n_rows: int
    decision_event_index: np.memmap
    decision_local_ts_us: np.memmap
    arrays: dict[str, np.memmap]
    temp_paths: tuple[Path, ...]
    row: int = 0

    @classmethod
    def create(cls, output_npz: Path, *, n_rows: int, output_dtype: str) -> "_SignalWriters":
        output_npz.parent.mkdir(parents=True, exist_ok=True)
        prefix = output_npz.parent / f".{output_npz.name}"
        event_path = Path(str(prefix) + ".decision_event_index.npy")
        ts_path = Path(str(prefix) + ".decision_local_ts_us.npy")
        signal_paths = {name: Path(str(prefix) + f".{name}.npy") for name in linear_signal_array_fields()}
        temp_paths = (event_path, ts_path, *signal_paths.values())
        for path in temp_paths:
            path.unlink(missing_ok=True)
        event_idx = np.lib.format.open_memmap(event_path, mode="w+", dtype=np.int64, shape=(n_rows,))
        local_ts = np.lib.format.open_memmap(ts_path, mode="w+", dtype=np.int64, shape=(n_rows,))
        arrays = {
            name: np.lib.format.open_memmap(path, mode="w+", dtype=np.dtype(output_dtype), shape=(n_rows,))
            for name, path in signal_paths.items()
        }
        return cls(n_rows=n_rows, decision_event_index=event_idx, decision_local_ts_us=local_ts, arrays=arrays, temp_paths=temp_paths)

    def append(self, dataset: ExecutionLinearFeatureDataset, arrays) -> None:
        n = int(arrays.n_rows)
        if dataset.num_decisions != n:
            raise ValueError("signal rows must match feature chunk rows")
        end = self.row + n
        if end > self.n_rows:
            raise ValueError("too many signal rows appended")
        self.decision_event_index[self.row:end] = dataset.decision_event_index
        self.decision_local_ts_us[self.row:end] = dataset.decision_local_ts_us
        for name in linear_signal_array_fields():
            self.arrays[name][self.row:end] = getattr(arrays, name)
        self.row = end

    def flush(self) -> None:
        self.decision_event_index.flush()
        self.decision_local_ts_us.flush()
        for arr in self.arrays.values():
            arr.flush()

    def close(self) -> None:
        for arr in (self.decision_event_index, self.decision_local_ts_us, *self.arrays.values()):
            mmap = getattr(arr, "_mmap", None)
            if mmap is not None:
                mmap.close()

    def cleanup(self) -> None:
        for path in self.temp_paths:
            path.unlink(missing_ok=True)


def _feature_dataset_from_chunk(
    chunk,
    *,
    replay_start_event_index: int,
    decision_schedule: dict[str, object],
    transform_config: dict[str, object],
) -> ExecutionLinearFeatureDataset:
    return ExecutionLinearFeatureDataset(
        decision_event_index=chunk.decision_event_index,
        decision_local_ts_us=chunk.decision_local_ts_us,
        features=chunk.features,
        feature_names=tuple(chunk.feature_names),
        replay_start_event_index=replay_start_event_index,
        start_event_index=int(chunk.decision_event_index[0]),
        decision_schedule=dict(decision_schedule),
        transform_config=dict(transform_config),
    )


def _scan_feature_chunks(
    tape,
    *,
    schedule,
    transform,
    start_event_index: int | None,
    max_decisions: int | None,
    chunk_rows: int,
    output_dtype: str,
) -> _FeatureScan:
    replay_start = 0 if start_event_index is None else int(start_event_index)
    schedule_payload = schedule.as_dict()
    transform_payload = transform.as_dict()
    expected_names = execution_linear_feature_names()
    total = 0
    first_event_index: int | None = None
    first_local_ts_us: int | None = None
    last_event_index: int | None = None
    last_local_ts_us: int | None = None
    for chunk in iter_execution_linear_feature_chunks(
        tape,
        schedule_config=schedule,
        start_event_index=start_event_index,
        max_decisions=max_decisions,
        chunk_rows=chunk_rows,
        output_dtype=output_dtype,
        transform_config=transform,
    ):
        dataset = _feature_dataset_from_chunk(
            chunk,
            replay_start_event_index=replay_start,
            decision_schedule=schedule_payload,
            transform_config=transform_payload,
        )
        if dataset.feature_names != expected_names:
            raise ValueError("execution feature names changed during chunk replay")
        if last_event_index is not None and int(dataset.decision_event_index[0]) <= last_event_index:
            raise ValueError("decision_event_index must be strictly increasing across chunks")
        if last_local_ts_us is not None and int(dataset.decision_local_ts_us[0]) <= last_local_ts_us:
            raise ValueError("decision_local_ts_us must be strictly increasing across chunks")
        if first_event_index is None:
            first_event_index = int(dataset.decision_event_index[0])
            first_local_ts_us = int(dataset.decision_local_ts_us[0])
        last_event_index = int(dataset.decision_event_index[-1])
        last_local_ts_us = int(dataset.decision_local_ts_us[-1])
        total += dataset.num_decisions
    if total <= 0:
        raise ValueError("feature_dataset must contain at least one decision")
    assert first_event_index is not None
    assert first_local_ts_us is not None
    assert last_event_index is not None
    assert last_local_ts_us is not None
    return _FeatureScan(
        num_decisions=total,
        feature_names=expected_names,
        first_decision_event_index=first_event_index,
        last_decision_event_index=last_event_index,
        first_decision_local_ts_us=first_local_ts_us,
        last_decision_local_ts_us=last_local_ts_us,
        replay_start_event_index=replay_start,
        start_event_index=first_event_index,
        decision_schedule=schedule_payload,
        transform_config=transform_payload,
    )


def _quantiles_disk_backed(arr: np.ndarray, *, chunk_rows: int, temp_path: Path) -> list[float]:
    n = int(arr.shape[0])
    tmp = np.lib.format.open_memmap(temp_path, mode="w+", dtype=np.float64, shape=(n,))
    try:
        for start in range(0, n, chunk_rows):
            end = min(start + chunk_rows, n)
            tmp[start:end] = np.asarray(arr[start:end], dtype=np.float64)
        tmp.sort()
        out: list[float] = []
        for q in (0.01, 0.50, 0.99):
            h = (n - 1) * q
            lo = int(np.floor(h))
            hi = int(np.ceil(h))
            out.append(float(tmp[lo]) if lo == hi else float((1.0 - (h - lo)) * tmp[lo] + (h - lo) * tmp[hi]))
        return out
    finally:
        tmp.flush()
        mmap = getattr(tmp, "_mmap", None)
        if mmap is not None:
            mmap.close()
        temp_path.unlink(missing_ok=True)


def _stats_chunked(arr: np.ndarray, *, include_std: bool, chunk_rows: int, quantile_temp_path: Path | None = None) -> dict[str, object]:
    n = int(arr.shape[0])
    total = 0.0
    min_value = float("inf")
    max_value = float("-inf")
    for start in range(0, n, chunk_rows):
        end = min(start + chunk_rows, n)
        chunk = np.asarray(arr[start:end], dtype=np.float64)
        total += float(np.sum(chunk, dtype=np.float64))
        min_value = min(min_value, float(np.min(chunk)))
        max_value = max(max_value, float(np.max(chunk)))
    mean = float(total / n)
    out = {"mean": mean, "min": min_value, "max": max_value}
    if include_std:
        ss = 0.0
        for start in range(0, n, chunk_rows):
            end = min(start + chunk_rows, n)
            centered = np.asarray(arr[start:end], dtype=np.float64) - mean
            ss += float(np.sum(centered * centered, dtype=np.float64))
        out["std"] = float(np.sqrt(max(ss / n, 0.0)))
    else:
        if quantile_temp_path is None:
            q = np.quantile(np.asarray(arr, dtype=np.float64), [0.01, 0.50, 0.99])
            values = [float(q[0]), float(q[1]), float(q[2])]
        else:
            values = _quantiles_disk_backed(arr, chunk_rows=chunk_rows, temp_path=quantile_temp_path)
        out.update({"p01": values[0], "p50": values[1], "p99": values[2]})
    return out


def _prediction_summary(arrays: dict[str, np.ndarray], *, n_rows: int, chunk_rows: int, temp_prefix: Path) -> dict[str, object]:
    def qpath(name: str) -> Path:
        return Path(str(temp_prefix) + f".summary.{name}.npy")

    return {
        "n_rows": int(n_rows),
        "p_no_move": _stats_chunked(arrays["p_no_move"], include_std=False, chunk_rows=chunk_rows, quantile_temp_path=qpath("p_no_move")),
        "expected_return_bps": _stats_chunked(arrays["expected_return_bps"], include_std=True, chunk_rows=chunk_rows)
        | {k: v for k, v in _stats_chunked(arrays["expected_return_bps"], include_std=False, chunk_rows=chunk_rows, quantile_temp_path=qpath("expected_return_bps")).items() if k.startswith("p")},
        "expected_abs_move_bps": _stats_chunked(arrays["expected_abs_move_bps"], include_std=True, chunk_rows=chunk_rows)
        | {k: v for k, v in _stats_chunked(arrays["expected_abs_move_bps"], include_std=False, chunk_rows=chunk_rows, quantile_temp_path=qpath("expected_abs_move_bps")).items() if k.startswith("p")},
        "predicted_vol_bps": _stats_chunked(arrays["predicted_vol_bps"], include_std=True, chunk_rows=chunk_rows),
        "confidence": _stats_chunked(arrays["confidence"], include_std=True, chunk_rows=chunk_rows),
    }


def _linear_signal_summary(
    *,
    path: Path,
    metadata: LinearSignalArtifactMetadata,
    dtype: str,
    decision_event_index: np.ndarray,
    decision_local_ts_us: np.ndarray,
) -> dict[str, object]:
    return {
        "schema": LINEAR_SIGNAL_ARTIFACT_SCHEMA,
        "path": str(path),
        "n_rows": metadata.n_rows,
        "dtype": str(np.dtype(dtype)),
        "fields": list(linear_signal_array_fields()),
        "metadata": metadata.as_dict(),
        "first_decision_event_index": int(decision_event_index[0]),
        "last_decision_event_index": int(decision_event_index[-1]),
        "first_decision_local_ts_us": int(decision_local_ts_us[0]),
        "last_decision_local_ts_us": int(decision_local_ts_us[-1]),
    }


def build_linear_signals_from_config(config: BuildLinearSignalsConfig) -> dict[str, object]:
    if not isinstance(config, BuildLinearSignalsConfig):
        raise ValueError("config must be BuildLinearSignalsConfig")
    output_npz = Path(config.output_npz) if config.output_npz is not None else _default_output_npz(config.tape_root)
    output_json = Path(config.output_json) if config.output_json is not None else _default_output_json(config.tape_root)
    if output_npz.exists() and not config.overwrite:
        raise FileExistsError(f"output_npz already exists: {output_npz}")
    if output_json.exists() and not config.overwrite:
        raise FileExistsError(f"output_json already exists: {output_json}")

    tape = load_execution_tape(
        config.tape_root,
        mmap_mode=config.mmap_mode,
        validation_mode=ExecutionTapeValidationMode.SHAPE_ONLY,
    )
    result = load_linear_train_result(config.linear_train_result_json)
    if result.schema != LINEAR_TRAINING_RESULT_SCHEMA:
        raise ValueError("linear train result schema mismatch")

    schedule = schedule_config_from_train_result(result)
    transform = transform_config_from_train_result(result)
    scan = _scan_feature_chunks(
        tape,
        schedule=schedule,
        transform=transform,
        start_event_index=config.start_event_index,
        max_decisions=config.max_decisions,
        chunk_rows=config.chunk_rows,
        output_dtype=config.output_dtype,
    )
    metadata = LinearSignalArtifactMetadata(
        tape_schema=tape.manifest.schema,
        exchange=tape.manifest.exchange,
        symbol=tape.manifest.symbol,
        num_events=tape.manifest.num_events,
        num_l2_batches=tape.manifest.num_l2_batches,
        num_trades=tape.manifest.num_trades,
        start_local_ts_us=tape.manifest.start_local_ts_us,
        end_local_ts_us=tape.manifest.end_local_ts_us,
        decision_schedule=dict(scan.decision_schedule),
        start_event_index=scan.start_event_index,
        n_rows=scan.num_decisions,
    )
    writers = _SignalWriters.create(output_npz, n_rows=scan.num_decisions, output_dtype=config.output_dtype)
    try:
        for chunk in iter_execution_linear_feature_chunks(
            tape,
            schedule_config=schedule,
            start_event_index=config.start_event_index,
            max_decisions=config.max_decisions,
            chunk_rows=config.chunk_rows,
            output_dtype=config.output_dtype,
            transform_config=transform,
        ):
            dataset = _feature_dataset_from_chunk(
                chunk,
                replay_start_event_index=scan.replay_start_event_index,
                decision_schedule=scan.decision_schedule,
                transform_config=scan.transform_config,
            )
            chunk_result = build_linear_signal_build_result(
                tape=tape,
                feature_dataset=dataset,
                linear_train_result=result,
                signal_config=LinearSignalConfig(magnitude_input=config.magnitude_input),
                output_dtype=config.output_dtype,
            )
            writers.append(dataset, chunk_result.artifact.arrays)
        if writers.row != scan.num_decisions:
            raise ValueError("signal row count changed during chunked replay")
        writers.flush()
        save_linear_signal_artifact_arrays(
            output_npz,
            metadata=metadata,
            decision_event_index=writers.decision_event_index,
            decision_local_ts_us=writers.decision_local_ts_us,
            arrays=writers.arrays,
            overwrite=config.overwrite,
            validate_chunk_rows=config.chunk_rows,
        )
        prediction_summary = _prediction_summary(
            writers.arrays,
            n_rows=scan.num_decisions,
            chunk_rows=config.chunk_rows,
            temp_prefix=output_npz.parent / f".{output_npz.name}",
        )
        linear_signals_summary = _linear_signal_summary(
            path=output_npz,
            metadata=metadata,
            dtype=config.output_dtype,
            decision_event_index=writers.decision_event_index,
            decision_local_ts_us=writers.decision_local_ts_us,
        )
    finally:
        writers.flush()
        writers.close()
        writers.cleanup()

    summary: dict[str, object] = {
        "status": "ok",
        "run_type": "build_linear_signals",
        "tape_root": str(Path(config.tape_root)),
        "linear_train_result_json": str(Path(config.linear_train_result_json)),
        "output_npz": str(output_npz),
        "output_json": str(output_json),
        "linear_train_result": {
            "schema": result.schema,
            "dataset_id": result.dataset_id,
            "manifest_hash": result.manifest_hash,
            "selection_summary": result.selection_summary,
        },
        "feature_dataset": scan.feature_dataset_summary(),
        "linear_signals": linear_signals_summary,
        "alignment": {
            "replay_start_event_index": scan.replay_start_event_index,
            "first_signal_event_index": scan.first_decision_event_index,
            "first_signal_local_ts_us": scan.first_decision_local_ts_us,
            "n_signal_rows": scan.num_decisions,
        },
        "prediction_summary": prediction_summary,
        "resource_mode": {
            "chunked_features": True,
            "disk_backed_signal_writers": True,
            "chunk_rows": config.chunk_rows,
        },
        "config": config.as_dict(),
    }
    _write_json_atomic(output_json, summary, overwrite=config.overwrite)
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tape-root", required=True)
    parser.add_argument("--linear-train-result-json", required=True)
    parser.add_argument("--output-npz")
    parser.add_argument("--output-json")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-mmap", action="store_true")
    parser.add_argument("--start-event-index", type=int)
    parser.add_argument("--max-decisions", type=int)
    parser.add_argument("--output-dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--magnitude-input", choices=MAGNITUDE_INPUT_MODES, default=MAGNITUDE_INPUT_LOG1P_BPS)
    parser.add_argument("--chunk-rows", type=int, default=100_000)
    return parser


def _config_from_args(args: argparse.Namespace) -> BuildLinearSignalsConfig:
    return BuildLinearSignalsConfig(
        tape_root=args.tape_root,
        linear_train_result_json=args.linear_train_result_json,
        output_npz=args.output_npz,
        output_json=args.output_json,
        overwrite=args.overwrite,
        mmap_mode=None if args.no_mmap else "r",
        start_event_index=args.start_event_index,
        max_decisions=args.max_decisions,
        output_dtype=args.output_dtype,
        magnitude_input=args.magnitude_input,
        chunk_rows=args.chunk_rows,
    )


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    summary = build_linear_signals_from_config(_config_from_args(args))
    print(json.dumps(summary, sort_keys=True, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
