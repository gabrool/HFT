"""Build adverse-selection signal artifacts from an execution tape and trained model."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import gc
import json
from pathlib import Path

import numpy as np

from mmrt.execution.adverse_selection import (
    adverse_selection_config_from_training_summary,
    adverse_label_config_from_config,
)
from mmrt.execution.adverse_selection_feature_store import (
    build_adverse_selection_features_to_disk,
    summarize_adverse_selection_feature_store,
)
from mmrt.execution.adverse_signal import (
    ADVERSE_SELECTION_SIGNALS_FILENAME,
    ADVERSE_SELECTION_SIGNALS_SCHEMA,
    load_adverse_selection_model,
    save_adverse_selection_signals_arrays,
)
from mmrt.execution.decision_grid import load_decision_grid, validate_decision_grid_for_execution_tape
from mmrt.execution.execution_tape import ExecutionTapeValidationMode, load_execution_tape

__all__ = [
    "BuildAdverseSelectionSignalsConfig",
    "build_adverse_selection_signals_from_config",
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


@dataclass(frozen=True, slots=True)
class BuildAdverseSelectionSignalsConfig:
    tape_root: str
    decision_grid_path: str
    model_npz: str
    output_npz: str | None = None
    output_json: str | None = None
    overwrite: bool = False
    mmap_mode: str | None = "r"
    feature_dataset_root: str | None = None
    work_dir: str | None = None
    chunk_rows: int = 100_000
    keep_feature_dataset: bool = True
    cleanup_work_dir: bool = False
    progress_interval: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "tape_root", _require_nonempty_str(self.tape_root, "tape_root"))
        object.__setattr__(self, "decision_grid_path", _require_nonempty_str(self.decision_grid_path, "decision_grid_path"))
        object.__setattr__(self, "model_npz", _require_nonempty_str(self.model_npz, "model_npz"))
        if self.output_npz is not None:
            object.__setattr__(self, "output_npz", _require_nonempty_str(self.output_npz, "output_npz"))
        if self.output_json is not None:
            object.__setattr__(self, "output_json", _require_nonempty_str(self.output_json, "output_json"))
        object.__setattr__(self, "overwrite", _require_bool(self.overwrite, "overwrite"))
        if self.mmap_mode not in (None, "r"):
            raise ValueError("mmap_mode must be None or 'r'")
        if self.feature_dataset_root is not None:
            object.__setattr__(self, "feature_dataset_root", _require_nonempty_str(self.feature_dataset_root, "feature_dataset_root"))
        if self.work_dir is not None:
            object.__setattr__(self, "work_dir", _require_nonempty_str(self.work_dir, "work_dir"))
        if isinstance(self.chunk_rows, bool) or self.chunk_rows <= 0:
            raise ValueError("chunk_rows must be a positive int")
        if self.progress_interval is not None and (isinstance(self.progress_interval, bool) or not isinstance(self.progress_interval, int) or self.progress_interval <= 0):
            raise ValueError("progress_interval must be None or a positive int")
        object.__setattr__(self, "keep_feature_dataset", _require_bool(self.keep_feature_dataset, "keep_feature_dataset"))
        object.__setattr__(self, "cleanup_work_dir", _require_bool(self.cleanup_work_dir, "cleanup_work_dir"))


def _default_output_npz(tape_root: str) -> Path:
    return Path(tape_root) / ADVERSE_SELECTION_SIGNALS_FILENAME


def _default_output_json(tape_root: str) -> Path:
    return Path(tape_root) / "adverse_selection_signals_summary.json"


def _write_json_atomic(path: Path, payload: dict[str, object], *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"output_json already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)



def _prediction_summary(arrays: dict[str, np.ndarray], *, chunk_rows: int) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for target, arr in arrays.items():
        n = int(arr.shape[0])
        if n == 0:
            out[target] = {"mean": 0.0, "min": 0.0, "max": 0.0}
            continue
        total = 0.0
        min_value = float("inf")
        max_value = float("-inf")
        for start in range(0, n, chunk_rows):
            chunk = np.asarray(arr[start:min(start + chunk_rows, n)], dtype=np.float32)
            total += float(np.sum(chunk, dtype=np.float64))
            min_value = min(min_value, float(np.min(chunk)))
            max_value = max(max_value, float(np.max(chunk)))
        out[target] = {"mean": float(total / n), "min": min_value, "max": max_value}
    return out

def build_adverse_selection_signals_from_config(
    config: BuildAdverseSelectionSignalsConfig,
) -> dict[str, object]:
    if not isinstance(config, BuildAdverseSelectionSignalsConfig):
        raise ValueError("config must be BuildAdverseSelectionSignalsConfig")
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
    decision_grid = load_decision_grid(config.decision_grid_path)
    validate_decision_grid_for_execution_tape(decision_grid, tape)
    model = load_adverse_selection_model(config.model_npz)
    if model.exchange != tape.manifest.exchange or model.symbol != tape.manifest.symbol:
        raise ValueError("adverse-selection model exchange/symbol must match execution tape")
    if model.decision_grid_hash != decision_grid.decision_grid_hash:
        raise ValueError("adverse-selection model decision_grid_hash must match decision grid")
    if model.decision_grid_schema != decision_grid.metadata.schema or model.decision_grid_n_rows != decision_grid.n_rows:
        raise ValueError("adverse-selection model decision grid metadata must match decision grid")

    payload = json.loads(model.config_json)
    adverse_config = adverse_selection_config_from_training_summary(payload)
    adverse_label_config = adverse_label_config_from_config(adverse_config)
    feature_root = Path(config.feature_dataset_root) if config.feature_dataset_root is not None else Path(config.tape_root) / "adverse_selection_feature_dataset"
    dataset = build_adverse_selection_features_to_disk(
        tape,
        config=adverse_config,
        decision_grid=decision_grid,
        output_root=feature_root,
        work_dir=config.work_dir,
        chunk_rows=config.chunk_rows,
        overwrite=config.overwrite,
        cleanup_chunks=not config.keep_feature_dataset,
        cleanup_work_dir=config.cleanup_work_dir,
        progress_interval=config.progress_interval,
    )
    if tuple(dataset.feature_names) != tuple(model.feature_names):
        raise ValueError("dataset feature_names must match model feature_names exactly")
    temp_pred_paths = {name: output_npz.parent / f".{output_npz.name}.{name}.npy" for name in model.target_names}
    preds = {name: np.lib.format.open_memmap(path, mode="w+", dtype=np.float32, shape=(dataset.num_rows,)) for name, path in temp_pred_paths.items()}
    for start_row in range(0, dataset.num_rows, config.chunk_rows):
        end_row = min(start_row + config.chunk_rows, dataset.num_rows)
        X = np.asarray(dataset.features[start_row:end_row], dtype=np.float64)
        raw = (X - model.feature_mean) / model.feature_scale @ model.coefficients.T + model.intercepts
        for i, target in enumerate(model.target_names):
            pred = raw[:, i]
            if target.endswith("_filled") or target.endswith("_toxic_fill"):
                pred = np.clip(pred, 0.0, 1.0)
            elif target.endswith("_toxic_cost_bps") or target.endswith("_adverse_bps") or target.endswith("_fill_latency_us"):
                pred = np.maximum(pred, 0.0)
            preds[target][start_row:end_row] = pred.astype(np.float32, copy=False)
        if config.progress_interval is not None and (end_row % config.progress_interval == 0 or end_row == dataset.num_rows):
            print(f"adverse_signals progress rows_predicted={end_row}/{dataset.num_rows}")
    for arr in preds.values():
        arr.flush()
    del arr
    prediction_summary = _prediction_summary(preds, chunk_rows=config.chunk_rows)
    save_adverse_selection_signals_arrays(
        output_npz,
        decision_local_ts_us=dataset.decision_local_ts_us,
        decision_event_index=dataset.decision_event_index,
        decision_event_seq=dataset.decision_event_seq,
        target_names=model.target_names,
        predictions=preds,
        adverse_label_config=adverse_label_config,
        decision_grid_schema=decision_grid.metadata.schema,
        decision_grid_hash=decision_grid.decision_grid_hash,
        decision_grid_n_rows=decision_grid.n_rows,
        decision_schedule=decision_grid.decision_schedule,
        overwrite=config.overwrite,
        validate_chunk_rows=config.chunk_rows,
    )
    del preds
    gc.collect()
    for path in temp_pred_paths.values():
        path.unlink(missing_ok=True)

    summary: dict[str, object] = {
        "status": "ok" if dataset.num_rows > 0 else "warning",
        "run_type": "build_adverse_selection_signals",
        "tape_root": str(Path(config.tape_root)),
        "decision_grid_path": str(Path(config.decision_grid_path)),
        "model_npz": str(Path(config.model_npz)),
        "output_npz": str(output_npz),
        "output_json": str(output_json),
        "feature_dataset_root": str(feature_root),
        "work_dir": str((Path(config.work_dir) if config.work_dir is not None else Path(config.tape_root)) / "adverse_selection_work"),
        "model": {
            "schema": model.schema,
            "exchange": model.exchange,
            "symbol": model.symbol,
            "num_features": len(model.feature_names),
            "num_targets": len(model.target_names),
            "target_names": list(model.target_names),
            "decision_grid_hash": model.decision_grid_hash,
            "adverse_label_config": adverse_label_config,
        },
        "signals": {
            "schema": ADVERSE_SELECTION_SIGNALS_SCHEMA,
            "num_decisions": int(dataset.num_rows),
            "target_names": list(model.target_names),
            "prediction_summary": prediction_summary,
            "adverse_label_config": adverse_label_config,
        },
        "fill_simulator": adverse_label_config,
        "adverse_label_config": adverse_label_config,
        "decision_grid": {
            "schema": decision_grid.metadata.schema,
            "hash": decision_grid.decision_grid_hash,
            "n_rows": decision_grid.n_rows,
            "schedule": decision_grid.decision_schedule,
        },
        "feature_dataset": summarize_adverse_selection_feature_store(dataset),
        "resource_mode": {"disk_backed_features": True, "disk_backed_index": True, "chunk_rows": config.chunk_rows, "keep_feature_dataset": config.keep_feature_dataset, "keep_work_dir": not config.cleanup_work_dir},
    }
    _write_json_atomic(output_json, summary, overwrite=config.overwrite)
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tape-root", required=True)
    parser.add_argument("--decision-grid", dest="decision_grid_path", required=True)
    parser.add_argument("--model-npz", required=True)
    parser.add_argument("--output-npz")
    parser.add_argument("--output-json")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-mmap", action="store_true")
    parser.add_argument("--feature-dataset-root")
    parser.add_argument("--work-dir")
    parser.add_argument("--chunk-rows", type=int, default=100_000)
    parser.add_argument("--keep-feature-dataset", dest="keep_feature_dataset", action="store_true", default=True)
    parser.add_argument("--delete-feature-dataset-after-success", dest="keep_feature_dataset", action="store_false")
    parser.add_argument("--cleanup-work-dir", dest="cleanup_work_dir", action="store_true", default=False)
    parser.add_argument("--keep-work-dir", dest="cleanup_work_dir", action="store_false")
    parser.add_argument("--progress-interval", type=int)
    return parser


def _config_from_args(args: argparse.Namespace) -> BuildAdverseSelectionSignalsConfig:
    return BuildAdverseSelectionSignalsConfig(
        tape_root=args.tape_root,
        decision_grid_path=args.decision_grid_path,
        model_npz=args.model_npz,
        output_npz=args.output_npz,
        output_json=args.output_json,
        overwrite=args.overwrite,
        mmap_mode=None if args.no_mmap else "r",
        feature_dataset_root=getattr(args, "feature_dataset_root", None),
        work_dir=getattr(args, "work_dir", None),
        chunk_rows=getattr(args, "chunk_rows", 100_000),
        keep_feature_dataset=getattr(args, "keep_feature_dataset", True),
        cleanup_work_dir=getattr(args, "cleanup_work_dir", False),
        progress_interval=getattr(args, "progress_interval", None),
    )


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    summary = build_adverse_selection_signals_from_config(_config_from_args(args))
    print(json.dumps(summary, sort_keys=True, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
