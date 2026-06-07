"""Build adverse-selection signal artifacts from an execution tape and trained model."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np

from mmrt.execution.adverse_selection import (
    adverse_selection_config_from_training_summary,
    build_adverse_selection_dataset,
    summarize_adverse_selection_dataset,
)
from mmrt.execution.adverse_signal import (
    ADVERSE_SELECTION_SIGNALS_FILENAME,
    build_adverse_selection_signal_artifact,
    load_adverse_selection_model,
    save_adverse_selection_signals,
)
from mmrt.execution.execution_tape import load_execution_tape

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
    model_npz: str
    output_npz: str | None = None
    output_json: str | None = None
    overwrite: bool = False
    mmap_mode: str | None = "r"

    def __post_init__(self) -> None:
        object.__setattr__(self, "tape_root", _require_nonempty_str(self.tape_root, "tape_root"))
        object.__setattr__(self, "model_npz", _require_nonempty_str(self.model_npz, "model_npz"))
        if self.output_npz is not None:
            object.__setattr__(self, "output_npz", _require_nonempty_str(self.output_npz, "output_npz"))
        if self.output_json is not None:
            object.__setattr__(self, "output_json", _require_nonempty_str(self.output_json, "output_json"))
        object.__setattr__(self, "overwrite", _require_bool(self.overwrite, "overwrite"))
        if self.mmap_mode not in (None, "r"):
            raise ValueError("mmap_mode must be None or 'r'")


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

    tape = load_execution_tape(config.tape_root, mmap_mode=config.mmap_mode)
    model = load_adverse_selection_model(config.model_npz)
    if model.exchange != tape.manifest.exchange or model.symbol != tape.manifest.symbol:
        raise ValueError("adverse-selection model exchange/symbol must match execution tape")

    payload = json.loads(model.config_json)
    adverse_config = adverse_selection_config_from_training_summary(payload)
    dataset = build_adverse_selection_dataset(tape, config=adverse_config)
    if tuple(dataset.feature_names) != tuple(model.feature_names):
        raise ValueError("dataset feature_names must match model feature_names exactly")
    signals = build_adverse_selection_signal_artifact(dataset, model)
    save_adverse_selection_signals(output_npz, signals, overwrite=config.overwrite)

    summary: dict[str, object] = {
        "status": "ok" if signals.decision_local_ts_us.size > 0 else "warning",
        "run_type": "build_adverse_selection_signals",
        "tape_root": str(Path(config.tape_root)),
        "model_npz": str(Path(config.model_npz)),
        "output_npz": str(output_npz),
        "output_json": str(output_json),
        "model": {
            "schema": model.schema,
            "exchange": model.exchange,
            "symbol": model.symbol,
            "num_features": len(model.feature_names),
            "num_targets": len(model.target_names),
            "target_names": list(model.target_names),
        },
        "signals": {
            "schema": signals.schema,
            "num_decisions": int(signals.decision_local_ts_us.shape[0]),
            "target_names": list(signals.target_names),
            "prediction_summary": {
                target: {
                    "mean": float(np.mean(arr)) if arr.size else 0.0,
                    "min": float(np.min(arr)) if arr.size else 0.0,
                    "max": float(np.max(arr)) if arr.size else 0.0,
                }
                for target, arr in signals.predictions.items()
            },
        },
        "dataset": summarize_adverse_selection_dataset(dataset),
    }
    _write_json_atomic(output_json, summary, overwrite=config.overwrite)
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tape-root", required=True)
    parser.add_argument("--model-npz", required=True)
    parser.add_argument("--output-npz")
    parser.add_argument("--output-json")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-mmap", action="store_true")
    return parser


def _config_from_args(args: argparse.Namespace) -> BuildAdverseSelectionSignalsConfig:
    return BuildAdverseSelectionSignalsConfig(
        tape_root=args.tape_root,
        model_npz=args.model_npz,
        output_npz=args.output_npz,
        output_json=args.output_json,
        overwrite=args.overwrite,
        mmap_mode=None if args.no_mmap else "r",
    )


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    summary = build_adverse_selection_signals_from_config(_config_from_args(args))
    print(json.dumps(summary, sort_keys=True, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
