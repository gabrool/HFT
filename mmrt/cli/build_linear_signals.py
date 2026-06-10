"""Build aligned no-move-gated linear signal artifacts from an execution tape."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path

from mmrt.execution.execution_tape import load_execution_tape
from mmrt.execution.linear_signal import (
    LINEAR_SIGNALS_FILENAME,
    MAGNITUDE_INPUT_LOG1P_BPS,
    MAGNITUDE_INPUT_MODES,
    LinearSignalConfig,
    linear_signal_artifact_summary,
    save_linear_signal_artifact_npz,
)
from mmrt.execution.linear_signal_builder import (
    build_execution_linear_feature_dataset,
    build_linear_signal_build_result,
    execution_linear_feature_dataset_summary,
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
    decision_interval_us: int = 500_000
    start_event_index: int | None = None
    max_decisions: int | None = None
    output_dtype: str = "float32"
    magnitude_input: str = MAGNITUDE_INPUT_LOG1P_BPS

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
        object.__setattr__(self, "decision_interval_us", _require_positive_int(self.decision_interval_us, "decision_interval_us"))
        object.__setattr__(self, "start_event_index", _require_optional_nonnegative_int(self.start_event_index, "start_event_index"))
        object.__setattr__(self, "max_decisions", _require_optional_positive_int(self.max_decisions, "max_decisions"))
        if self.output_dtype not in ("float32", "float64"):
            raise ValueError("output_dtype must be 'float32' or 'float64'")
        if self.magnitude_input not in MAGNITUDE_INPUT_MODES:
            raise ValueError("magnitude_input must be a supported magnitude input mode")

    def as_dict(self) -> dict[str, object]:
        return {
            "tape_root": self.tape_root,
            "linear_train_result_json": self.linear_train_result_json,
            "output_npz": self.output_npz,
            "output_json": self.output_json,
            "overwrite": self.overwrite,
            "mmap_mode": self.mmap_mode,
            "decision_interval_us": self.decision_interval_us,
            "start_event_index": self.start_event_index,
            "max_decisions": self.max_decisions,
            "output_dtype": self.output_dtype,
            "magnitude_input": self.magnitude_input,
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


def build_linear_signals_from_config(config: BuildLinearSignalsConfig) -> dict[str, object]:
    if not isinstance(config, BuildLinearSignalsConfig):
        raise ValueError("config must be BuildLinearSignalsConfig")
    output_npz = Path(config.output_npz) if config.output_npz is not None else _default_output_npz(config.tape_root)
    output_json = Path(config.output_json) if config.output_json is not None else _default_output_json(config.tape_root)
    if output_npz.exists() and not config.overwrite:
        raise FileExistsError(f"output_npz already exists: {output_npz}")
    if output_json.exists() and not config.overwrite:
        raise FileExistsError(f"output_json already exists: {output_json}")

    tape = load_execution_tape(config.tape_root, mmap_mode=config.mmap_mode)
    result = load_linear_train_result(config.linear_train_result_json)
    if result.schema != LINEAR_TRAINING_RESULT_SCHEMA:
        raise ValueError("linear train result schema mismatch")

    features = build_execution_linear_feature_dataset(
        tape,
        decision_interval_us=config.decision_interval_us,
        start_event_index=config.start_event_index,
        max_decisions=config.max_decisions,
        output_dtype=config.output_dtype,
    )
    build_result = build_linear_signal_build_result(
        tape=tape,
        feature_dataset=features,
        linear_train_result=result,
        signal_config=LinearSignalConfig(magnitude_input=config.magnitude_input),
        output_dtype=config.output_dtype,
    )
    artifact = build_result.artifact
    save_linear_signal_artifact_npz(output_npz, artifact, overwrite=config.overwrite)

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
        "feature_dataset": execution_linear_feature_dataset_summary(features),
        "linear_signals": linear_signal_artifact_summary(artifact, path=str(output_npz)),
        "alignment": {
            "replay_start_event_index": features.replay_start_event_index,
            "first_signal_event_index": int(artifact.decision_event_index[0]),
            "first_signal_local_ts_us": int(artifact.decision_local_ts_us[0]),
            "n_signal_rows": artifact.n_rows,
        },
        "prediction_summary": build_result.prediction_summary,
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
    parser.add_argument("--decision-interval-us", type=int, default=500_000)
    parser.add_argument("--start-event-index", type=int)
    parser.add_argument("--max-decisions", type=int)
    parser.add_argument("--output-dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--magnitude-input", choices=MAGNITUDE_INPUT_MODES, default=MAGNITUDE_INPUT_LOG1P_BPS)
    return parser


def _config_from_args(args: argparse.Namespace) -> BuildLinearSignalsConfig:
    return BuildLinearSignalsConfig(
        tape_root=args.tape_root,
        linear_train_result_json=args.linear_train_result_json,
        output_npz=args.output_npz,
        output_json=args.output_json,
        overwrite=args.overwrite,
        mmap_mode=None if args.no_mmap else "r",
        decision_interval_us=args.decision_interval_us,
        start_event_index=args.start_event_index,
        max_decisions=args.max_decisions,
        output_dtype=args.output_dtype,
        magnitude_input=args.magnitude_input,
    )


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    summary = build_linear_signals_from_config(_config_from_args(args))
    print(json.dumps(summary, sort_keys=True, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
