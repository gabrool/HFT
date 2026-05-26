"""CLI audit for existing MMRT storage datasets.

This command validates and summarizes an already-written storage dataset.
It does not ingest raw market data, create splits, recompute labels, train
models, inspect raw events, repair ordering, or mutate the dataset manifest.
"""

import argparse
import json
import math
from pathlib import Path
from typing import Sequence

from mmrt.contracts import SplitRole
from mmrt.storage import manifest as mf
from mmrt.storage import reader as rd


def _positive_int(text: str) -> int:
    value = int(text)
    if value <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return value


def _nonnegative_int(text: str) -> int:
    value = int(text)
    if value < 0:
        raise argparse.ArgumentTypeError("value must be a non-negative integer")
    return value


def _require_positive_int(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive int")
    return value


def _role_order() -> tuple[SplitRole, ...]:
    return (SplitRole.TRAIN, SplitRole.VAL, SplitRole.TEST)


def _role_to_str(role: SplitRole | str) -> str:
    return SplitRole(role).value


def _segment_summary(manifest: mf.StorageManifest) -> dict[str, object]:
    first = manifest.segments[0] if manifest.segments else None
    last = manifest.segments[-1] if manifest.segments else None
    return {
        "count": len(manifest.segments),
        "total_rows": manifest.total_rows,
        "total_labels": manifest.total_labels,
        "first_segment_key": first.segment_key if first is not None else None,
        "last_segment_key": last.segment_key if last is not None else None,
        "first_row_idx": first.first_row_idx if first is not None else None,
        "last_row_idx": last.last_row_idx if last is not None else None,
    }


def _split_summary_from_manifest(manifest: mf.StorageManifest) -> dict[str, dict[str, object]]:
    out: dict[str, dict[str, object]] = {}
    for role in _role_order():
        entries = tuple(sp for sp in manifest.splits if sp.role == role)
        out[role.value] = {
            "entry_count": len(entries),
            "row_count": sum(sp.end_row - sp.start_row for sp in entries),
            "segment_count": len({sp.segment_key for sp in entries}),
            "first_local_start_us": min((sp.local_time_range.start_us for sp in entries), default=None),
            "last_local_end_us": max((sp.local_time_range.end_us for sp in entries), default=None),
            "scan": None,
        }
    return out


def _scan_split(
    reader: rd.StorageDatasetReader,
    role: SplitRole,
    *,
    batch_size: int,
    max_scan_rows: int,
) -> dict[str, object]:
    entries = reader.split_entries(role)
    manifest_row_count = sum(sp.end_row - sp.start_row for sp in entries)
    if not entries:
        return {
            "scanned_rows": 0,
            "manifest_row_count": 0,
            "scan_limit_hit": False,
            "batch_count": 0,
            "max_batch_rows": 0,
            "first_row_idx": None,
            "last_row_idx": None,
            "strictly_increasing_row_idx": True,
        }

    scanned_rows = 0
    batch_count = 0
    max_batch_rows = 0
    first_row_idx: int | None = None
    last_row_idx: int | None = None
    strictly_increasing = True
    prev: int | None = None

    for batch in reader.iter_split_batches(role, columns=(mf.ROW_IDX_COLUMN,), batch_size=batch_size):
        remaining = max_scan_rows - scanned_rows
        if remaining <= 0:
            break
        take = min(batch.num_rows, remaining)
        values = batch.column(0).slice(0, take).to_pylist()
        scanned_rows += take
        batch_count += 1
        max_batch_rows = max(max_batch_rows, take)
        if values:
            if first_row_idx is None:
                first_row_idx = values[0]
            last_row_idx = values[-1]
            for value in values:
                if prev is not None and value <= prev:
                    strictly_increasing = False
                prev = value
        if scanned_rows >= max_scan_rows:
            break

    return {
        "scanned_rows": scanned_rows,
        "manifest_row_count": manifest_row_count,
        "scan_limit_hit": scanned_rows < manifest_row_count,
        "batch_count": batch_count,
        "max_batch_rows": max_batch_rows,
        "first_row_idx": first_row_idx,
        "last_row_idx": last_row_idx,
        "strictly_increasing_row_idx": strictly_increasing,
    }


def _manifest_summary(manifest: mf.StorageManifest) -> dict[str, object]:
    return {
        "dataset_id": manifest.dataset_id,
        "manifest_hash": manifest.content_hash(),
        "created_at_utc": manifest.created_at_utc,
        "exchange": manifest.exchange,
        "symbol": manifest.symbol,
        "storage_format": manifest.storage_format.value,
        "time_unit": manifest.time_unit.value,
        "decision_stride_us": manifest.decision_stride_us,
        "feature_count": len(manifest.feature_columns),
        "label_count": len(manifest.label_columns),
        "required_column_count": len(manifest.required_columns),
        "segment_count": len(manifest.segments),
        "split_entry_count": len(manifest.splits),
        "total_rows": manifest.total_rows,
        "total_labels": manifest.total_labels,
    }


def _warnings_from_report(report: dict[str, object]) -> list[str]:
    warnings: list[str] = []
    manifest = report["manifest"]
    splits = report["splits"]
    if manifest["split_entry_count"] == 0:
        warnings.append("no_split_entries")
    for role_name in (_role_to_str(SplitRole.TRAIN), _role_to_str(SplitRole.VAL)):
        if splits[role_name]["row_count"] == 0:
            warnings.append(f"missing_{role_name}_split")
    for role in _role_order():
        role_name = _role_to_str(role)
        scan = splits[role_name]["scan"]
        if scan is None:
            continue
        if scan["scan_limit_hit"]:
            warnings.append(f"split_scan_limit_hit:{role_name}")
        if not scan["strictly_increasing_row_idx"]:
            warnings.append(f"split_row_idx_not_increasing:{role_name}")
    return warnings


def audit_dataset(
    dataset_root: str,
    *,
    validate_on_open: bool = True,
    batch_size: int = rd.DEFAULT_BATCH_SIZE,
    max_scan_rows: int = 200_000,
    scan_splits: bool = True,
) -> dict[str, object]:
    if not isinstance(dataset_root, str) or not dataset_root.strip():
        raise ValueError("dataset_root must be a non-empty string")
    if not isinstance(validate_on_open, bool):
        raise ValueError("validate_on_open must be bool")
    _require_positive_int(batch_size, "batch_size")
    _require_positive_int(max_scan_rows, "max_scan_rows")
    if not isinstance(scan_splits, bool):
        raise ValueError("scan_splits must be bool")

    reader = rd.open_dataset(dataset_root, validate_on_open=validate_on_open, batch_size=batch_size)
    manifest = reader.manifest
    manifest.validate_against_current_code()
    split_summary = _split_summary_from_manifest(manifest)
    if scan_splits:
        for role in _role_order():
            split_summary[role.value]["scan"] = _scan_split(
                reader,
                role,
                batch_size=batch_size,
                max_scan_rows=max_scan_rows,
            )

    has_train = split_summary["train"]["row_count"] > 0
    has_val = split_summary["val"]["row_count"] > 0
    has_test = split_summary["test"]["row_count"] > 0

    report = {
        "status": "ok",
        "dataset_root": dataset_root,
        "validation": {
            "validate_on_open": validate_on_open,
            "manifest_code_compatible": True,
            "split_scan_enabled": scan_splits,
            "batch_size": batch_size,
            "max_scan_rows": max_scan_rows,
        },
        "manifest": _manifest_summary(manifest),
        "segments": _segment_summary(manifest),
        "splits": split_summary,
        "readiness": {
            "has_train_split": has_train,
            "has_val_split": has_val,
            "has_test_split": has_test,
            "train_ready": has_train and has_val,
        },
        "warnings": [],
    }
    report["warnings"] = _warnings_from_report(report)
    return report


def _write_json_atomic(report: dict[str, object], path: str) -> str:
    if not isinstance(path, str) or not path.strip():
        raise ValueError("path must be a non-empty string")
    target = Path(path)
    if target.suffix != ".json":
        raise ValueError("output path must end with .json")
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(report, sort_keys=True, indent=2, allow_nan=True) + "\n"
    tmp = target.with_suffix(target.suffix + ".tmp")
    tmp.write_text(payload, encoding="utf-8")
    tmp.replace(target)
    return str(target)


def _print_json(payload: dict[str, object]) -> None:
    print(json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=True))


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mmrt-audit-dataset",
        description="Validate and summarize an existing MMRT storage dataset.",
    )
    parser.add_argument("--dataset-root", required=True, help="Path to an existing MMRT storage dataset root.")
    parser.add_argument("--output-json", default=None, help="Optional path where the full audit JSON report will be written.")
    parser.add_argument("--batch-size", type=_positive_int, default=rd.DEFAULT_BATCH_SIZE)
    parser.add_argument("--max-scan-rows", type=_positive_int, default=200_000)
    parser.add_argument("--no-validate-on-open", action="store_true", help="Skip full reader validation on open.")
    parser.add_argument("--no-scan-splits", action="store_true", help="Skip bounded split row scans and use manifest metadata only.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    report = audit_dataset(
        args.dataset_root,
        validate_on_open=not args.no_validate_on_open,
        batch_size=args.batch_size,
        max_scan_rows=args.max_scan_rows,
        scan_splits=not args.no_scan_splits,
    )
    if args.output_json is not None:
        report = dict(report)
        report["output_json"] = _write_json_atomic(report, args.output_json)
    _print_json(report)
    return 0


__all__ = [
    "build_arg_parser",
    "audit_dataset",
    "main",
]


if __name__ == "__main__":
    raise SystemExit(main())
