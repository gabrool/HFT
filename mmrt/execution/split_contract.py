"""Shared train/val/test split contract for decision-grid execution artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

from mmrt.contracts import SplitRole
from mmrt.execution.decision_grid import DecisionGrid
from mmrt.storage import manifest as storage_manifest

EXECUTION_SPLIT_CONTRACT_SCHEMA = "mmrt_execution_split_contract_v1"
EXECUTION_SPLIT_CONTRACT_VERSION = 1
SPLIT_NAMES = ("train", "val", "test")

__all__ = [
    "EXECUTION_SPLIT_CONTRACT_SCHEMA",
    "EXECUTION_SPLIT_CONTRACT_VERSION",
    "SPLIT_NAMES",
    "DecisionSplitRange",
    "ExecutionSplitContract",
    "load_execution_split_contract",
    "validate_split_contract_payload",
    "validate_signal_split_contract",
    "split_contracts_equal",
    "ranges_for_split",
]


def _nonempty_str(value: str, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()


def _nonnegative_int(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a nonnegative int")
    return value


def _positive_int(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive int")
    return value


def _hash64(value: str, name: str) -> str:
    out = _nonempty_str(value, name)
    if len(out) != 64 or any(ch not in "0123456789abcdef" for ch in out):
        raise ValueError(f"{name} must be 64 lowercase hex characters")
    return out


def _role_name(value: str, name: str) -> str:
    out = _nonempty_str(value, name)
    if out not in SPLIT_NAMES:
        raise ValueError(f"{name} must be one of {SPLIT_NAMES}")
    return out


@dataclass(frozen=True, slots=True)
class DecisionSplitRange:
    role: str
    segment_key: str
    start_decision_row: int
    end_decision_row: int
    start_local_ts_us: int
    end_local_ts_us: int
    embargo_before_us: int = 0
    embargo_after_us: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "role", _role_name(self.role, "role"))
        object.__setattr__(self, "segment_key", _nonempty_str(self.segment_key, "segment_key"))
        start = _nonnegative_int(int(self.start_decision_row), "start_decision_row")
        end = _nonnegative_int(int(self.end_decision_row), "end_decision_row")
        if end <= start:
            raise ValueError("end_decision_row must be greater than start_decision_row")
        object.__setattr__(self, "start_decision_row", start)
        object.__setattr__(self, "end_decision_row", end)
        start_ts = _nonnegative_int(int(self.start_local_ts_us), "start_local_ts_us")
        end_ts = _nonnegative_int(int(self.end_local_ts_us), "end_local_ts_us")
        if end_ts <= start_ts:
            raise ValueError("end_local_ts_us must be greater than start_local_ts_us")
        object.__setattr__(self, "start_local_ts_us", start_ts)
        object.__setattr__(self, "end_local_ts_us", end_ts)
        object.__setattr__(self, "embargo_before_us", _nonnegative_int(int(self.embargo_before_us), "embargo_before_us"))
        object.__setattr__(self, "embargo_after_us", _nonnegative_int(int(self.embargo_after_us), "embargo_after_us"))

    @property
    def row_count(self) -> int:
        return int(self.end_decision_row - self.start_decision_row)

    @property
    def rollout_step_capacity(self) -> int:
        return max(0, self.row_count - 1)

    def as_dict(self) -> dict[str, object]:
        return {
            "role": self.role,
            "segment_key": self.segment_key,
            "start_decision_row": self.start_decision_row,
            "end_decision_row": self.end_decision_row,
            "row_count": self.row_count,
            "start_local_ts_us": self.start_local_ts_us,
            "end_local_ts_us": self.end_local_ts_us,
            "embargo_before_us": self.embargo_before_us,
            "embargo_after_us": self.embargo_after_us,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, object]) -> "DecisionSplitRange":
        if not isinstance(raw, Mapping):
            raise ValueError("split range must be a mapping")
        return cls(
            role=str(raw["role"]),
            segment_key=str(raw["segment_key"]),
            start_decision_row=int(raw["start_decision_row"]),
            end_decision_row=int(raw["end_decision_row"]),
            start_local_ts_us=int(raw["start_local_ts_us"]),
            end_local_ts_us=int(raw["end_local_ts_us"]),
            embargo_before_us=int(raw.get("embargo_before_us", 0)),
            embargo_after_us=int(raw.get("embargo_after_us", 0)),
        )


@dataclass(frozen=True, slots=True)
class ExecutionSplitContract:
    split_source_dataset_root: str
    split_source_dataset_id: str
    split_source_manifest_hash: str
    decision_grid_schema: str
    decision_grid_hash: str
    decision_grid_n_rows: int
    decision_schedule: Mapping[str, object]
    ranges_by_split: Mapping[str, Sequence[DecisionSplitRange]]

    def __post_init__(self) -> None:
        object.__setattr__(self, "split_source_dataset_root", _nonempty_str(self.split_source_dataset_root, "split_source_dataset_root"))
        object.__setattr__(self, "split_source_dataset_id", _nonempty_str(self.split_source_dataset_id, "split_source_dataset_id"))
        object.__setattr__(self, "split_source_manifest_hash", _hash64(self.split_source_manifest_hash, "split_source_manifest_hash"))
        object.__setattr__(self, "decision_grid_schema", _nonempty_str(self.decision_grid_schema, "decision_grid_schema"))
        object.__setattr__(self, "decision_grid_hash", _hash64(self.decision_grid_hash, "decision_grid_hash"))
        object.__setattr__(self, "decision_grid_n_rows", _positive_int(int(self.decision_grid_n_rows), "decision_grid_n_rows"))
        if not isinstance(self.decision_schedule, Mapping):
            raise ValueError("decision_schedule must be a mapping")
        object.__setattr__(self, "decision_schedule", dict(self.decision_schedule))
        ranges: dict[str, tuple[DecisionSplitRange, ...]] = {}
        all_entries: list[DecisionSplitRange] = []
        for role in SPLIT_NAMES:
            raw_entries = self.ranges_by_split.get(role) if isinstance(self.ranges_by_split, Mapping) else None
            if raw_entries is None:
                raise ValueError(f"split contract missing {role} ranges")
            entries = tuple(
                entry if isinstance(entry, DecisionSplitRange) else DecisionSplitRange.from_dict(entry)  # type: ignore[arg-type]
                for entry in raw_entries
            )
            for entry in entries:
                if entry.role != role:
                    raise ValueError(f"split range role mismatch for {role}")
                if entry.end_decision_row > self.decision_grid_n_rows:
                    raise ValueError(f"{role} split range exceeds decision_grid_n_rows")
            previous_end: int | None = None
            previous_start: int | None = None
            for entry in entries:
                if previous_start is not None and entry.start_decision_row < previous_start:
                    raise ValueError(f"{role} split ranges must be sorted by start_decision_row")
                if previous_end is not None and entry.start_decision_row < previous_end:
                    raise ValueError(f"{role} split ranges must not overlap")
                previous_start = entry.start_decision_row
                previous_end = entry.end_decision_row
                all_entries.append(entry)
            ranges[role] = entries
        for previous, current in zip(
            sorted(all_entries, key=lambda item: (item.start_decision_row, item.end_decision_row, item.role)),
            sorted(all_entries, key=lambda item: (item.start_decision_row, item.end_decision_row, item.role))[1:],
        ):
            if current.start_decision_row < previous.end_decision_row:
                raise ValueError(
                    "train/val/test split ranges must not overlap across roles"
                )
        object.__setattr__(self, "ranges_by_split", ranges)

    @property
    def row_counts_by_split(self) -> dict[str, int]:
        return {
            role: int(sum(entry.row_count for entry in entries))
            for role, entries in self.ranges_by_split.items()
        }

    def as_dict(self) -> dict[str, object]:
        return {
            "schema": EXECUTION_SPLIT_CONTRACT_SCHEMA,
            "version": EXECUTION_SPLIT_CONTRACT_VERSION,
            "split_source_dataset_root": self.split_source_dataset_root,
            "split_source_dataset_id": self.split_source_dataset_id,
            "split_source_manifest_hash": self.split_source_manifest_hash,
            "decision_grid_schema": self.decision_grid_schema,
            "decision_grid_hash": self.decision_grid_hash,
            "decision_grid_n_rows": self.decision_grid_n_rows,
            "decision_schedule": dict(self.decision_schedule),
            "ranges_by_split": {
                role: [entry.as_dict() for entry in entries]
                for role, entries in self.ranges_by_split.items()
            },
            "row_counts_by_split": self.row_counts_by_split,
        }


def _manifest_decision_grid_lineage(manifest: storage_manifest.StorageManifest) -> dict[str, object]:
    notes = manifest.notes or {}
    lineage = notes.get("decision_grid")
    if not isinstance(lineage, Mapping):
        raise ValueError("split source manifest notes must include decision_grid lineage")
    required = ("decision_grid_schema", "decision_grid_hash", "decision_grid_n_rows", "decision_schedule")
    missing = [key for key in required if key not in lineage]
    if missing:
        raise ValueError(f"split source manifest decision_grid lineage missing fields: {missing}")
    schedule = dict(lineage["decision_schedule"])  # type: ignore[arg-type]
    if schedule != manifest.decision_schedule:
        raise ValueError("split source manifest decision_grid schedule must match manifest decision_schedule")
    return {
        "decision_grid_schema": _nonempty_str(str(lineage["decision_grid_schema"]), "decision_grid_schema"),
        "decision_grid_hash": _hash64(str(lineage["decision_grid_hash"]), "decision_grid_hash"),
        "decision_grid_n_rows": _positive_int(int(lineage["decision_grid_n_rows"]), "decision_grid_n_rows"),
        "decision_schedule": schedule,
    }


def _validate_manifest_lineage(manifest: storage_manifest.StorageManifest, decision_grid: DecisionGrid) -> dict[str, object]:
    if not isinstance(decision_grid, DecisionGrid):
        raise ValueError("decision_grid must be DecisionGrid")
    lineage = _manifest_decision_grid_lineage(manifest)
    expected = {
        "decision_grid_schema": decision_grid.metadata.schema,
        "decision_grid_hash": decision_grid.decision_grid_hash,
        "decision_grid_n_rows": decision_grid.n_rows,
        "decision_schedule": decision_grid.decision_schedule,
    }
    for key, value in expected.items():
        if lineage[key] != value:
            raise ValueError(f"split source decision grid mismatch for {key}: expected={value!r} actual={lineage[key]!r}")
    return lineage


def _range_from_storage_split(split: storage_manifest.SplitMetadata, decision_grid: DecisionGrid) -> DecisionSplitRange:
    start = int(split.start_row)
    end = int(split.end_row)
    if end > decision_grid.n_rows:
        raise ValueError("split source row range exceeds decision grid rows")
    start_ts = int(decision_grid.decision_local_ts_us[start])
    last_ts = int(decision_grid.decision_local_ts_us[end - 1])
    if start_ts < int(split.local_time_range.start_us) or last_ts >= int(split.local_time_range.end_us):
        raise ValueError("split source row range local timestamps do not match decision grid")
    return DecisionSplitRange(
        role=split.role.value,
        segment_key=split.segment_key,
        start_decision_row=start,
        end_decision_row=end,
        start_local_ts_us=int(split.local_time_range.start_us),
        end_local_ts_us=int(split.local_time_range.end_us),
        embargo_before_us=int(split.embargo_before_us),
        embargo_after_us=int(split.embargo_after_us),
    )


def load_execution_split_contract(split_source_dataset_root: str | Path, decision_grid: DecisionGrid) -> ExecutionSplitContract:
    root = Path(_nonempty_str(str(split_source_dataset_root), "split_source_dataset_root"))
    manifest_path = root / storage_manifest.DEFAULT_MANIFEST_FILENAME
    if not manifest_path.exists():
        raise FileNotFoundError(f"split source manifest not found: {manifest_path}")
    manifest = storage_manifest.read_manifest_json(manifest_path)
    manifest.validate_against_current_code()
    lineage = _validate_manifest_lineage(manifest, decision_grid)
    entries_by_role = {
        role.value: tuple(split for split in manifest.splits if split.role == role)
        for role in (SplitRole.TRAIN, SplitRole.VAL, SplitRole.TEST)
    }
    missing = [role for role, entries in entries_by_role.items() if not entries]
    if missing:
        raise ValueError(f"split source manifest must include train/val/test splits; missing={missing}")
    ranges = {
        role: tuple(_range_from_storage_split(split, decision_grid) for split in entries)
        for role, entries in entries_by_role.items()
    }
    return ExecutionSplitContract(
        split_source_dataset_root=str(root),
        split_source_dataset_id=manifest.dataset_id,
        split_source_manifest_hash=manifest.content_hash(),
        ranges_by_split=ranges,
        **lineage,
    )


def validate_split_contract_payload(value: Mapping[str, object]) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError("split_contract must be a mapping")
    out = dict(value)
    if out.get("schema") != EXECUTION_SPLIT_CONTRACT_SCHEMA:
        raise ValueError("split_contract schema mismatch")
    if int(out.get("version", 0)) != EXECUTION_SPLIT_CONTRACT_VERSION:
        raise ValueError("split_contract version mismatch")
    contract = ExecutionSplitContract(
        split_source_dataset_root=str(out["split_source_dataset_root"]),
        split_source_dataset_id=str(out["split_source_dataset_id"]),
        split_source_manifest_hash=str(out["split_source_manifest_hash"]),
        decision_grid_schema=str(out["decision_grid_schema"]),
        decision_grid_hash=str(out["decision_grid_hash"]),
        decision_grid_n_rows=int(out["decision_grid_n_rows"]),
        decision_schedule=dict(out["decision_schedule"]),  # type: ignore[arg-type]
        ranges_by_split=dict(out["ranges_by_split"]),  # type: ignore[arg-type]
    )
    canonical = contract.as_dict()
    row_counts = out.get("row_counts_by_split")
    if row_counts is not None and dict(row_counts) != canonical["row_counts_by_split"]:  # type: ignore[arg-type]
        raise ValueError("split_contract row_counts_by_split mismatch")
    for key in ("adverse_row_counts", "adverse_dataset_rows_total"):
        if key in out:
            canonical[key] = out[key]
    return canonical


def validate_signal_split_contract(
    *,
    expected: Mapping[str, object],
    actual: Mapping[str, object],
    name: str,
) -> None:
    expected_contract = validate_split_contract_payload(expected)
    actual_contract = validate_split_contract_payload(actual)
    if expected_contract != actual_contract:
        raise ValueError(f"{name} split_contract mismatch")


def split_contracts_equal(left: Mapping[str, object], right: Mapping[str, object]) -> bool:
    return validate_split_contract_payload(left) == validate_split_contract_payload(right)


def ranges_for_split(contract: Mapping[str, object], split: str) -> tuple[DecisionSplitRange, ...]:
    payload = validate_split_contract_payload(contract)
    split = _role_name(split, "split")
    entries = payload["ranges_by_split"][split]  # type: ignore[index]
    return tuple(DecisionSplitRange.from_dict(entry) for entry in entries)  # type: ignore[arg-type]
