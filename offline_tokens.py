from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from CMSSL17 import TIME_GRID_GUARD_MS, TIME_GRID_STEP_MS

# Time-grid contract is centralized in CMSSL17.py.


def read_json(path: Path) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def load_global_meta(out_root: Path) -> dict:
    meta_path = out_root / "meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Not found: {meta_path}. Did you run offline_ingest.py?")
    meta = read_json(meta_path)

    def _meta_error(detail: str) -> ValueError:
        return ValueError(f"Malformed meta.json: {detail}. Please rerun offline_ingest.py.")

    weeks_in_order = meta.get("weeks_in_order")
    if not (isinstance(weeks_in_order, list) and weeks_in_order):
        raise _meta_error("missing non-empty 'weeks_in_order'")

    weeks_meta = meta.get("weeks_meta")
    if not (isinstance(weeks_meta, dict) and weeks_meta):
        raise _meta_error("missing non-empty 'weeks_meta'")

    time_grid = meta.get("time_grid")
    if not isinstance(time_grid, dict):
        raise _meta_error("missing 'time_grid' object")

    step_ms = time_grid.get("step_ms")
    if step_ms != TIME_GRID_STEP_MS:
        raise _meta_error(
            f"'time_grid.step_ms' must be {TIME_GRID_STEP_MS} (got {step_ms!r})"
        )

    guard_ms = time_grid.get("guard_ms")
    if guard_ms != TIME_GRID_GUARD_MS:
        raise _meta_error(
            f"'time_grid.guard_ms' must be {TIME_GRID_GUARD_MS} (got {guard_ms!r})"
        )

    decision_policy = meta.get("decision_policy")
    if not isinstance(decision_policy, str) or (
        decision_policy != "ob_only_grid_quantized" and "grid" not in decision_policy
    ):
        raise _meta_error(
            "missing/invalid 'decision_policy' (must be 'ob_only_grid_quantized' or contain 'grid')"
        )

    week_counts = meta.get("week_counts")
    if week_counts is not None and not isinstance(week_counts, dict):
        raise _meta_error("'week_counts' must be a dict when present")

    return meta


def resolve_week_meta_paths(out_root: Path, meta: dict) -> List[Path]:
    weeks = meta.get("weeks_in_order")
    if not (isinstance(weeks, list) and weeks):
        raise ValueError("Malformed meta.json: missing non-empty 'weeks_in_order'.")

    weeks_meta = meta.get("weeks_meta")
    if not isinstance(weeks_meta, dict) or not weeks_meta:
        raise ValueError("Malformed meta.json: missing non-empty 'weeks_meta' mapping.")

    missing_weeks = [w for w in weeks if w not in weeks_meta]
    if missing_weeks:
        sample_missing = ", ".join(missing_weeks[:5])
        if len(missing_weeks) > 5:
            sample_missing += ", ..."
        raise ValueError(
            "Malformed meta.json: 'weeks_meta' is missing entries for weeks listed in "
            f"'weeks_in_order': {sample_missing}."
        )

    return [out_root / weeks_meta[w] for w in weeks]


def iter_week_chunks(out_root: Path, meta: Optional[dict] = None) -> Iterable[Tuple[str, dict, Path]]:
    if meta is None:
        meta = load_global_meta(out_root)
    for week_meta_path in resolve_week_meta_paths(out_root, meta):
        week_meta = read_json(week_meta_path)
        week_dir = week_meta_path.parent
        week_key = week_meta.get("week", week_dir.name)
        yield week_key, week_meta, week_dir


@dataclass
class ChunkRef:
    week_dir: Path
    core_file: Path
    aux_file: Path
    y_file: Path
    n: int
    offset: int = 0


def build_chunk_refs(meta_week_path: Path) -> List[ChunkRef]:
    wmeta = read_json(meta_week_path)
    week_dir = meta_week_path.parent
    refs: List[ChunkRef] = []
    for ch in wmeta.get("chunks", []):
        files = ch["files"]
        refs.append(ChunkRef(
            week_dir=week_dir,
            core_file=week_dir / files["core"],
            aux_file=week_dir / files["aux"],
            y_file=week_dir / files["y"],
            n=int(ch["n"]),
            offset=0,
        ))
    return refs


def slice_week_chunks(meta_week_path: Path, start_idx: int, end_idx: int) -> List[ChunkRef]:
    """
    Build ChunkRefs that cover only [start_idx, end_idx) of a given week,
    assuming chunks are in chronological order.
    """
    assert 0 <= start_idx <= end_idx
    wmeta = read_json(meta_week_path)
    week_dir = meta_week_path.parent
    chunks = wmeta.get("chunks", [])
    refs: List[ChunkRef] = []

    cursor = 0  # global index of first row in current chunk
    for ch in chunks:
        ch_n = int(ch["n"])
        chunk_start = cursor
        chunk_end = cursor + ch_n

        s = max(start_idx, chunk_start)
        e = min(end_idx, chunk_end)
        if e > s:
            offset_in_chunk = s - chunk_start
            n_here = e - s
            files = ch["files"]
            refs.append(ChunkRef(
                week_dir=week_dir,
                core_file=week_dir / files["core"],
                aux_file=week_dir / files["aux"],
                y_file=week_dir / files["y"],
                n=n_here,
                offset=offset_in_chunk,
            ))

        cursor = chunk_end
        if cursor >= end_idx:
            break

    return refs
