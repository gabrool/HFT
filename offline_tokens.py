from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


def read_json(path: Path) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def load_global_meta(out_root: Path) -> dict:
    meta_path = out_root / "meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Not found: {meta_path}. Did you run offline_ingest.py?")
    meta = read_json(meta_path)
    if not (isinstance(meta.get("weeks", []), list) or isinstance(meta.get("week_counts", {}), dict)):
        raise ValueError("Malformed meta.json")
    return meta


def resolve_week_meta_paths(out_root: Path, meta: dict) -> List[Path]:
    w2m = meta.get("weeks_meta", {})
    weeks = meta.get("weeks", [])
    if w2m and weeks:
        return [out_root / w2m[w] for w in weeks if w in w2m]
    paths = []
    for w in weeks:
        p = out_root / w / "meta_week.json"
        if p.exists():
            paths.append(p)
    return paths


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
