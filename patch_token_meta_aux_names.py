from __future__ import annotations

import argparse
import json
from pathlib import Path

from CMSSL17 import AUX_DIM, FEATURE_AUX_TAIL


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens-root", required=True)
    args = ap.parse_args()

    root = Path(args.tokens_root)
    meta_path = root / "meta.json"
    if not meta_path.exists():
        raise SystemExit(f"Missing meta.json: {meta_path}")

    meta = json.loads(meta_path.read_text())
    aux_feature_names = list(meta.get("aux_feature_names") or [])
    canonical = list(FEATURE_AUX_TAIL)

    if len(canonical) != int(AUX_DIM):
        raise SystemExit(f"Canonical AUX names mismatch: len={len(canonical)} AUX_DIM={AUX_DIM}")

    if len(aux_feature_names) == int(AUX_DIM):
        print("[patch-meta] aux_feature_names already present; no changes")
        return

    feature_names = list(meta.get("feature_names") or [])
    feature_dim_total = int(meta.get("feature_dim_total", -1))
    if feature_dim_total - len(feature_names) != len(canonical):
        raise SystemExit(
            f"Cannot infer aux names safely: feature_dim_total={feature_dim_total} len(feature_names)={len(feature_names)}"
        )

    backup_path = root / "meta.json.bak_before_aux_names"
    backup_path.write_text(meta_path.read_text())
    meta["aux_feature_names"] = canonical
    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")
    print(f"[patch-meta] wrote aux_feature_names ({len(canonical)}) to {meta_path}")


if __name__ == "__main__":
    main()
