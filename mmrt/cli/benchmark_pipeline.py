"""Run deterministic MMRT pipeline performance baselines."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from mmrt.analysis.performance_baseline import PipelineBenchmarkConfig, run_pipeline_benchmarks

__all__ = ["build_arg_parser", "main"]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--work-root")
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--inventory-min-bytes", type=int, default=13_000)
    parser.add_argument("--no-optional", action="store_true")
    parser.add_argument("--output-json")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    result = run_pipeline_benchmarks(
        PipelineBenchmarkConfig(
            iterations=args.iterations,
            include_optional=not args.no_optional,
            inventory_min_bytes=args.inventory_min_bytes,
            work_root=args.work_root,
        ),
        repo_root=args.repo_root,
    )
    text = json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + "\n"
    if args.output_json:
        path = Path(args.output_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    else:
        print(text, end="")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
