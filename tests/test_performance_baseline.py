import json

from mmrt.analysis.performance_baseline import (
    PERFORMANCE_BASELINE_SCHEMA,
    PipelineBenchmarkConfig,
    build_pipeline_golden_outputs,
    heavy_pipeline_file_inventory,
    run_pipeline_benchmarks,
)
from mmrt.cli.benchmark_pipeline import main as benchmark_main


def test_heavy_pipeline_inventory_is_sorted_and_covers_hot_modules():
    rows = heavy_pipeline_file_inventory(".", min_bytes=13_000)

    assert rows == sorted(rows, key=lambda row: (-int(row["bytes"]), str(row["path"])))
    paths = {str(row["path"]) for row in rows}
    assert "mmrt/execution/env.py" in paths
    assert "mmrt/cli/build_execution_tape.py" in paths
    assert "mmrt/linear/train.py" in paths
    assert "mmrt/rl/ppo.py" in paths


def test_golden_outputs_are_deterministic(tmp_path):
    first = build_pipeline_golden_outputs(tmp_path / "first")
    second = build_pipeline_golden_outputs(tmp_path / "second")

    assert first == second
    assert first["schema"] == PERFORMANCE_BASELINE_SCHEMA
    assert first["fixture"]["decision_grid_rows"] == 4
    assert first["reason_counts"]["first_valid_book"] == 1


def test_pipeline_benchmarks_return_expected_cases(tmp_path):
    result = run_pipeline_benchmarks(
        PipelineBenchmarkConfig(iterations=1, include_optional=False, work_root=str(tmp_path / "bench")),
        repo_root=".",
    )

    assert result["schema"] == PERFORMANCE_BASELINE_SCHEMA
    case_names = {case["name"] for case in result["benchmarks"]}
    assert case_names == {
        "execution_tape_materialized",
        "execution_tape_streaming_writer",
        "decision_grid_build",
        "labels_batch",
        "linear_direction_update",
    }
    for case in result["benchmarks"]:
        assert case["iterations"] == 1
        assert case["min_ms"] >= 0.0


def test_benchmark_cli_writes_json(tmp_path):
    output = tmp_path / "baseline.json"

    exit_code = benchmark_main(
        [
            "--iterations",
            "1",
            "--no-optional",
            "--work-root",
            str(tmp_path / "work"),
            "--output-json",
            str(output),
        ]
    )

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert exit_code == 0
    assert payload["schema"] == PERFORMANCE_BASELINE_SCHEMA
    assert payload["config"]["iterations"] == 1
