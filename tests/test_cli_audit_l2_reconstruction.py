from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from mmrt.cli.audit_l2_reconstruction import (
    L2AuditConfig,
    _write_json_atomic,
    audit_l2_reconstruction,
    build_arg_parser,
)


def _write_l2_csv(path: Path, rows: list[dict[str, object]]) -> Path:
    columns = ["timestamp", "local_timestamp", "is_snapshot", "side", "price", "amount"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    return path


def _good_rows() -> list[dict[str, object]]:
    return [
        {"timestamp": 90, "local_timestamp": 100, "is_snapshot": "true", "side": "bid", "price": 100.0, "amount": 1.0},
        {"timestamp": 91, "local_timestamp": 100, "is_snapshot": "true", "side": "ask", "price": 100.2, "amount": 1.0},
        {"timestamp": 190, "local_timestamp": 200, "is_snapshot": "false", "side": "bid", "price": 100.1, "amount": 1.0},
        {"timestamp": 290, "local_timestamp": 300, "is_snapshot": "false", "side": "bid", "price": 100.0, "amount": 0.0},
    ]


def test_basic_audit_succeeds(tmp_path: Path) -> None:
    path = _write_l2_csv(tmp_path / "l2.csv", _good_rows())
    report = audit_l2_reconstruction(L2AuditConfig(l2_inputs=(str(path),)))

    assert report["status"] == "ok"
    assert report["counts"]["rows_seen"] == len(_good_rows())
    assert report["reconstruction"]["is_ready"] is True
    assert report["reconstruction"]["snapshot_reset_count"] == 1
    assert report["counts"]["events_emitted"] > 0
    assert report["reconstruction"]["max_bid_depth"] >= 1
    assert report["reconstruction"]["max_ask_depth"] >= 1
    assert "no_snapshot_seen" not in report["warnings"]


def test_pre_snapshot_skip_warning(tmp_path: Path) -> None:
    rows = [
        {"timestamp": 10, "local_timestamp": 10, "is_snapshot": "false", "side": "bid", "price": 99.9, "amount": 1.0},
        *_good_rows(),
    ]
    path = _write_l2_csv(tmp_path / "l2.csv", rows)
    report = audit_l2_reconstruction(L2AuditConfig(l2_inputs=(str(path),)))

    assert report["reconstruction"]["skipped_pre_snapshot_updates"] > 0
    assert "high_pre_snapshot_skip_fraction" in report["warnings"]


def test_no_snapshot(tmp_path: Path) -> None:
    rows = [
        {"timestamp": 10, "local_timestamp": 10, "is_snapshot": "false", "side": "bid", "price": 100.0, "amount": 1.0},
        {"timestamp": 20, "local_timestamp": 20, "is_snapshot": "false", "side": "ask", "price": 100.2, "amount": 1.0},
    ]
    path = _write_l2_csv(tmp_path / "l2.csv", rows)
    report = audit_l2_reconstruction(L2AuditConfig(l2_inputs=(str(path),)))

    assert report["status"] == "ok"
    assert report["reconstruction"]["is_ready"] is False
    assert report["counts"]["events_emitted"] == 0
    assert "no_snapshot_seen" in report["warnings"]
    assert "no_reconstructed_events" in report["warnings"]


def test_crossed_repair_observed(tmp_path: Path) -> None:
    rows = [
        {"timestamp": 90, "local_timestamp": 100, "is_snapshot": "true", "side": "bid", "price": 100.0, "amount": 1.0},
        {"timestamp": 91, "local_timestamp": 100, "is_snapshot": "true", "side": "ask", "price": 100.2, "amount": 1.0},
        {"timestamp": 190, "local_timestamp": 200, "is_snapshot": "false", "side": "bid", "price": 100.3, "amount": 1.0},
    ]
    path = _write_l2_csv(tmp_path / "l2.csv", rows)
    report = audit_l2_reconstruction(L2AuditConfig(l2_inputs=(str(path),)))

    assert report["reconstruction"]["crossed_repair_count"] > 0
    assert "crossed_repairs_observed" in report["warnings"]


def test_missing_delete_observed(tmp_path: Path) -> None:
    rows = [
        *_good_rows()[:2],
        {"timestamp": 190, "local_timestamp": 200, "is_snapshot": "false", "side": "bid", "price": 99.0, "amount": 0.0},
    ]
    path = _write_l2_csv(tmp_path / "l2.csv", rows)
    report = audit_l2_reconstruction(L2AuditConfig(l2_inputs=(str(path),)))

    assert report["reconstruction"]["missing_delete_count"] > 0
    assert "missing_deletes_observed" in report["warnings"]


def test_max_rows(tmp_path: Path) -> None:
    path = _write_l2_csv(tmp_path / "l2.csv", _good_rows())
    report = audit_l2_reconstruction(L2AuditConfig(l2_inputs=(str(path),), max_rows=2))

    assert report["counts"]["rows_seen"] == 2
    assert "scan_limit_hit" in report["warnings"]


def test_output_json(tmp_path: Path) -> None:
    path = tmp_path / "report.json"
    returned = _write_json_atomic({"status": "ok"}, str(path))

    assert returned == str(path)
    assert path.exists()
    assert json.loads(path.read_text(encoding="utf-8")) == {"status": "ok"}


def test_cli_parser_smoke() -> None:
    args = build_arg_parser().parse_args(
        [
            "--l2-input",
            "a.csv",
            "b.parquet",
            "--exchange",
            "ex",
            "--symbol",
            "SYM",
            "--tick-size",
            "0.5",
            "--max-rows",
            "10",
            "--batch-size",
            "128",
            "--sample-event-limit",
            "2",
        ]
    )

    assert args.l2_inputs == ["a.csv", "b.parquet"]
    assert args.exchange == "ex"
    assert args.symbol == "SYM"
    assert args.tick_size == 0.5
    assert args.max_rows == 10
    assert args.batch_size == 128
    assert args.sample_event_limit == 2


def test_rejects_unsorted_local_timestamp(tmp_path: Path) -> None:
    rows = [
        {"timestamp": 90, "local_timestamp": 100, "is_snapshot": "true", "side": "bid", "price": 100.0, "amount": 1.0},
        {"timestamp": 91, "local_timestamp": 100, "is_snapshot": "true", "side": "ask", "price": 100.2, "amount": 1.0},
        {"timestamp": 80, "local_timestamp": 90, "is_snapshot": "false", "side": "bid", "price": 99.9, "amount": 1.0},
    ]
    path = _write_l2_csv(tmp_path / "l2.csv", rows)

    with pytest.raises(ValueError, match="nondecreasing local_ts_us"):
        audit_l2_reconstruction(L2AuditConfig(l2_inputs=(str(path),)))


def test_no_forbidden_imports() -> None:
    source = Path("mmrt/cli/audit_l2_reconstruction.py").read_text(encoding="utf-8")

    assert "import torch" not in source
    assert "import pandas" not in source
    assert "import sklearn" not in source


def test_parquet_input(tmp_path: Path) -> None:
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    rows = _good_rows()
    path = tmp_path / "l2.parquet"
    pq.write_table(pa.Table.from_pylist(rows), path)

    report = audit_l2_reconstruction(L2AuditConfig(l2_inputs=(str(path),)))

    assert report["status"] == "ok"
    assert report["counts"]["rows_seen"] == len(rows)
    assert report["reconstruction"]["snapshot_reset_count"] == 1
