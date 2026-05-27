import csv
import gzip
import inspect
import json
from pathlib import Path
import subprocess
import sys

import pytest

pytest.importorskip("pyarrow")
pytest.importorskip("pyarrow.parquet")
pytest.importorskip("polars")

from mmrt import config as cfg
from mmrt.cli import ingest as cli
from mmrt.contracts import TardisDataType
from mmrt.features.labels import LabelResult
from mmrt.schemas import tardis_csv_schema
from mmrt.storage import manifest as mf
from mmrt.storage import reader as rd


def _write_csv(path: Path, header: list[str], rows: list[list[object]]) -> None:
    if path.suffix == ".gz":
        with gzip.open(path, "wt", newline="") as f:
            w = csv.writer(f)
            w.writerow(header)
            w.writerows(rows)
    else:
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(header)
            w.writerows(rows)


def _book_trade_files(
    tmp_path: Path,
    n_book: int = 8,
    n_trade: int = 4,
    symbol: str = cfg.DEFAULT_SYMBOL,
    step_us: int = cfg.DEFAULT_DECISION_STRIDE_US,
):
    b_schema = tardis_csv_schema(TardisDataType.BOOK_SNAPSHOT_25)
    t_schema = tardis_csv_schema(TardisDataType.TRADES)
    bh = list(b_schema.column_names)
    th = list(t_schema.column_names)
    book = tmp_path / "book.csv"
    trade = tmp_path / "trades.csv"
    brows = []
    for i in range(n_book):
        row = []
        for c in bh:
            if c == "exchange": row.append(cfg.DEFAULT_EXCHANGE)
            elif c == "symbol": row.append(symbol)
            elif c == "timestamp": row.append(1_000_000 + i * step_us)
            elif c == "local_timestamp": row.append(1_000_000 + i * step_us)
            elif c.startswith("asks[") and c.endswith("].price"):
                lvl = int(c.split("[")[1].split("]")[0]); row.append(100.1 + lvl * 0.1 + i * 0.01)
            elif c.startswith("asks[") and c.endswith("].amount"):
                row.append(1.0)
            elif c.startswith("bids[") and c.endswith("].price"):
                lvl = int(c.split("[")[1].split("]")[0]); row.append(99.9 - lvl * 0.1 + i * 0.01)
            elif c.startswith("bids[") and c.endswith("].amount"):
                row.append(1.0)
            else:
                row.append("")
        brows.append(row)
    trows = []
    for i in range(n_trade):
        row = []
        for c in th:
            if c == "exchange": row.append(cfg.DEFAULT_EXCHANGE)
            elif c == "symbol": row.append(symbol)
            elif c == "timestamp": row.append(1_000_000 + i * step_us)
            elif c == "local_timestamp": row.append(1_000_000 + i * step_us)
            elif c == "price": row.append(100.0 + i * 0.01)
            elif c == "amount": row.append(0.5)
            elif c == "side": row.append("buy" if i % 2 == 0 else "sell")
            elif "id" in c: row.append(str(i + 1))
            else: row.append("")
        trows.append(row)
    _write_csv(book, bh, brows)
    _write_csv(trade, th, trows)
    return book, trade


def _run_ok(tmp_path: Path, capsys, *extra: str):
    b, t = _book_trade_files(tmp_path, n_book=12, n_trade=8)
    root = tmp_path / "ds"
    rc = cli.main([
        "--dataset-root", str(root), "--dataset-id", "tiny", "--book-csv", str(b), "--trades-csv", str(t),
        "--label-horizons-us", "1000", "--label-entry-delay-us", "1", "--event-batch-size", "2",
        "--chunk-rows", "2", "--row-group-rows", "2", *extra,
    ])
    assert rc == 0
    out = json.loads(capsys.readouterr().out.strip())
    return root, out


def test_public_api_boundary():
    assert cli.__all__ == ["build_arg_parser", "main"]


def test_write_matured_labels_uses_values_bps():
    class W:
        kwargs = None
        def append_values(self, **kwargs):
            self.kwargs = kwargs

    pending = {10: cli.PendingDecision(1, 10, 10, 1, 100.0, (0.1,))}
    label = LabelResult(decision_ts_us=10, entry_ts_us=11, target_ts_us=(12,), values_bps=(1.0, 2.0, 3.0))
    counters = cli.IngestCounters()
    w = W()
    cli._write_matured_labels([label], pending, w, counters)
    assert w.kwargs["label_values"] == (1.0, 2.0, 3.0)
    assert pending == {}
    assert counters.labels_matured == 1
    assert counters.rows_written == 1


def test_parser_defaults(tmp_path: Path):
    b, t = _book_trade_files(tmp_path)
    args = cli.build_arg_parser().parse_args(["--dataset-root", str(tmp_path / "ds"), "--dataset-id", "x", "--book-csv", str(b), "--trades-csv", str(t)])
    assert args.exchange == cfg.DEFAULT_EXCHANGE
    assert args.symbol == cfg.DEFAULT_SYMBOL
    assert args.book_data_type == "book_snapshot_25"
    assert args.validate_output is True
    assert args.decision_stride_us == cfg.DEFAULT_DECISION_STRIDE_US


def test_reject_unsupported_book_data_type(tmp_path: Path):
    b, t = _book_trade_files(tmp_path)
    with pytest.raises(ValueError, match="book_snapshot_25"):
        cli.main(["--dataset-root", str(tmp_path / "ds"), "--dataset-id", "x", "--book-csv", str(b), "--trades-csv", str(t), "--book-data-type", "incremental_book_L2"])


def test_rejects_other_unsupported_book_type(tmp_path: Path):
    b, t = _book_trade_files(tmp_path)
    with pytest.raises(ValueError, match="supports only book_snapshot_25"):
        cli.main(["--dataset-root", str(tmp_path / "ds"), "--dataset-id", "x", "--book-csv", str(b), "--trades-csv", str(t), "--book-data-type", "book_snapshot_5"])


def test_rejects_non_default_decision_stride(tmp_path: Path):
    b, t = _book_trade_files(tmp_path)
    root, wd = tmp_path / "ds", tmp_path / "wd"
    with pytest.raises(ValueError, match="500_000"):
        cli.main([
            "--dataset-root", str(root),
            "--dataset-id", "x",
            "--book-csv", str(b),
            "--trades-csv", str(t),
            "--decision-stride-us", "500",
            "--work-dir", str(wd),
        ])
    assert not (root / "manifest.json").exists()
    assert not wd.exists()


def test_parse_us_range():
    assert cli._parse_us_range("100:200", "x") == (100, 200)
    for bad in ["200:100", "abc:200", "100", "100:200:300"]:
        with pytest.raises(ValueError):
            cli._parse_us_range(bad, "x")


def test_no_stale_imports_source_residue():
    src = inspect.getsource(cli)
    for token in ["BY" + "BIT", "CM" + "SSL", "offline_" + "ingest", "linear_" + "offline", "Mini" + "Rocket", "Multi" + "Rocket", "Hy" + "dra", "Ae" + "on", "sk" + "learn", "to" + "rch", "P" + "CA", "Standard" + "Scaler", "stage" + "1", "stage" + "2", "stage" + "3", "stage" + "4", "stage" + "5", "pan" + "das"]:
        assert token not in src
    for token in ["mmrt." + "linear", "read_" + "split_" + "table", "read_" + "ta" + "ble(", "to_" + "pan" + "das"]:
        assert token not in src


def test_ingest_uses_canonical_market_symbol(tmp_path: Path, capsys):
    b, t = _book_trade_files(tmp_path)
    root = tmp_path / "ds"
    cli.main(["--dataset-root", str(root), "--dataset-id", "x", "--book-csv", str(b), "--trades-csv", str(t), "--symbol", cfg.DEFAULT_SYMBOL.lower(), "--label-horizons-us", "1000", "--label-entry-delay-us", "1"])
    out = json.loads(capsys.readouterr().out.strip())
    man = mf.read_manifest_json(root / "manifest.json")
    assert man.pipeline_config.market.symbol == cfg.DEFAULT_SYMBOL
    assert out["symbol"] == cfg.DEFAULT_SYMBOL


def test_ingest_rejects_csv_market_mismatch(tmp_path: Path, capsys):
    b, t = _book_trade_files(tmp_path, symbol="ETHUSDT")
    root = tmp_path / "ds"
    with pytest.raises(ValueError, match="market mismatch"):
        cli.main(["--dataset-root", str(root), "--dataset-id", "x", "--book-csv", str(b), "--trades-csv", str(t)])
    assert capsys.readouterr().out.strip() == ""
    assert not (root / "manifest.json").exists()


def test_manifest_notes_include_complete_ingest_counters(tmp_path: Path, capsys):
    root, _ = _run_ok(tmp_path, capsys)
    man = mf.read_manifest_json(root / "manifest.json")
    c = man.notes["ingest_counters"]
    assert c["input_book_files"] == 1 and c["input_trade_files"] == 1 and c["normalized_files"] == 2
    assert c["output_segments"] == len(man.segments)
    assert c["output_rows"] == man.total_rows
    assert c["rows_written"] == man.total_rows


def test_max_events_counts_only_processed_rows(tmp_path: Path, capsys):
    b, t = _book_trade_files(tmp_path, n_book=12, n_trade=8)
    root = tmp_path / "ds"

    rc = cli.main([
        "--dataset-root", str(root),
        "--dataset-id", "x",
        "--book-csv", str(b),
        "--trades-csv", str(t),
        "--max-events", "18",
        "--label-horizons-us", "1000",
        "--label-entry-delay-us", "1",
        "--event-batch-size", "2",
        "--chunk-rows", "2",
        "--row-group-rows", "2",
    ])

    assert rc == 0
    out = json.loads(capsys.readouterr().out.strip())
    man = mf.read_manifest_json(root / "manifest.json")

    assert man.notes["ingest_counters"]["merged_events_seen"] == 18
    assert out["rows_written"] == man.total_rows
    assert man.total_rows > 0


def test_work_dir_removed_on_success(tmp_path: Path, capsys):
    wd = tmp_path / "wd"
    _, out = _run_ok(tmp_path, capsys, "--work-dir", str(wd))
    assert not wd.exists()
    assert out["work_dir_removed"] is True


def test_work_dir_preserved_on_failure(tmp_path: Path):
    b, t = _book_trade_files(tmp_path, symbol="ETHUSDT")
    root, wd = tmp_path / "ds", tmp_path / "wd"
    with pytest.raises(ValueError, match="market mismatch"):
        cli.main(["--dataset-root", str(root), "--dataset-id", "x", "--book-csv", str(b), "--trades-csv", str(t), "--work-dir", str(wd)])
    assert wd.exists()
    assert not (root / "manifest.json").exists()


def test_end_to_end_with_explicit_splits(tmp_path: Path, capsys):
    root, out = _run_ok(tmp_path, capsys, "--split-train", "1000000:3500000", "--split-val", "3500000:6500000")
    man = mf.read_manifest_json(root / "manifest.json")
    roles = {s.role.value for s in man.splits}
    assert "train" in roles and "val" in roles
    assert out["splits_written"] is True
    assert "train" in out["split_roles"] and "val" in out["split_roles"]


def test_reject_partial_split_args(tmp_path: Path):
    b, t = _book_trade_files(tmp_path)
    root, wd = tmp_path / "ds", tmp_path / "wd"
    with pytest.raises(ValueError, match="both --split-train and --split-val"):
        cli.main(["--dataset-root", str(root), "--dataset-id", "x", "--book-csv", str(b), "--trades-csv", str(t), "--split-train", "1:2", "--work-dir", str(wd)])
    assert not (root / "manifest.json").exists()
    assert not wd.exists()


def test_pending_eof_decisions_are_not_force_labeled(tmp_path: Path, capsys):
    b, t = _book_trade_files(tmp_path, n_book=12, n_trade=8)
    root = tmp_path / "ds"
    cli.main(["--dataset-root", str(root), "--dataset-id", "x", "--book-csv", str(b), "--trades-csv", str(t), "--label-horizons-us", "1500000", "--label-entry-delay-us", "1"])
    out = json.loads(capsys.readouterr().out.strip())
    assert out["pending_decisions_at_eof"] > 0
    assert out["rows_written"] < out["decisions_emitted"]
    r = rd.open_dataset(str(root), validate_on_open=True)
    assert r.total_rows == out["rows_written"]


def test_stdout_summary_is_compact_and_has_no_large_state(tmp_path: Path, capsys):
    _, out = _run_ok(tmp_path, capsys)
    assert {"status", "dataset_root", "dataset_id", "segments", "rows", "decisions_emitted", "rows_written"}.issubset(out)
    for k in ["feature_values", "features", "label_values", "model", "preprocess", "diagnostics_state", "transform_state"]:
        assert k not in out
    assert len(json.dumps(out)) < 5000


def test_existing_manifest_fails_before_work(tmp_path: Path):
    b, t = _book_trade_files(tmp_path)
    root, wd = tmp_path / "ds", tmp_path / "wd"
    root.mkdir(parents=True)
    (root / "manifest.json").write_text("{}")
    with pytest.raises(FileExistsError, match="manifest already exists"):
        cli.main(["--dataset-root", str(root), "--dataset-id", "x", "--book-csv", str(b), "--trades-csv", str(t), "--work-dir", str(wd)])
    assert not wd.exists()


def test_existing_segments_fail_before_work(tmp_path: Path):
    b, t = _book_trade_files(tmp_path)
    root, wd = tmp_path / "ds", tmp_path / "wd"
    (root / "segments").mkdir(parents=True)
    (root / "segments" / "x.parquet").write_bytes(b"x")
    with pytest.raises(FileExistsError, match="existing parquet segments"):
        cli.main(["--dataset-root", str(root), "--dataset-id", "x", "--book-csv", str(b), "--trades-csv", str(t), "--work-dir", str(wd)])
    assert not wd.exists()


def test_subprocess_help_entrypoint():
    p = subprocess.run(
        [sys.executable, "-m", "mmrt.cli.ingest", "--help"],
        capture_output=True,
        text=True,
    )
    assert p.returncode == 0
    assert "--dataset-root" in p.stdout
    assert "--book-csv" in p.stdout
    assert "--trades-csv" in p.stdout
    assert "--split-train" in p.stdout
