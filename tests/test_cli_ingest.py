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


def _book_trade_files(tmp_path: Path, *, n_book: int = 12, n_trade: int = 10, step_us: int = 1000):
    b_schema = tardis_csv_schema(TardisDataType.BOOK_SNAPSHOT_25)
    t_schema = tardis_csv_schema(TardisDataType.TRADES)
    bh = list(b_schema.column_names)
    th = list(t_schema.column_names)
    book = tmp_path / "book.csv"
    trade = tmp_path / "trades.csv"
    brows = []
    for i in range(n_book):
        row = []
        ts = 1_000_000 + i * step_us
        for c in bh:
            if c == "exchange":
                row.append(cfg.DEFAULT_EXCHANGE)
            elif c == "symbol":
                row.append(cfg.DEFAULT_SYMBOL)
            elif c == "timestamp":
                row.append(ts)
            elif c == "local_timestamp":
                row.append(ts)
            elif c.startswith("asks[") and c.endswith("].price"):
                lvl = int(c.split("[")[1].split("]")[0])
                row.append(100.1 + lvl * 0.1 + i * 0.01)
            elif c.startswith("asks[") and c.endswith("].amount"):
                row.append(1.0)
            elif c.startswith("bids[") and c.endswith("].price"):
                lvl = int(c.split("[")[1].split("]")[0])
                row.append(99.9 - lvl * 0.1 + i * 0.01)
            elif c.startswith("bids[") and c.endswith("].amount"):
                row.append(1.0)
            else:
                row.append("")
        brows.append(row)

    trows = []
    for i in range(n_trade):
        row = []
        ts = 1_000_000 + i * step_us
        for c in th:
            if c == "exchange":
                row.append(cfg.DEFAULT_EXCHANGE)
            elif c == "symbol":
                row.append(cfg.DEFAULT_SYMBOL)
            elif c == "timestamp":
                row.append(ts)
            elif c == "local_timestamp":
                row.append(ts)
            elif c == "price":
                row.append(100.0 + i * 0.01)
            elif c == "amount":
                row.append(0.5)
            elif c == "side":
                row.append("buy" if i % 2 == 0 else "sell")
            elif "id" in c:
                row.append(str(i + 1))
            else:
                row.append("")
        trows.append(row)

    _write_csv(book, bh, brows)
    _write_csv(trade, th, trows)
    return book, trade


def test_public_api_boundary():
    assert cli.__all__ == ["build_arg_parser", "main"]


def test_parser_defaults(tmp_path: Path):
    b, t = _book_trade_files(tmp_path)
    args = cli.build_arg_parser().parse_args(["--dataset-root", str(tmp_path / "ds"), "--dataset-id", "x", "--book-csv", str(b), "--trades-csv", str(t)])
    assert args.exchange == cfg.DEFAULT_EXCHANGE
    assert args.symbol == cfg.DEFAULT_SYMBOL
    assert args.book_data_type == "book_snapshot_25"
    assert args.validate_output is True


def test_reject_unsupported_book_data_type(tmp_path: Path):
    b, t = _book_trade_files(tmp_path)
    with pytest.raises(ValueError, match="book_snapshot_25"):
        cli.main(["--dataset-root", str(tmp_path / "ds"), "--dataset-id", "x", "--book-csv", str(b), "--trades-csv", str(t), "--book-data-type", "incremental_book_L2"])


def test_rejects_other_unsupported_book_type(tmp_path: Path):
    b, t = _book_trade_files(tmp_path)
    with pytest.raises(ValueError, match="supports only book_snapshot_25"):
        cli.main(["--dataset-root", str(tmp_path / "ds"), "--dataset-id", "x", "--book-csv", str(b), "--trades-csv", str(t), "--book-data-type", "book_snapshot_5"])


def test_parse_us_range():
    assert cli._parse_us_range("100:200", "x") == (100, 200)
    for bad in ["200:100", "abc:200", "100", "100:200:300"]:
        with pytest.raises(ValueError):
            cli._parse_us_range(bad, "x")


def test_no_stale_imports_source_residue():
    src = inspect.getsource(cli)
    for token in [
        "BY" + "BIT", "CM" + "SSL", "offline_" + "ingest", "linear_" + "offline", "Mini" + "Rocket", "Multi" + "Rocket",
        "Hy" + "dra", "Ae" + "on", "sk" + "learn", "to" + "rch", "P" + "CA", "Standard" + "Scaler",
        "stage" + "1", "stage" + "2", "stage" + "3", "stage" + "4", "stage" + "5", "pan" + "das",
    ]:
        assert token not in src
    for token in ["mmrt." + "linear", "read_" + "split_table", "read_" + "table(", "to_" + "pan" + "das"]:
        assert token not in src


def test_write_matured_labels_uses_values_bps():
    pending = {123: cli.PendingDecision(1, 123, 123, 1, 100.0, (0.1, 0.2))}
    label = LabelResult(decision_ts_us=123, entry_ts_us=125, values_bps=(1.0, 2.0, 3.0))

    class W:
        kwargs = None

        def append_values(self, **kwargs):
            self.kwargs = kwargs

    counters = cli.IngestCounters()
    writer = W()
    cli._write_matured_labels([label], pending, writer, counters)
    assert writer.kwargs["label_values"] == (1.0, 2.0, 3.0)
    assert pending == {}
    assert counters.labels_matured == 1
    assert counters.rows_written == 1


def _run_ok(tmp_path: Path, capsys, extra: list[str] | None = None):
    b, t = _book_trade_files(tmp_path)
    root = tmp_path / "ds"
    argv = ["--dataset-root", str(root), "--dataset-id", "tiny", "--book-csv", str(b), "--trades-csv", str(t), "--label-horizons-us", "1000", "--label-entry-delay-us", "1", "--decision-stride-us", "1000", "--event-batch-size", "2", "--chunk-rows", "2", "--row-group-rows", "2"]
    if extra:
        argv.extend(extra)
    rc = cli.main(argv)
    out = json.loads(capsys.readouterr().out.strip())
    return rc, out, root


def test_end_to_end_smoke(tmp_path: Path, capsys):
    rc, out, root = _run_ok(tmp_path, capsys)
    assert rc == 0
    assert out["status"] == "ok"
    man = mf.read_manifest_json(root / "manifest.json")
    assert man.pipeline_config.market.exchange == cfg.DEFAULT_EXCHANGE
    assert man.pipeline_config.market.symbol == cfg.DEFAULT_SYMBOL


def test_ingest_uses_canonical_market_symbol(tmp_path: Path, capsys):
    b, t = _book_trade_files(tmp_path)
    root = tmp_path / "ds"
    cli.main(["--dataset-root", str(root), "--dataset-id", "x", "--book-csv", str(b), "--trades-csv", str(t), "--symbol", cfg.DEFAULT_SYMBOL.lower(), "--label-horizons-us", "1000", "--label-entry-delay-us", "1", "--decision-stride-us", "1000"])
    out = json.loads(capsys.readouterr().out.strip())
    man = mf.read_manifest_json(root / "manifest.json")
    assert man.pipeline_config.market.symbol == cfg.DEFAULT_SYMBOL
    assert out["symbol"] == cfg.DEFAULT_SYMBOL


def test_ingest_rejects_csv_market_mismatch(tmp_path: Path):
    b, t = _book_trade_files(tmp_path)
    lines = b.read_text().splitlines()
    lines[2] = lines[2].replace(cfg.DEFAULT_SYMBOL, "ETHUSDT")
    b.write_text("\n".join(lines) + "\n")
    root = tmp_path / "ds"
    with pytest.raises(ValueError, match="market mismatch"):
        cli.main(["--dataset-root", str(root), "--dataset-id", "x", "--book-csv", str(b), "--trades-csv", str(t)])
    assert not (root / "manifest.json").exists()


def test_manifest_notes_include_complete_ingest_counters(tmp_path: Path, capsys):
    _, _, root = _run_ok(tmp_path, capsys)
    man = mf.read_manifest_json(root / "manifest.json")
    ctrs = man.notes["ingest_counters"]
    assert ctrs["input_book_files"] == 1
    assert ctrs["input_trade_files"] == 1
    assert ctrs["normalized_files"] == 2
    assert ctrs["output_segments"] == len(man.segments)
    assert ctrs["output_rows"] == man.total_rows
    assert ctrs["rows_written"] == man.total_rows


def test_max_events_counts_only_processed_rows(monkeypatch, tmp_path: Path):
    b, t = _book_trade_files(tmp_path)
    root = tmp_path / "ds"
    seen = {}
    orig = cli._run_causal_ingest

    def wrapped(*args, **kwargs):
        counters, tcfg, tdiag = orig(*args, **kwargs)
        seen["n"] = counters.merged_events_seen
        return counters, tcfg, tdiag

    monkeypatch.setattr(cli, "_run_causal_ingest", wrapped)
    cli.main(["--dataset-root", str(root), "--dataset-id", "x", "--book-csv", str(b), "--trades-csv", str(t), "--label-horizons-us", "1000", "--label-entry-delay-us", "1", "--decision-stride-us", "1000", "--max-events", "2"])
    assert seen["n"] == 2


def test_work_dir_removed_on_success(tmp_path: Path, capsys):
    work = tmp_path / "work"
    _, out, _ = _run_ok(tmp_path, capsys, ["--work-dir", str(work)])
    assert not work.exists()
    assert out["work_dir_removed"] is True


def test_work_dir_preserved_on_failure(tmp_path: Path):
    b, t = _book_trade_files(tmp_path)
    lines = b.read_text().splitlines()
    lines[2] = lines[2].replace(cfg.DEFAULT_SYMBOL, "ETHUSDT")
    b.write_text("\n".join(lines) + "\n")
    root = tmp_path / "ds"
    work = tmp_path / "work"
    with pytest.raises(ValueError):
        cli.main(["--dataset-root", str(root), "--dataset-id", "x", "--book-csv", str(b), "--trades-csv", str(t), "--work-dir", str(work)])
    assert work.exists()
    assert not (root / "manifest.json").exists()


def test_end_to_end_with_explicit_splits(tmp_path: Path, capsys):
    b, t = _book_trade_files(tmp_path, n_book=20, n_trade=20, step_us=1000)
    root = tmp_path / "ds"
    cli.main([
        "--dataset-root", str(root), "--dataset-id", "x", "--book-csv", str(b), "--trades-csv", str(t), "--label-horizons-us", "1000", "--label-entry-delay-us", "1", "--decision-stride-us", "1000",
        "--split-train", "1000000:1006000", "--split-val", "1006000:1015000",
    ])
    out = json.loads(capsys.readouterr().out.strip())
    man = mf.read_manifest_json(root / "manifest.json")
    roles = {s.role.value for s in man.splits}
    assert "train" in roles and "val" in roles
    assert out["splits_written"] is True
    assert "train" in out["split_roles"] and "val" in out["split_roles"]


def test_reject_partial_split_args(tmp_path: Path):
    b, t = _book_trade_files(tmp_path)
    root = tmp_path / "ds"
    work = tmp_path / "work"
    with pytest.raises(ValueError, match="both --split-train and --split-val"):
        cli.main(["--dataset-root", str(root), "--dataset-id", "x", "--book-csv", str(b), "--trades-csv", str(t), "--split-train", "1:2", "--work-dir", str(work)])
    assert not (root / "manifest.json").exists()
    assert not work.exists()


def test_pending_eof_decisions_are_not_force_labeled(tmp_path: Path, capsys):
    b, t = _book_trade_files(tmp_path, n_book=12, n_trade=12, step_us=1000)
    root = tmp_path / "ds"
    cli.main(["--dataset-root", str(root), "--dataset-id", "x", "--book-csv", str(b), "--trades-csv", str(t), "--label-horizons-us", "3000", "--label-entry-delay-us", "1", "--decision-stride-us", "1000"])
    out = json.loads(capsys.readouterr().out.strip())
    assert out["pending_decisions_at_eof"] > 0
    assert out["rows_written"] < out["decisions_emitted"]
    man = mf.read_manifest_json(root / "manifest.json")
    assert man.total_rows == out["rows_written"]


def test_stdout_summary_is_compact_and_has_no_large_state(tmp_path: Path, capsys):
    _, out, _ = _run_ok(tmp_path, capsys)
    for k in ["status", "dataset_root", "dataset_id", "segments", "rows", "decisions_emitted", "rows_written"]:
        assert k in out
    for bad in ["feature_values", "features", "label_values", "model", "preprocess", "diagnostics_state", "transform_state"]:
        assert bad not in out
    assert len(json.dumps(out)) < 5000


def test_existing_manifest_fails_before_work(tmp_path: Path):
    b, t = _book_trade_files(tmp_path)
    root = tmp_path / "ds"
    root.mkdir(parents=True)
    (root / "manifest.json").write_text("{}")
    work = tmp_path / "work"
    with pytest.raises(FileExistsError, match="manifest already exists"):
        cli.main(["--dataset-root", str(root), "--dataset-id", "x", "--book-csv", str(b), "--trades-csv", str(t), "--work-dir", str(work)])
    assert not work.exists()


def test_existing_segments_fail_before_work(tmp_path: Path):
    b, t = _book_trade_files(tmp_path)
    root = tmp_path / "ds"
    (root / "segments").mkdir(parents=True)
    (root / "segments" / "foo.parquet").write_text("x")
    work = tmp_path / "work"
    with pytest.raises(FileExistsError, match="existing parquet segments"):
        cli.main(["--dataset-root", str(root), "--dataset-id", "x", "--book-csv", str(b), "--trades-csv", str(t), "--work-dir", str(work)])
    assert not work.exists()


def test_subprocess_help_entrypoint():
    p = subprocess.run([sys.executable, "-m", "mmrt.cli.ingest", "--help"], capture_output=True, text=True)
    assert p.returncode == 0
    assert "--dataset-root" in p.stdout
    assert "--book-csv" in p.stdout
    assert "--trades-csv" in p.stdout
    assert "--split-train" in p.stdout
