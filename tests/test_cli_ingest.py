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


def _book_trade_files(tmp_path: Path):
    b_schema = tardis_csv_schema(TardisDataType.BOOK_SNAPSHOT_25)
    t_schema = tardis_csv_schema(TardisDataType.TRADES)
    bh = list(b_schema.column_names)
    th = list(t_schema.column_names)
    book = tmp_path / "book.csv"
    trade = tmp_path / "trades.csv"
    brows = []
    for i in range(8):
        row = []
        for c in bh:
            if c == "exchange": row.append(cfg.DEFAULT_EXCHANGE)
            elif c == "symbol": row.append(cfg.DEFAULT_SYMBOL)
            elif c == "timestamp": row.append(1_000_000 + i * 1_000)
            elif c == "local_timestamp": row.append(1_000_000 + i * 1_000)
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
    for i in range(4):
        row = []
        for c in th:
            if c == "exchange": row.append(cfg.DEFAULT_EXCHANGE)
            elif c == "symbol": row.append(cfg.DEFAULT_SYMBOL)
            elif c == "timestamp": row.append(1_000_000 + i * 1_000)
            elif c == "local_timestamp": row.append(1_000_000 + i * 1_000)
            elif c == "price": row.append(100.0 + i * 0.01)
            elif c == "amount": row.append(0.5)
            elif c == "side": row.append("buy" if i % 2 == 0 else "sell")
            elif "id" in c: row.append(str(i + 1))
            else: row.append("")
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


def test_parse_us_range():
    assert cli._parse_us_range("100:200", "x") == (100, 200)
    for bad in ["200:100", "abc:200", "100", "100:200:300"]:
        with pytest.raises(ValueError):
            cli._parse_us_range(bad, "x")


def test_no_stale_imports_source_residue():
    src = inspect.getsource(cli)
    for token in ["BYBIT", "CMSSL", "offline_ingest", "linear_offline", "MiniRocket", "MultiRocket", "Hydra", "Aeon", "sklearn", "torch", "PCA", "StandardScaler", "stage1", "stage2", "stage3", "stage4", "stage5", "pandas"]:
        assert token not in src
    for token in ["mmrt.linear", "read_split_table", "read_table(", "to_pandas"]:
        assert token not in src


def test_end_to_end_smoke(tmp_path: Path, capsys):
    b, t = _book_trade_files(tmp_path)
    root = tmp_path / "ds"
    rc = cli.main(["--dataset-root", str(root), "--dataset-id", "tiny", "--book-csv", str(b), "--trades-csv", str(t), "--label-horizons-us", "1000", "--label-entry-delay-us", "1", "--decision-stride-us", "500000", "--event-batch-size", "2", "--chunk-rows", "2", "--row-group-rows", "2"])
    assert rc == 0
    out = json.loads(capsys.readouterr().out.strip())
    assert out["status"] == "ok"
    assert (root / "manifest.json").exists()
    assert list((root / "segments").glob("*.parquet"))
    r = rd.open_dataset(str(root), validate_on_open=True)
    assert r.total_rows > 0
    man = mf.read_manifest_json(root / "manifest.json")
    assert man.pipeline_config.market.exchange == cfg.DEFAULT_EXCHANGE
    assert man.pipeline_config.market.symbol == cfg.DEFAULT_SYMBOL
    assert man.storage_format.value == "flat_decision_rows_us_v1"
    assert man.time_unit.value == "microsecond"
    assert man.transform_config
    assert int(man.transform_diagnostics.get("rows_seen", 0)) > 0
    assert man.splits == ()


def test_subprocess_help_entrypoint():
    p = subprocess.run([sys.executable, "-m", "mmrt.cli.ingest", "--help"], capture_output=True, text=True)
    assert p.returncode == 0
    assert "--dataset-root" in p.stdout
    assert "--book-csv" in p.stdout
    assert "--trades-csv" in p.stdout
    assert "--split-train" in p.stdout
