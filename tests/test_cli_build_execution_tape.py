import csv
import json
from pathlib import Path

import pytest

from mmrt.cli.build_execution_tape import (
    ExecutionTapeBuildConfig,
    build_arg_parser,
    build_execution_tape_from_config,
    load_reconstructed_l2_events,
    main,
)
from mmrt.execution.contracts import SymbolSpec
from mmrt.execution.event_merge import ExecutionMergeTiePolicy
from mmrt.execution.execution_tape import EVENT_TYPE_CODE_L2_BATCH, EVENT_TYPE_CODE_TRADE, load_execution_tape


def _write_l2_csv(path: Path, rows: list[dict[str, object]]) -> Path:
    columns = ["timestamp", "local_timestamp", "is_snapshot", "side", "price", "amount"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return path


def _write_trade_csv(path: Path, rows: list[dict[str, object]], columns: list[str] | None = None) -> Path:
    if columns is None:
        columns = ["timestamp", "local_timestamp", "side", "price", "amount", "trade_id"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return path


def _good_l2_rows() -> list[dict[str, object]]:
    return [
        {"timestamp": 100, "local_timestamp": 100, "is_snapshot": "true", "side": "bid", "price": 100.0, "amount": 1.0},
        {"timestamp": 100, "local_timestamp": 100, "is_snapshot": "true", "side": "ask", "price": 100.2, "amount": 1.0},
        {"timestamp": 200, "local_timestamp": 200, "is_snapshot": "false", "side": "bid", "price": 100.1, "amount": 1.0},
        {"timestamp": 300, "local_timestamp": 300, "is_snapshot": "false", "side": "ask", "price": 100.3, "amount": 1.0},
    ]


def _good_trade_rows() -> list[dict[str, object]]:
    return [
        {"timestamp": 150, "local_timestamp": 150, "side": "buy", "price": 100.2, "amount": 0.01, "trade_id": "t1"},
        {"timestamp": 250, "local_timestamp": 250, "side": "sell", "price": 100.1, "amount": 0.02, "trade_id": "t2"},
    ]


def test_build_execution_tape_from_csv_inputs(tmp_path):
    l2_path = _write_l2_csv(tmp_path / "l2.csv", _good_l2_rows())
    trade_path = _write_trade_csv(tmp_path / "trades.csv", _good_trade_rows())
    output_root = tmp_path / "tape"

    summary = build_execution_tape_from_config(
        ExecutionTapeBuildConfig(
            l2_inputs=(str(l2_path),),
            trade_inputs=(str(trade_path),),
            output_root=str(output_root),
        )
    )

    assert summary["status"] == "ok"
    assert (output_root / "manifest.json").exists()
    assert (output_root / "arrays" / "events.npy").exists()
    assert (output_root / "arrays" / "l2_events.npy").exists()
    assert (output_root / "arrays" / "trades.npy").exists()
    assert (output_root / "arrays" / "book_bid_ticks.npy").exists()
    assert (output_root / "arrays" / "book_bid_sizes.npy").exists()
    assert (output_root / "arrays" / "book_ask_ticks.npy").exists()
    assert (output_root / "arrays" / "book_ask_sizes.npy").exists()
    assert (output_root / "build_summary.json").exists()

    loaded = load_execution_tape(output_root)
    assert loaded.manifest.num_l2_batches == summary["tape"]["num_l2_batches"]
    assert loaded.manifest.num_trades == summary["tape"]["num_trades"]
    assert loaded.manifest.num_events == summary["tape"]["num_events"]
    assert loaded.manifest.num_decisions == 0
    assert summary["tape"]["book_depth"] == 25
    assert loaded.arrays.book_bid_ticks.shape[1] == 25
    assert loaded.arrays.book_ask_ticks.shape[1] == 25


def test_event_ordering_and_tie_policy(tmp_path):
    l2_path = _write_l2_csv(tmp_path / "l2.csv", _good_l2_rows()[:2])
    trade_path = _write_trade_csv(
        tmp_path / "trades.csv",
        [{"timestamp": 100, "local_timestamp": 100, "side": "buy", "price": 100.2, "amount": 0.01, "trade_id": "t1"}],
    )

    default_summary = build_execution_tape_from_config(
        ExecutionTapeBuildConfig(
            l2_inputs=(str(l2_path),),
            trade_inputs=(str(trade_path),),
            output_root=str(tmp_path / "default"),
        )
    )
    default_tape = load_execution_tape(tmp_path / "default")
    assert default_summary["merge"]["same_local_ts_tie_count"] == 1
    assert list(default_tape.arrays.events["event_type_code"]) == [EVENT_TYPE_CODE_L2_BATCH, EVENT_TYPE_CODE_TRADE]

    trade_first_summary = build_execution_tape_from_config(
        ExecutionTapeBuildConfig(
            l2_inputs=(str(l2_path),),
            trade_inputs=(str(trade_path),),
            output_root=str(tmp_path / "trade_first"),
            tie_policy="trade_before_l2",
        )
    )
    trade_first_tape = load_execution_tape(tmp_path / "trade_first")
    assert trade_first_summary["merge"]["tie_policy"] == ExecutionMergeTiePolicy.TRADE_BEFORE_L2.value
    assert list(trade_first_tape.arrays.events["event_type_code"]) == [EVENT_TYPE_CODE_TRADE, EVENT_TYPE_CODE_L2_BATCH]


def test_overwrite_protection(tmp_path):
    l2_path = _write_l2_csv(tmp_path / "l2.csv", _good_l2_rows())
    trade_path = _write_trade_csv(tmp_path / "trades.csv", _good_trade_rows())
    output_root = tmp_path / "tape"
    config = ExecutionTapeBuildConfig(l2_inputs=(str(l2_path),), trade_inputs=(str(trade_path),), output_root=str(output_root))

    build_execution_tape_from_config(config)
    with pytest.raises(FileExistsError):
        build_execution_tape_from_config(config)

    summary = build_execution_tape_from_config(
        ExecutionTapeBuildConfig(
            l2_inputs=(str(l2_path),),
            trade_inputs=(str(trade_path),),
            output_root=str(output_root),
            overwrite=True,
        )
    )
    assert summary["status"] == "ok"


def test_existing_build_summary_preflight_blocks_partial_tape_write(tmp_path):
    l2_path = _write_l2_csv(tmp_path / "l2.csv", _good_l2_rows())
    trade_path = _write_trade_csv(tmp_path / "trades.csv", _good_trade_rows())
    output_root = tmp_path / "tape"
    output_root.mkdir()
    (output_root / "build_summary.json").write_text('{"old": true}\n', encoding="utf-8")

    with pytest.raises(FileExistsError, match="build_summary"):
        build_execution_tape_from_config(
            ExecutionTapeBuildConfig(
                l2_inputs=(str(l2_path),),
                trade_inputs=(str(trade_path),),
                output_root=str(output_root),
            )
        )

    assert not (output_root / "manifest.json").exists()
    assert not (output_root / "arrays").exists()


def test_fallback_trade_source_rows_are_zero_based(tmp_path):
    l2_path = _write_l2_csv(tmp_path / "l2.csv", _good_l2_rows())
    trade_path = _write_trade_csv(tmp_path / "trades.csv", _good_trade_rows())
    output_root = tmp_path / "tape"

    build_execution_tape_from_config(
        ExecutionTapeBuildConfig(
            l2_inputs=(str(l2_path),),
            trade_inputs=(str(trade_path),),
            output_root=str(output_root),
        )
    )

    loaded = load_execution_tape(output_root)
    assert loaded.arrays.trades["source_row"].tolist() == [0, 1]


def test_explicit_trade_source_rows_are_preserved(tmp_path):
    l2_path = _write_l2_csv(tmp_path / "l2.csv", _good_l2_rows())
    trade_path = _write_trade_csv(
        tmp_path / "trades.csv",
        [
            {
                "timestamp": 150,
                "local_timestamp": 150,
                "side": "buy",
                "price": 100.2,
                "amount": 0.01,
                "trade_id": "t1",
                "source_row": 10,
            },
            {
                "timestamp": 250,
                "local_timestamp": 250,
                "side": "sell",
                "price": 100.1,
                "amount": 0.02,
                "trade_id": "t2",
                "source_row": 11,
            },
        ],
        columns=["timestamp", "local_timestamp", "side", "price", "amount", "trade_id", "source_row"],
    )
    output_root = tmp_path / "tape"

    build_execution_tape_from_config(
        ExecutionTapeBuildConfig(
            l2_inputs=(str(l2_path),),
            trade_inputs=(str(trade_path),),
            output_root=str(output_root),
        )
    )

    loaded = load_execution_tape(output_root)
    assert loaded.arrays.trades["source_row"].tolist() == [10, 11]


def test_l2_stats_are_json_safe_and_do_not_include_reconstructor(tmp_path):
    l2_path = _write_l2_csv(tmp_path / "l2.csv", _good_l2_rows())

    _, stats = load_reconstructed_l2_events(
        (l2_path,),
        symbol_spec=SymbolSpec(
            exchange="binance-futures",
            symbol="BTCUSDT",
            tick_size=0.1,
            step_size=0.001,
            min_qty=0.001,
            max_qty=100.0,
            min_notional=5.0,
        ),
        batch_size=65_536,
    )

    assert "reconstructor" not in stats
    json.dumps(stats, sort_keys=True, allow_nan=True)


def test_max_row_limits(tmp_path):
    l2_path = _write_l2_csv(tmp_path / "l2.csv", _good_l2_rows())
    trade_path = _write_trade_csv(tmp_path / "trades.csv", _good_trade_rows())

    summary = build_execution_tape_from_config(
        ExecutionTapeBuildConfig(
            l2_inputs=(str(l2_path),),
            trade_inputs=(str(trade_path),),
            output_root=str(tmp_path / "tape"),
            max_l2_rows=2,
            max_trade_rows=1,
        )
    )

    assert summary["counts"]["l2_rows_seen"] == 2
    assert summary["counts"]["trade_rows_seen"] == 1
    assert "l2_scan_limit_hit" in summary["warnings"]
    assert "trade_scan_limit_hit" in summary["warnings"]


def test_rejects_no_reconstructed_l2_events(tmp_path):
    l2_path = _write_l2_csv(
        tmp_path / "l2.csv",
        [{"timestamp": 100, "local_timestamp": 100, "is_snapshot": "false", "side": "bid", "price": 100.0, "amount": 1.0}],
    )
    trade_path = _write_trade_csv(tmp_path / "trades.csv", _good_trade_rows())

    with pytest.raises(ValueError, match="without reconstructed L2 events"):
        build_execution_tape_from_config(
            ExecutionTapeBuildConfig(l2_inputs=(str(l2_path),), trade_inputs=(str(trade_path),), output_root=str(tmp_path / "tape"))
        )


def test_rejects_no_trades(tmp_path):
    l2_path = _write_l2_csv(tmp_path / "l2.csv", _good_l2_rows())
    trade_path = _write_trade_csv(tmp_path / "trades.csv", [])

    with pytest.raises(ValueError, match="without trades"):
        build_execution_tape_from_config(
            ExecutionTapeBuildConfig(l2_inputs=(str(l2_path),), trade_inputs=(str(trade_path),), output_root=str(tmp_path / "tape"))
        )


def test_rejects_unsorted_trades(tmp_path):
    l2_path = _write_l2_csv(tmp_path / "l2.csv", _good_l2_rows())
    trade_path = _write_trade_csv(
        tmp_path / "trades.csv",
        [
            {"timestamp": 250, "local_timestamp": 250, "side": "buy", "price": 100.2, "amount": 0.01, "trade_id": "t1"},
            {"timestamp": 150, "local_timestamp": 150, "side": "sell", "price": 100.1, "amount": 0.02, "trade_id": "t2"},
        ],
    )

    with pytest.raises(ValueError, match="trades must be sorted"):
        build_execution_tape_from_config(
            ExecutionTapeBuildConfig(l2_inputs=(str(l2_path),), trade_inputs=(str(trade_path),), output_root=str(tmp_path / "tape"))
        )


def test_rejects_missing_input_files(tmp_path):
    trade_path = _write_trade_csv(tmp_path / "trades.csv", _good_trade_rows())

    with pytest.raises(FileNotFoundError):
        build_execution_tape_from_config(
            ExecutionTapeBuildConfig(l2_inputs=(str(tmp_path / "missing.csv"),), trade_inputs=(str(trade_path),), output_root=str(tmp_path / "tape"))
        )


def test_parser_smoke():
    args = build_arg_parser().parse_args(
        [
            "--l2-input",
            "l2.csv",
            "--trade-input",
            "trades.csv",
            "--output-root",
            "out",
            "--tie-policy",
            "trade_before_l2",
            "--overwrite",
            "--max-l2-rows",
            "10",
            "--max-trade-rows",
            "11",
            "--book-depth",
            "3",
        ]
    )
    assert args.l2_input == ["l2.csv"]
    assert args.trade_input == ["trades.csv"]
    assert args.output_root == "out"
    assert args.tie_policy == "trade_before_l2"
    assert args.overwrite is True
    assert args.max_l2_rows == 10
    assert args.max_trade_rows == 11
    assert args.book_depth == 3


def test_cli_main_writes_summary_and_prints_same_summary(tmp_path, capsys):
    l2_path = _write_l2_csv(tmp_path / "l2.csv", _good_l2_rows())
    trade_path = _write_trade_csv(tmp_path / "trades.csv", _good_trade_rows())
    output_root = tmp_path / "tape"

    rc = main(["--l2-input", str(l2_path), "--trade-input", str(trade_path), "--output-root", str(output_root)])
    printed = json.loads(capsys.readouterr().out)
    saved = json.loads((output_root / "build_summary.json").read_text(encoding="utf-8"))

    assert rc == 0
    assert printed == saved
    assert printed["status"] == "ok"


def test_parquet_input(tmp_path):
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    l2_path = tmp_path / "l2.parquet"
    trade_path = tmp_path / "trades.parquet"
    pq.write_table(pa.Table.from_pylist(_good_l2_rows()), l2_path)
    pq.write_table(pa.Table.from_pylist(_good_trade_rows()), trade_path)

    summary = build_execution_tape_from_config(
        ExecutionTapeBuildConfig(l2_inputs=(str(l2_path),), trade_inputs=(str(trade_path),), output_root=str(tmp_path / "tape"))
    )

    assert summary["status"] == "ok"
    assert (tmp_path / "tape" / "manifest.json").exists()
    assert (tmp_path / "tape" / "arrays" / "events.npy").exists()


def test_no_forbidden_imports():
    source = Path("mmrt/cli/build_execution_tape.py").read_text(encoding="utf-8")
    assert "import torch" not in source
    assert "import pandas" not in source
    assert "import sklearn" not in source


def test_book_depth_argument_controls_saved_snapshot_depth(tmp_path):
    l2_path = _write_l2_csv(tmp_path / "l2.csv", _good_l2_rows())
    trade_path = _write_trade_csv(tmp_path / "trades.csv", _good_trade_rows())
    output_root = tmp_path / "tape"

    summary = build_execution_tape_from_config(
        ExecutionTapeBuildConfig(
            l2_inputs=(str(l2_path),),
            trade_inputs=(str(trade_path),),
            output_root=str(output_root),
            book_depth=3,
        )
    )
    loaded = load_execution_tape(output_root)

    assert summary["tape"]["book_depth"] == 3
    assert summary["reconstruction"]["book_depth"] == 3
    assert loaded.arrays.book_bid_ticks.shape[1] == 3
    assert loaded.arrays.book_ask_ticks.shape[1] == 3
    assert loaded.manifest.notes["book_depth"] == "3"


def test_config_rejects_nonpositive_book_depth(tmp_path):
    with pytest.raises(ValueError, match="book_depth"):
        ExecutionTapeBuildConfig(l2_inputs=("l2.csv",), trade_inputs=("trades.csv",), output_root=str(tmp_path / "tape"), book_depth=0)
