from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from mmrt.cli.alpha_actionability import compute_alpha_actionability_summary
from mmrt.cli.profile_execution_observations import (
    _config_from_args,
    build_arg_parser,
    main,
    run_execution_observation_profile,
    ExecutionObservationProfileConfig,
)
from mmrt.execution.adverse_selection_dataset import AdverseSelectionDatasetWriter, AdverseSelectionDatasetWriterConfig
from mmrt.execution.adverse_selection_index import ADVERSE_SELECTION_INDEX_SCHEMA
from mmrt.execution.decision_grid import DECISION_GRID_SCHEMA, load_decision_grid
from mmrt.execution.linear_signal import (
    LINEAR_SIGNALS_FILENAME,
    LinearSignalArrays,
    LinearSignalArtifact,
    LinearSignalArtifactMetadata,
    load_linear_signal_artifact_npz,
)
from mmrt.execution.split_contract import load_execution_split_contract, ranges_for_split
from mmrt.features.schedule import DecisionScheduleConfig
from tests.grid_helpers import adverse_split_contract_fields
from tests.test_ppo_tiny_env import _tiny_tape_root


GRID_HASH = "1" * 64


def _required_args(tmp_path: Path) -> list[str]:
    tape_root = tmp_path / "tape"
    return [
        "--tape-root",
        str(tape_root),
        "--decision-grid",
        str(tape_root / "decision_grid"),
        "--split-source-dataset-root",
        str(tape_root / "split_source"),
        "--split",
        "train",
        "--linear-signals-npz",
        str(tape_root / LINEAR_SIGNALS_FILENAME),
        "--output-json",
        str(tmp_path / "profile.json"),
    ]


def _linear_artifact(n_rows: int, *, grid_hash: str = GRID_HASH) -> LinearSignalArtifact:
    score = np.linspace(-1.0, 1.0, n_rows, dtype=np.float64)
    p_move = np.full(n_rows, 0.5, dtype=np.float64)
    p_up_given_move = (score + 1.0) * 0.5
    p_up_move = p_move * p_up_given_move
    p_down_move = p_move * (1.0 - p_up_given_move)
    expected_up = np.maximum(score, 0.0)
    expected_down = np.maximum(-score, 0.0)
    arrays = LinearSignalArrays(
        p_no_move=1.0 - p_move,
        p_move=p_move,
        p_up_move=p_up_move,
        p_down_move=p_down_move,
        signed_move_prob=p_up_move - p_down_move,
        expected_up_bps=expected_up,
        expected_down_bps=expected_down,
        expected_return_bps=expected_up - expected_down,
        expected_abs_move_bps=expected_up + expected_down,
        predicted_vol_bps=np.abs(score),
        confidence=np.abs(p_up_move - p_down_move),
    )
    metadata = LinearSignalArtifactMetadata(
        tape_schema="schema",
        exchange="ex",
        symbol="SYM",
        num_events=n_rows + 1,
        num_l2_batches=n_rows,
        num_trades=0,
        start_local_ts_us=1,
        end_local_ts_us=n_rows + 1,
        decision_grid_schema=DECISION_GRID_SCHEMA,
        decision_grid_hash=grid_hash,
        decision_grid_n_rows=n_rows,
        decision_schedule=DecisionScheduleConfig().as_dict(),
        start_event_index=0,
        n_rows=n_rows,
    )
    return LinearSignalArtifact(
        arrays=arrays,
        metadata=metadata,
        decision_event_index=np.arange(n_rows, dtype=np.int64),
        decision_local_ts_us=np.arange(1, n_rows + 1, dtype=np.int64),
        decision_event_seq=np.arange(n_rows, dtype=np.int64),
    )


def _tail_label_names(*, include_incomplete_inside: bool = False) -> tuple[str, ...]:
    labels = []
    for side in ("bid", "ask"):
        prefix = f"{side}_touch"
        labels.extend(
            [
                f"{prefix}_filled",
                f"{prefix}_fill_latency_us",
                f"{prefix}_adverse_bps",
                f"{prefix}_toxic_fill",
                f"{prefix}_toxic_cost_bps",
            ]
        )
    if include_incomplete_inside:
        labels.append("bid_inside_1_filled")
    return tuple(labels)


def _split_contract(tmp_path: Path, linear: LinearSignalArtifact) -> dict[str, object]:
    n_rows = linear.n_rows
    ranges = {
        "train": [
            {
                "role": "train",
                "segment_key": "seg_000",
                "start_decision_row": 0,
                "end_decision_row": n_rows,
                "row_count": n_rows,
                "start_local_ts_us": 1,
                "end_local_ts_us": n_rows + 1,
                "embargo_before_us": 0,
                "embargo_after_us": 0,
            }
        ],
        "val": [],
        "test": [],
    }
    fields = adverse_split_contract_fields(
        n_rows=n_rows,
        grid_hash=linear.metadata.decision_grid_hash,
        root=str(tmp_path / "split_source"),
        ranges=ranges,
    )
    contract = dict(fields["split_contract"])
    return contract


def _dataset_metadata(tmp_path: Path, linear: LinearSignalArtifact, contract: dict[str, object]) -> dict[str, object]:
    return {
        "exchange": linear.metadata.exchange,
        "symbol": linear.metadata.symbol,
        "tape_schema": linear.metadata.tape_schema,
        "tape_num_events": linear.metadata.num_events,
        "tape_num_l2_batches": linear.metadata.num_l2_batches,
        "tape_num_trades": linear.metadata.num_trades,
        "tape_start_local_ts_us": linear.metadata.start_local_ts_us,
        "tape_end_local_ts_us": linear.metadata.end_local_ts_us,
        "decision_grid_schema": linear.metadata.decision_grid_schema,
        "decision_grid_hash": linear.metadata.decision_grid_hash,
        "decision_grid_n_rows": linear.metadata.decision_grid_n_rows,
        "decision_schedule": dict(linear.metadata.decision_schedule),
        "split_source_dataset_root": str(contract["split_source_dataset_root"]),
        "split_source_dataset_id": str(contract["split_source_dataset_id"]),
        "split_source_manifest_hash": str(contract["split_source_manifest_hash"]),
        "split_contract": contract,
        "config_json": "{}",
        "index_schema": ADVERSE_SELECTION_INDEX_SCHEMA,
        "index_manifest_sha256": "0" * 64,
        "index_root": str(tmp_path / "index"),
    }


def _labels_for_rows(linear: LinearSignalArtifact, rows: np.ndarray, label_names: tuple[str, ...]) -> np.ndarray:
    score = np.asarray(linear.arrays.signed_move_prob[rows] / linear.arrays.p_move[rows], dtype=np.float64)
    labels = np.zeros((rows.size, len(label_names)), dtype=np.float32)
    index = {name: i for i, name in enumerate(label_names)}
    for side, fill_mask in (
        ("bid", score >= 0.7),
        ("ask", score <= -0.7),
    ):
        prefix = f"{side}_touch"
        if f"{prefix}_filled" not in index:
            continue
        labels[:, index[f"{prefix}_filled"]] = fill_mask.astype(np.float32)
        if f"{prefix}_fill_latency_us" in index:
            labels[:, index[f"{prefix}_fill_latency_us"]] = np.where(fill_mask, 100.0, 0.0)
        if f"{prefix}_adverse_bps" in index:
            labels[:, index[f"{prefix}_adverse_bps"]] = np.where(fill_mask, 0.25, 0.0)
        if f"{prefix}_toxic_fill" in index:
            labels[:, index[f"{prefix}_toxic_fill"]] = np.where(fill_mask & (score * (1 if side == "bid" else -1) > 0.9), 1.0, 0.0)
        if f"{prefix}_toxic_cost_bps" in index:
            labels[:, index[f"{prefix}_toxic_cost_bps"]] = np.where(fill_mask, 0.25, 0.0)
    return labels


def _write_adverse_dataset(
    tmp_path: Path,
    *,
    linear: LinearSignalArtifact,
    contract: dict[str, object],
    label_names: tuple[str, ...] | None = None,
    rows: np.ndarray | None = None,
):
    label_names = label_names or _tail_label_names()
    rows = np.asarray(np.arange(linear.n_rows, dtype=np.int64) if rows is None else rows, dtype=np.int64)
    contract = dict(contract)
    if int(contract.get("adverse_dataset_rows_total", 0)) == 0 and rows.size:
        contract["adverse_dataset_rows_total"] = int(rows.size)
    adverse_counts = contract.get("adverse_row_counts")
    if not isinstance(adverse_counts, dict) or sum(int(v) for v in adverse_counts.values()) == 0:
        contract["adverse_row_counts"] = {"train": int(rows.size), "val": 0, "test": 0, "out_of_split": 0}
    writer = AdverseSelectionDatasetWriter(
        AdverseSelectionDatasetWriterConfig(
            output_root=str(tmp_path / "adverse_dataset"),
            feature_names=("x",),
            label_names=label_names,
            manifest_metadata=_dataset_metadata(tmp_path, linear, contract),
            overwrite=True,
        )
    )
    labels = _labels_for_rows(linear, rows, label_names)
    writer.append_many(
        decision_local_ts_us=linear.decision_local_ts_us[rows],
        decision_event_index=linear.decision_event_index[rows],
        decision_event_seq=linear.decision_event_seq[rows],
        features=np.zeros((rows.size, 1), dtype=np.float32),
        labels=labels,
        label_masks=np.ones_like(labels, dtype=np.bool_),
    )
    return writer.finalize()


def test_alpha_actionability_parser_defaults_and_validation(tmp_path):
    parser = build_arg_parser()
    config = _config_from_args(parser.parse_args(_required_args(tmp_path)))
    assert config.alpha_actionability is False
    assert config.alpha_actionability_percentiles == (10, 20)
    assert config.alpha_actionability_max_rows == 1_000_000

    with pytest.raises(ValueError, match="requires adverse_dataset_root"):
        _config_from_args(parser.parse_args([*_required_args(tmp_path), "--alpha-actionability"]))

    config = _config_from_args(
        parser.parse_args(
            [
                *_required_args(tmp_path),
                "--alpha-actionability",
                "--adverse-dataset-root",
                str(tmp_path / "adverse"),
                "--alpha-actionability-percentiles",
                "20,10,10",
            ]
        )
    )
    assert config.alpha_actionability_percentiles == (10, 20)

    for value in ("0", "50", "-1", "abc"):
        with pytest.raises(ValueError):
            _config_from_args(
                parser.parse_args([*_required_args(tmp_path), "--alpha-actionability-percentiles", value])
            )


def test_synthetic_alpha_actionability_reports_directional_tails(tmp_path):
    linear = _linear_artifact(200)
    contract = _split_contract(tmp_path, linear)
    dataset = _write_adverse_dataset(tmp_path, linear=linear, contract=contract)

    summary = compute_alpha_actionability_summary(
        adverse_dataset_root=dataset.root,
        split="train",
        split_contract=contract,
        decision_grid_hash=linear.metadata.decision_grid_hash,
        decision_grid_n_rows=linear.metadata.decision_grid_n_rows,
        linear_signals=linear,
        max_rows=200,
        percentiles=(10, 20),
    )

    direction = summary["axes"]["direction_score"]
    assert direction["buckets"]["top_10"]["quotes"]["bid_touch"]["fill_rate_lift_vs_unconditional"] > 0
    assert direction["buckets"]["bottom_10"]["quotes"]["ask_touch"]["fill_rate_lift_vs_unconditional"] > 0
    assert set(direction["buckets"]) == {"bottom_10", "bottom_20", "top_20", "top_10"}
    assert "expected_return_bps" in summary["axes"]
    assert summary["compact"]["direction_score_top10_bid_touch_fill_rate"] == pytest.approx(1.0)


def test_alpha_actionability_uses_dataset_labels_not_prediction_artifact(tmp_path):
    linear = _linear_artifact(80)
    contract = _split_contract(tmp_path, linear)
    dataset = _write_adverse_dataset(tmp_path, linear=linear, contract=contract)
    (Path(dataset.root) / "adverse_selection_signals.npz").write_bytes(b"not a valid prediction artifact")

    summary = compute_alpha_actionability_summary(
        adverse_dataset_root=dataset.root,
        split="train",
        split_contract=contract,
        decision_grid_hash=linear.metadata.decision_grid_hash,
        decision_grid_n_rows=linear.metadata.decision_grid_n_rows,
        linear_signals=linear,
        max_rows=80,
    )

    assert summary["source"] == "empirical_adverse_dataset_labels"


def test_alpha_actionability_lineage_and_label_validation(tmp_path):
    linear = _linear_artifact(40)
    contract = _split_contract(tmp_path, linear)
    dataset = _write_adverse_dataset(tmp_path, linear=linear, contract=contract)

    with pytest.raises(ValueError, match="decision_grid_hash"):
        compute_alpha_actionability_summary(
            adverse_dataset_root=dataset.root,
            split="train",
            split_contract=contract,
            decision_grid_hash="2" * 64,
            decision_grid_n_rows=linear.metadata.decision_grid_n_rows,
            linear_signals=linear,
        )

    missing = _write_adverse_dataset(
        tmp_path / "missing",
        linear=linear,
        contract=contract,
        label_names=("bid_touch_filled",),
    )
    with pytest.raises(ValueError, match="missing required touch labels"):
        compute_alpha_actionability_summary(
            adverse_dataset_root=missing.root,
            split="train",
            split_contract=contract,
            decision_grid_hash=linear.metadata.decision_grid_hash,
            decision_grid_n_rows=linear.metadata.decision_grid_n_rows,
            linear_signals=linear,
        )


def test_alpha_actionability_omits_incomplete_optional_candidates(tmp_path):
    linear = _linear_artifact(60)
    contract = _split_contract(tmp_path, linear)
    dataset = _write_adverse_dataset(
        tmp_path,
        linear=linear,
        contract=contract,
        label_names=_tail_label_names(include_incomplete_inside=True),
    )

    summary = compute_alpha_actionability_summary(
        adverse_dataset_root=dataset.root,
        split="train",
        split_contract=contract,
        decision_grid_hash=linear.metadata.decision_grid_hash,
        decision_grid_n_rows=linear.metadata.decision_grid_n_rows,
        linear_signals=linear,
        max_rows=60,
    )

    quotes = summary["axes"]["direction_score"]["buckets"]["top_10"]["quotes"]
    assert set(quotes) == {"bid_touch", "ask_touch"}
    assert "bid_inside_1" not in summary["lineage"]["label_names_used"]


def _write_profile_adverse_dataset(tmp_path: Path, tape_root: Path):
    linear = load_linear_signal_artifact_npz(tape_root / LINEAR_SIGNALS_FILENAME)
    grid = load_decision_grid(tape_root / "decision_grid")
    contract = load_execution_split_contract(tape_root / "split_source", grid).as_dict()
    adverse_contract = dict(contract)
    row_counts = {
        role: int(sum(item.row_count for item in ranges_for_split(contract, role)))
        for role in ("train", "val", "test")
    }
    adverse_contract["adverse_dataset_rows_total"] = linear.n_rows
    adverse_contract["adverse_row_counts"] = {**row_counts, "out_of_split": 0}
    return _write_adverse_dataset(tmp_path, linear=linear, contract=adverse_contract)


def test_profile_alpha_actionability_integration_and_concise_stdout(tmp_path, capsys):
    tape_root = _tiny_tape_root(tmp_path)
    dataset = _write_profile_adverse_dataset(tmp_path, tape_root)
    output_json = tmp_path / "profile_alpha.json"
    rc = main(
        [
            "--tape-root",
            str(tape_root),
            "--decision-grid",
            str(tape_root / "decision_grid"),
            "--split-source-dataset-root",
            str(tape_root / "split_source"),
            "--split",
            "train",
            "--linear-signals-npz",
            str(tape_root / LINEAR_SIGNALS_FILENAME),
            "--output-json",
            str(output_json),
            "--sample-rows",
            "3",
            "--num-envs",
            "1",
            "--alpha-actionability",
            "--adverse-dataset-root",
            str(dataset.root),
            "--alpha-actionability-max-rows",
            "20",
            "--overwrite",
        ]
    )

    stdout = capsys.readouterr().out
    payload = json.loads(output_json.read_text())
    assert rc == 0
    assert payload["alpha_actionability_summary"]["enabled"] is True
    assert set(payload["alpha_actionability_summary"]["axes"]) == {
        "direction_score",
        "signed_move_prob",
        "expected_return_bps",
    }
    assert "top_10" in payload["alpha_actionability_summary"]["axes"]["direction_score"]["buckets"]
    assert "alpha_actionability: enabled" in stdout
    assert len(stdout.splitlines()) < 50
    assert "buckets" not in stdout


def test_profile_alpha_actionability_disabled_payload_is_light(tmp_path):
    tape_root = _tiny_tape_root(tmp_path)
    summary = run_execution_observation_profile(
        ExecutionObservationProfileConfig(
            tape_root=str(tape_root),
            decision_grid_path=str(tape_root / "decision_grid"),
            split_source_dataset_root=str(tape_root / "split_source"),
            split="train",
            linear_signals_npz=str(tape_root / LINEAR_SIGNALS_FILENAME),
            output_json=str(tmp_path / "profile.json"),
            sample_rows=3,
            num_envs=1,
            stdout_mode="none",
            overwrite=True,
        )
    )

    assert summary["alpha_actionability_summary"] == {"enabled": False}
