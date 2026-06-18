import json

import pytest

from mmrt.execution.split_contract import load_execution_split_contract, validate_split_contract_payload
from tests.test_ppo_tiny_env import _tiny_tape
from tests.grid_helpers import decision_grid_for_tape, write_split_source_manifest


def _split_source(tmp_path):
    tape = _tiny_tape()
    grid = decision_grid_for_tape(tape)
    root = write_split_source_manifest(tmp_path / "split_source", grid)
    return root, grid


def test_execution_split_contract_loader_exposes_lineage_and_counts(tmp_path):
    root, grid = _split_source(tmp_path)

    contract = load_execution_split_contract(root, grid).as_dict()

    assert contract["schema"] == "mmrt_execution_split_contract_v1"
    assert contract["split_source_dataset_id"] == "split-source"
    assert contract["decision_grid_hash"] == grid.decision_grid_hash
    assert contract["decision_grid_n_rows"] == grid.n_rows
    assert set(contract["ranges_by_split"]) == {"train", "val", "test"}
    assert contract["row_counts_by_split"]["train"] > 0
    assert contract["row_counts_by_split"]["val"] > 0
    assert contract["row_counts_by_split"]["test"] > 0


def test_execution_split_contract_missing_named_split_fails(tmp_path):
    root, grid = _split_source(tmp_path)
    manifest_path = root / "manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["splits"] = [entry for entry in payload["splits"] if entry["role"] != "test"]
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="train/val/test"):
        load_execution_split_contract(root, grid)


def test_execution_split_contract_decision_grid_hash_mismatch_fails(tmp_path):
    root, grid = _split_source(tmp_path)
    manifest_path = root / "manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["notes"]["decision_grid"]["decision_grid_hash"] = "0" * 64
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="decision_grid_hash"):
        load_execution_split_contract(root, grid)


def test_execution_split_contract_rejects_cross_role_overlap(tmp_path):
    root, grid = _split_source(tmp_path)
    payload = load_execution_split_contract(root, grid).as_dict()
    payload["ranges_by_split"]["val"][0]["start_decision_row"] = 3
    payload["ranges_by_split"]["val"][0]["end_decision_row"] = 5

    with pytest.raises(ValueError, match="overlap across roles"):
        validate_split_contract_payload(payload)


def test_execution_split_contract_rejects_unsorted_role_ranges(tmp_path):
    root, grid = _split_source(tmp_path)
    payload = load_execution_split_contract(root, grid).as_dict()
    payload["ranges_by_split"]["train"] = [
        {
            "role": "train",
            "segment_key": "seg_001",
            "start_decision_row": 2,
            "end_decision_row": 3,
            "start_local_ts_us": 200,
            "end_local_ts_us": 201,
        },
        {
            "role": "train",
            "segment_key": "seg_000",
            "start_decision_row": 0,
            "end_decision_row": 1,
            "start_local_ts_us": 100,
            "end_local_ts_us": 101,
        },
    ]

    with pytest.raises(ValueError, match="sorted"):
        validate_split_contract_payload(payload)


def test_execution_split_contract_rejects_overlap_within_role(tmp_path):
    root, grid = _split_source(tmp_path)
    payload = load_execution_split_contract(root, grid).as_dict()
    payload["ranges_by_split"]["train"] = [
        {
            "role": "train",
            "segment_key": "seg_000",
            "start_decision_row": 0,
            "end_decision_row": 2,
            "start_local_ts_us": 100,
            "end_local_ts_us": 201,
        },
        {
            "role": "train",
            "segment_key": "seg_001",
            "start_decision_row": 1,
            "end_decision_row": 3,
            "start_local_ts_us": 200,
            "end_local_ts_us": 301,
        },
    ]

    with pytest.raises(ValueError, match="must not overlap"):
        validate_split_contract_payload(payload)


def test_execution_split_contract_rejects_row_count_mismatch(tmp_path):
    root, grid = _split_source(tmp_path)
    payload = load_execution_split_contract(root, grid).as_dict()
    payload["row_counts_by_split"]["train"] += 1

    with pytest.raises(ValueError, match="row_counts_by_split"):
        validate_split_contract_payload(payload)
