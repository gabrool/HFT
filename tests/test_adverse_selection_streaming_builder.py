import inspect
import numpy as np

import mmrt.execution.adverse_selection as adverse_mod
from mmrt.execution.adverse_selection import build_adverse_selection_dataset_to_disk
import mmrt.execution.adverse_selection_feature_store as feature_store_mod
from tests.adverse_helpers import build_tiny_adverse_selection_dataset
from tests.test_adverse_selection import _tape, _l2, _trade, _base_config, AggressorSide
from tests.grid_helpers import adverse_split_contract_for_grid, decision_grid_for_tape


def test_build_adverse_selection_dataset_to_disk_matches_in_memory_dataset(tmp_path):
    tape=_tape([_l2(seq=0, local_ts_us=100), _l2(seq=1, local_ts_us=1_300_000)], [_trade(local_ts_us=200, side=AggressorSide.SELL, price_tick=1000, amount=2.0, source_row=0)])
    cfg=_base_config()
    grid = decision_grid_for_tape(tape)
    mem=build_tiny_adverse_selection_dataset(tape, config=cfg, tmp_path=tmp_path, max_rows=grid.n_rows)
    split_contract = adverse_split_contract_for_grid(grid, root=str(tmp_path / "split_source"))["split_contract"]
    disk=build_adverse_selection_dataset_to_disk(tape, config=cfg, decision_grid=grid, split_contract=split_contract, output_root=tmp_path/"ds", chunk_rows=4096)
    np.testing.assert_array_equal(disk.arrays.decision_local_ts_us, mem.decision_local_ts_us)
    np.testing.assert_array_equal(disk.arrays.decision_event_index, mem.decision_event_index)
    np.testing.assert_array_equal(disk.arrays.decision_event_seq, mem.decision_event_seq)
    assert disk.feature_names == mem.feature_names
    assert disk.label_names == mem.label_names
    np.testing.assert_allclose(disk.arrays.features, mem.features, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(disk.arrays.labels, mem.labels, rtol=1e-6, atol=1e-6, equal_nan=True)
    np.testing.assert_array_equal(disk.arrays.label_masks, mem.label_masks)


def test_adverse_disk_builders_use_batch_writer_hot_paths():
    dataset_source = inspect.getsource(adverse_mod.build_adverse_selection_dataset_to_disk)
    assert "writer.append_many(" in dataset_source
    assert "writer.append(" not in dataset_source

    feature_source = inspect.getsource(feature_store_mod.build_adverse_selection_features_to_disk)
    assert "writer.append_many(" in feature_source
    assert "writer.append(" not in feature_source
