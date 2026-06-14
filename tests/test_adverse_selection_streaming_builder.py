import numpy as np

from mmrt.execution.adverse_selection import build_adverse_selection_dataset, build_adverse_selection_dataset_to_disk
from tests.test_adverse_selection import _tape, _l2, _trade, _base_config, AggressorSide
from tests.grid_helpers import decision_grid_for_tape


def test_build_adverse_selection_dataset_to_disk_matches_in_memory_dataset(tmp_path):
    tape=_tape([_l2(seq=0, local_ts_us=100), _l2(seq=1, local_ts_us=1_300_000)], [_trade(local_ts_us=200, side=AggressorSide.SELL, price_tick=1000, amount=2.0, source_row=0)])
    cfg=_base_config()
    grid = decision_grid_for_tape(tape)
    mem=build_adverse_selection_dataset(tape, config=cfg, decision_grid=grid)
    disk=build_adverse_selection_dataset_to_disk(tape, config=cfg, decision_grid=grid, output_root=tmp_path/"ds", chunk_rows=1)
    np.testing.assert_array_equal(disk.arrays.decision_local_ts_us, mem.decision_local_ts_us)
    np.testing.assert_array_equal(disk.arrays.decision_event_index, mem.decision_event_index)
    np.testing.assert_array_equal(disk.arrays.decision_event_seq, mem.decision_event_seq)
    assert disk.feature_names == mem.feature_names
    assert disk.label_names == mem.label_names
    np.testing.assert_allclose(disk.arrays.features, mem.features, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(disk.arrays.labels, mem.labels, rtol=1e-6, atol=1e-6, equal_nan=True)
    np.testing.assert_array_equal(disk.arrays.label_masks, mem.label_masks)
