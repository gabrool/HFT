from __future__ import annotations

from itertools import count
from pathlib import Path

import numpy as np

from mmrt.execution.adverse_selection import (
    AdverseSelectionConfig,
    AdverseSelectionDataset,
    build_adverse_selection_dataset_to_disk,
)
from tests.grid_helpers import decision_grid_for_tape


_DATASET_COUNTER = count()


def build_tiny_adverse_selection_dataset(
    tape,
    *,
    config: AdverseSelectionConfig,
    tmp_path: Path,
    max_rows: int | None = 1,
) -> AdverseSelectionDataset:
    dataset_id = next(_DATASET_COUNTER)
    grid = decision_grid_for_tape(tape, max_rows=max_rows)
    disk = build_adverse_selection_dataset_to_disk(
        tape,
        config=config,
        decision_grid=grid,
        output_root=tmp_path / f"adverse_ds_{dataset_id}",
        work_dir=tmp_path / f"adverse_work_{dataset_id}",
        chunk_rows=4096,
        overwrite=True,
    )
    return AdverseSelectionDataset(
        decision_local_ts_us=np.array(disk.arrays.decision_local_ts_us, copy=True),
        decision_event_index=np.array(disk.arrays.decision_event_index, copy=True),
        decision_event_seq=np.array(disk.arrays.decision_event_seq, copy=True),
        feature_names=disk.feature_names,
        features=np.array(disk.arrays.features, copy=True),
        label_names=disk.label_names,
        labels=np.array(disk.arrays.labels, copy=True),
        label_masks=np.array(disk.arrays.label_masks, copy=True),
        config=config,
        decision_grid_schema=disk.manifest.decision_grid_schema,
        decision_grid_hash=disk.manifest.decision_grid_hash,
        decision_grid_n_rows=disk.manifest.decision_grid_n_rows,
        decision_schedule=disk.manifest.decision_schedule,
    )
