import json
import gc
import numpy as np
import pytest

from mmrt.execution.adverse_selection_dataset import AdverseSelectionDatasetWriter, AdverseSelectionDatasetWriterConfig, load_adverse_selection_dataset
from tests.grid_helpers import grid_lineage_fields


def _meta():
    return {"exchange":"ex","symbol":"SYM","tape_schema":"schema","tape_num_events":1,"tape_num_l2_batches":1,"tape_num_trades":0,"tape_start_local_ts_us":1,"tape_end_local_ts_us":2,**grid_lineage_fields(n_rows=3),"config_json":"{}","index_schema":"mmrt_adverse_selection_index_v2","index_manifest_sha256":"0"*64,"index_root":"/tmp/index"}


def test_adverse_dataset_writer_roundtrip_tiny(tmp_path):
    w=AdverseSelectionDatasetWriter(AdverseSelectionDatasetWriterConfig(str(tmp_path/"ds"),("f",),("y",),_meta(),chunk_rows=2))
    w.append(decision_local_ts_us=1,decision_event_index=0,decision_event_seq=0,features=[1.5],labels=[2.5],label_masks=[True])
    ds=w.finalize()
    assert ds.num_rows==1
    np.testing.assert_allclose(ds.arrays.features, [[1.5]])
    assert bool(ds.arrays.label_masks[0,0])


def test_adverse_dataset_writer_chunking_multiple_chunks(tmp_path):
    w=AdverseSelectionDatasetWriter(AdverseSelectionDatasetWriterConfig(str(tmp_path/"ds"),("f",),("y",),_meta(),chunk_rows=1))
    for i in range(3):
        w.append(decision_local_ts_us=i+1,decision_event_index=i,decision_event_seq=0,features=[i],labels=[i+1],label_masks=[i%2==0])
    ds=w.finalize()
    assert ds.arrays.features.shape==(3,1)
    np.testing.assert_array_equal(ds.arrays.decision_event_index, [0,1,2])


def test_adverse_dataset_manifest_written_last(tmp_path):
    root=tmp_path/"ds"
    w=AdverseSelectionDatasetWriter(AdverseSelectionDatasetWriterConfig(str(root),("f",),("y",),_meta()))
    assert not (root/"manifest.json").exists()
    w.finalize()
    assert json.loads((root/"manifest.json").read_text())["num_rows"]==0


def test_load_adverse_dataset_rejects_bad_shapes(tmp_path):
    w=AdverseSelectionDatasetWriter(AdverseSelectionDatasetWriterConfig(str(tmp_path/"ds"),("f",),("y",),_meta()))
    ds=w.finalize()
    root = ds.root
    del ds
    gc.collect()
    features_path = root/"arrays"/"features.npy"
    features_path.unlink()
    np.save(features_path, np.zeros((2,1), dtype=np.float32))
    with pytest.raises(ValueError):
        load_adverse_selection_dataset(root)
