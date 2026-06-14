import numpy as np

from mmrt.execution.adverse_selection_dataset import AdverseSelectionDatasetWriter, AdverseSelectionDatasetWriterConfig
from mmrt.execution.adverse_selection_index import ADVERSE_SELECTION_INDEX_SCHEMA
from mmrt.execution.adverse_selection_fit import fit_adverse_baselines_streaming
from tests.grid_helpers import grid_lineage_fields


def _dataset(tmp_path, rows=6):
    meta={"exchange":"ex","symbol":"SYM","tape_schema":"schema","tape_num_events":1,"tape_num_l2_batches":1,"tape_num_trades":0,"tape_start_local_ts_us":1,"tape_end_local_ts_us":2,**grid_lineage_fields(n_rows=rows),"config_json":"{}","index_schema":ADVERSE_SELECTION_INDEX_SCHEMA,"index_manifest_sha256":"0"*64,"index_root":"/tmp/index"}
    w=AdverseSelectionDatasetWriter(AdverseSelectionDatasetWriterConfig(str(tmp_path/"ds"),("x",),("bid_touch_filled","cost"),meta,chunk_rows=2))
    for i in range(rows):
        w.append(decision_local_ts_us=i+1, decision_event_index=i, decision_event_seq=0, features=[float(i)], labels=[float(i%2), float(2*i+1)], label_masks=[True, i != 1])
    return w.finalize()


def test_streaming_fit_respects_label_masks(tmp_path):
    ds=_dataset(tmp_path)
    fit=fit_adverse_baselines_streaming(ds, target_names=("cost",), train_fraction=0.7, ridge_l2=0.0, min_train_samples=2, chunk_rows=2)
    assert fit.target_names == ("cost",)
    assert fit.metrics["targets"]["cost"]["train_rows"] == 3


def test_streaming_fit_skips_targets_with_insufficient_rows(tmp_path):
    ds=_dataset(tmp_path, rows=3)
    fit=fit_adverse_baselines_streaming(ds, target_names=("cost",), train_fraction=0.7, ridge_l2=0.0, min_train_samples=10, chunk_rows=2)
    assert fit.target_names == ()
    assert fit.metrics["targets"]["cost"]["skipped"] is True


def test_streaming_metrics_approx_auc_reasonable(tmp_path):
    ds=_dataset(tmp_path)
    fit=fit_adverse_baselines_streaming(ds, target_names=("bid_touch_filled",), train_fraction=0.7, ridge_l2=1e-3, min_train_samples=2, chunk_rows=2)
    auc=fit.metrics["targets"]["bid_touch_filled"]["val_auc"]
    assert auc is None or 0.0 <= auc <= 1.0
