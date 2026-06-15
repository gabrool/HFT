import numpy as np
import inspect

from mmrt.execution.adverse_selection_dataset import AdverseSelectionDatasetWriter, AdverseSelectionDatasetWriterConfig
from mmrt.execution.adverse_selection_index import ADVERSE_SELECTION_INDEX_SCHEMA
import mmrt.execution.adverse_selection_fit as fit_mod
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


def test_streaming_fit_matches_augmented_design_normal_equations(tmp_path):
    ds = _dataset(tmp_path, rows=8)
    target_names = ("bid_touch_filled", "cost")
    train_fraction = 0.75
    ridge_l2 = 1e-3
    min_train_samples = 2

    fit = fit_adverse_baselines_streaming(
        ds,
        target_names=target_names,
        train_fraction=train_fraction,
        ridge_l2=ridge_l2,
        min_train_samples=min_train_samples,
        chunk_rows=3,
        metrics_mode="none",
    )

    train_rows, _ = fit_mod._split(ds.num_rows, train_fraction)
    X_train = np.asarray(ds.arrays.features[:train_rows], dtype=np.float64)
    mean = X_train.mean(axis=0)
    var = X_train.var(axis=0)
    scale = np.where(np.sqrt(np.maximum(var, 0.0)) <= 1e-12, 1.0, np.sqrt(np.maximum(var, 0.0)))
    Xz_train = (X_train - mean) / scale
    augmented = np.concatenate([np.ones((train_rows, 1), dtype=np.float64), Xz_train], axis=1)
    label_index = {name: i for i, name in enumerate(ds.label_names)}
    reg = np.eye(ds.num_features + 1, dtype=np.float64) * ridge_l2
    reg[0, 0] = 0.0

    expected_names = []
    expected_betas = []
    for target_name in target_names:
        target_idx = label_index[target_name]
        train_mask = np.asarray(ds.arrays.label_masks[:train_rows, target_idx], dtype=np.bool_)
        val_count = int(np.count_nonzero(ds.arrays.label_masks[train_rows:, target_idx]))
        if int(np.count_nonzero(train_mask)) < min_train_samples or val_count == 0:
            continue
        rows = augmented[train_mask]
        y = np.asarray(ds.arrays.labels[:train_rows, target_idx], dtype=np.float64)[train_mask]
        expected_names.append(target_name)
        expected_betas.append(fit_mod._solve(rows.T @ rows + reg, rows.T @ y))

    expected = np.vstack(expected_betas)
    assert fit.target_names == tuple(expected_names)
    np.testing.assert_allclose(fit.intercepts, expected[:, 0], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(fit.coefficients, expected[:, 1:], rtol=1e-12, atol=1e-12)


def test_streaming_fit_avoids_augmented_design_matrix_concatenate():
    source = inspect.getsource(fit_mod.fit_adverse_baselines_streaming)
    assert "np.concatenate([np.ones" not in source
