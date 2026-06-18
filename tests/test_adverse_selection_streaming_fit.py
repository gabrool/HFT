import numpy as np
import inspect

from mmrt.execution.adverse_selection_dataset import AdverseSelectionDatasetWriter, AdverseSelectionDatasetWriterConfig
from mmrt.execution.adverse_selection_index import ADVERSE_SELECTION_INDEX_SCHEMA
import mmrt.execution.adverse_selection_fit as fit_mod
from mmrt.execution.adverse_selection_fit import fit_adverse_baselines_streaming
from tests.grid_helpers import adverse_split_contract_fields, grid_lineage_fields


def _dataset(tmp_path, rows=6, *, val_cost=None, test_cost=None):
    train_count = max(rows - 2, 1)
    ranges = {
        "train": [{"role": "train", "segment_key": "seg_000", "start_decision_row": 0, "end_decision_row": train_count, "row_count": train_count, "start_local_ts_us": 1, "end_local_ts_us": 1 + train_count, "embargo_before_us": 0, "embargo_after_us": 0}],
        "val": [{"role": "val", "segment_key": "seg_000", "start_decision_row": train_count, "end_decision_row": train_count + 1, "row_count": 1, "start_local_ts_us": 1 + train_count, "end_local_ts_us": 2 + train_count, "embargo_before_us": 0, "embargo_after_us": 0}],
        "test": [{"role": "test", "segment_key": "seg_000", "start_decision_row": train_count + 1, "end_decision_row": train_count + 2, "row_count": 1, "start_local_ts_us": 2 + train_count, "end_local_ts_us": 3 + train_count, "embargo_before_us": 0, "embargo_after_us": 0}],
    }
    split_fields = adverse_split_contract_fields(n_rows=rows, ranges=ranges)
    meta={"exchange":"ex","symbol":"SYM","tape_schema":"schema","tape_num_events":1,"tape_num_l2_batches":1,"tape_num_trades":0,"tape_start_local_ts_us":1,"tape_end_local_ts_us":2,**grid_lineage_fields(n_rows=rows),**split_fields,"config_json":"{}","index_schema":ADVERSE_SELECTION_INDEX_SCHEMA,"index_manifest_sha256":"0"*64,"index_root":"/tmp/index"}
    w=AdverseSelectionDatasetWriter(AdverseSelectionDatasetWriterConfig(str(tmp_path/"ds"),("x",),("bid_touch_filled","cost"),meta,chunk_rows=2))
    for i in range(rows):
        cost = float(2 * i + 1)
        if i == train_count and val_cost is not None:
            cost = float(val_cost)
        elif i == train_count + 1 and test_cost is not None:
            cost = float(test_cost)
        w.append(decision_local_ts_us=i+1, decision_event_index=i, decision_event_seq=0, features=[float(i)], labels=[float(i%2), cost], label_masks=[True, i != 1])
    return w.finalize()


def test_streaming_fit_respects_label_masks(tmp_path):
    ds=_dataset(tmp_path)
    fit=fit_adverse_baselines_streaming(ds, target_names=("cost",), split_contract=ds.manifest.split_contract, ridge_l2=0.0, min_train_samples=2, chunk_rows=2)
    assert fit.target_names == ("cost",)
    assert fit.metrics["targets"]["cost"]["train_rows"] == 3


def test_streaming_fit_skips_targets_with_insufficient_rows(tmp_path):
    ds=_dataset(tmp_path, rows=3)
    fit=fit_adverse_baselines_streaming(ds, target_names=("cost",), split_contract=ds.manifest.split_contract, ridge_l2=0.0, min_train_samples=10, chunk_rows=2)
    assert fit.target_names == ()
    assert fit.metrics["targets"]["cost"]["skipped"] is True


def test_streaming_metrics_approx_auc_reasonable(tmp_path):
    ds=_dataset(tmp_path)
    fit=fit_adverse_baselines_streaming(ds, target_names=("bid_touch_filled",), split_contract=ds.manifest.split_contract, ridge_l2=1e-3, min_train_samples=2, chunk_rows=2)
    auc=fit.metrics["targets"]["bid_touch_filled"]["val"]["auc"]
    assert auc is None or 0.0 <= auc <= 1.0


def test_streaming_fit_matches_augmented_design_normal_equations(tmp_path):
    ds = _dataset(tmp_path, rows=8)
    target_names = ("bid_touch_filled", "cost")
    ridge_l2 = 1e-3
    min_train_samples = 2

    fit = fit_adverse_baselines_streaming(
        ds,
        target_names=target_names,
        split_contract=ds.manifest.split_contract,
        ridge_l2=ridge_l2,
        min_train_samples=min_train_samples,
        chunk_rows=3,
        metrics_mode="none",
    )

    train_rows = int(fit.metrics["adverse_train_rows"])
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


def test_streaming_fit_uses_train_only_for_coefficients_and_separate_holdouts(tmp_path):
    ds = _dataset(tmp_path, rows=6, val_cost=10_000.0, test_cost=-10_000.0)
    fit = fit_adverse_baselines_streaming(
        ds,
        target_names=("cost",),
        split_contract=ds.manifest.split_contract,
        ridge_l2=0.0,
        min_train_samples=2,
        chunk_rows=2,
        metrics_mode="none",
    )

    assert fit.target_names == ("cost",)
    train_rows = int(fit.metrics["adverse_train_rows"])
    X_train = np.asarray(ds.arrays.features[:train_rows], dtype=np.float64)
    y_train = np.asarray(ds.arrays.labels[:train_rows, ds.label_names.index("cost")], dtype=np.float64)
    mask = np.asarray(ds.arrays.label_masks[:train_rows, ds.label_names.index("cost")], dtype=np.bool_)
    mean = X_train.mean(axis=0)
    scale = np.where(np.sqrt(np.maximum(X_train.var(axis=0), 0.0)) <= 1e-12, 1.0, np.sqrt(np.maximum(X_train.var(axis=0), 0.0)))
    Xz = (X_train - mean) / scale
    design = np.column_stack([np.ones(int(np.count_nonzero(mask)), dtype=np.float64), Xz[mask]])
    expected = fit_mod._solve(design.T @ design, design.T @ y_train[mask])

    np.testing.assert_allclose(fit.intercepts, [expected[0]], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(fit.coefficients, expected[1:].reshape(1, -1), rtol=1e-12, atol=1e-12)
    target_metrics = fit.metrics["targets"]["cost"]
    assert target_metrics["val"]["label_mean"] == 10_000.0
    assert target_metrics["test"]["label_mean"] == -10_000.0
    assert fit.metrics["selection_split"] == "val"
    assert fit.metrics["final_holdout_split"] == "test"


def test_streaming_fit_avoids_augmented_design_matrix_concatenate():
    source = inspect.getsource(fit_mod.fit_adverse_baselines_streaming)
    assert "np.concatenate([np.ones" not in source
