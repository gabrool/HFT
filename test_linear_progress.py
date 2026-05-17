import numpy as np


def test_decision_row_count_matches_dataset_positions(monkeypatch):
    import linear_offline

    monkeypatch.setattr(linear_offline, "LINEAR_DECISION_STRIDE_ROWS", 5)
    monkeypatch.setattr(linear_offline, "LINEAR_DECISION_OFFSET_ROWS", 0)

    assert linear_offline.decision_row_count(20, 0) == 4
    assert linear_offline.decision_row_count(20, 2) == 2

    pos = linear_offline._dataset_positions(20, 0)
    assert linear_offline.decision_row_count(20, 0) == len(pos)


def test_decision_row_count_with_offset(monkeypatch):
    import linear_offline

    monkeypatch.setattr(linear_offline, "LINEAR_DECISION_STRIDE_ROWS", 5)
    monkeypatch.setattr(linear_offline, "LINEAR_DECISION_OFFSET_ROWS", 1)

    assert linear_offline.decision_row_count(20, 0) == 4


def test_progress_iter_rows_yields_all_items(monkeypatch, capsys):
    import linear_offline

    monkeypatch.setattr(linear_offline, "LINEAR_PROGRESS", True)
    monkeypatch.setattr(linear_offline, "LINEAR_PROGRESS_BACKEND", "log")
    monkeypatch.setattr(linear_offline, "LINEAR_PROGRESS_EVERY_SEC", 0.0001)

    items = [
        (np.zeros((3, 2), dtype=np.float32), np.zeros((3, 3), dtype=np.float32), np.arange(3)),
        (np.zeros((2, 2), dtype=np.float32), np.zeros((2, 3), dtype=np.float32), np.arange(2)),
    ]

    out = list(linear_offline.progress_iter_rows(items, total_rows=5, desc="unit"))
    assert len(out) == 2
    captured = capsys.readouterr().out
    assert "[linear-progress]" in captured
    assert "unit" in captured


def test_progress_iter_rows_propagates_consumer_exception(monkeypatch):
    import pytest
    import linear_offline

    monkeypatch.setattr(linear_offline, "LINEAR_PROGRESS", True)
    monkeypatch.setattr(linear_offline, "LINEAR_PROGRESS_BACKEND", "auto")

    items = [
        (
            np.zeros((1, 2), dtype=np.float32),
            np.zeros((1, 3), dtype=np.float32),
            None,
        )
    ]

    def consume():
        for _ in linear_offline.progress_iter_rows(items, total_rows=1, desc="unit"):
            raise RuntimeError("consumer failed")

    with pytest.raises(RuntimeError, match="consumer failed"):
        consume()


def test_progress_iter_rows_disabled(monkeypatch, capsys):
    import linear_offline

    monkeypatch.setattr(linear_offline, "LINEAR_PROGRESS", False)

    items = [(None, np.zeros((1, 3), dtype=np.float32), None)]
    list(linear_offline.progress_iter_rows(items, total_rows=1, desc="unit"))
    captured = capsys.readouterr().out
    assert "[linear-progress]" not in captured


def test_stage4_training_uses_progress_wrapper(monkeypatch):
    import linear_offline

    calls = []

    def fake_progress(iterable, *, total_rows, desc, row_getter=None):
        calls.append((total_rows, desc))
        yield from iterable

    class FakeModel:
        def partial_fit(self, X, y, classes=None, sample_weight=None):
            return self

    class FakeBundle:
        def __init__(self):
            self.config = {"alpha": 0.1}
            n_h = len(linear_offline.HORIZONS_MS)
            self.direction_models = [FakeModel() for _ in range(n_h)]
            self.mag_up_models = [FakeModel() for _ in range(n_h)]
            self.mag_down_models = [FakeModel() for _ in range(n_h)]
            self.fit_summary = {}

    class FakeDataset:
        def __len__(self):
            return 10

    n_h = len(linear_offline.HORIZONS_MS)

    def fake_iter(**kwargs):
        del kwargs
        yield np.ones((2, 4), dtype=np.float32), np.ones((2, n_h), dtype=np.float32), None

    def fake_masks(y, stats):
        del stats
        mask = np.ones_like(y, dtype=bool)
        return mask, mask, mask

    monkeypatch.setattr(linear_offline, "progress_iter_rows", fake_progress)
    monkeypatch.setattr(linear_offline, "compute_global_direction_weights_from_train_labels_plan", lambda **kwargs: [(1.0, 1.0) for _ in range(n_h)])
    monkeypatch.setattr(linear_offline, "initialize_stage4_candidate_bundle", lambda **kwargs: FakeBundle())
    monkeypatch.setattr(linear_offline, "iter_preprocessed_batches_from_train_plan", fake_iter)
    monkeypatch.setattr(linear_offline, "train_decision_row_count_from_plan", lambda plan, max_rows=0: 2)
    monkeypatch.setattr(linear_offline, "build_signed_side_trim_masks_from_stats_np", fake_masks)
    monkeypatch.setattr(linear_offline, "LINEAR_DECISION_STRIDE_ROWS", 5)
    monkeypatch.setattr(linear_offline, "LINEAR_DECISION_OFFSET_ROWS", 0)

    bundles = linear_offline.train_stage4_candidates_streaming_from_plan(
        extractor=object(),
        preprocess_bundle=object(),
        plan={"train_split_entries": [{}], "train_week_keys": ["w0"]},
        stats={},
        alpha_values=[0.1],
        config={
            "schema": "unit",
            "epochs": 1,
            "batch_rows": 2,
            "random_state": 17,
            "direction_weighting": "none",
            "mag_floor": 1e-4,
        },
    )

    assert len(bundles) == 1
    assert any("stage4 train epoch" in desc for _, desc in calls)
