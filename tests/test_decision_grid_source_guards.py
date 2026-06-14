from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _text(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def test_only_decision_grid_builder_applies_decision_schedule():
    offenders = []
    for path in (ROOT / "mmrt").rglob("*.py"):
        rel = path.relative_to(ROOT).as_posix()
        if rel in {"mmrt/cli/build_decision_grid.py", "mmrt/features/schedule.py"}:
            continue
        text = path.read_text(encoding="utf-8")
        if "DecisionSchedule(" in text or "DecisionSchedule," in text:
            offenders.append(rel)
    assert offenders == []


def test_downstream_clis_do_not_import_decision_schedule_or_fixed_grid_terms():
    downstream = [
        "mmrt/cli/ingest.py",
        "mmrt/cli/build_linear_signals.py",
        "mmrt/cli/audit_linear_execution_features.py",
        "mmrt/cli/train_adverse_selection.py",
        "mmrt/cli/build_adverse_selection_signals.py",
        "mmrt/cli/audit_execution_sim.py",
        "mmrt/cli/train_execution_ppo.py",
        "mmrt/cli/evaluate_execution_policy.py",
    ]
    for rel in downstream:
        text = _text(rel)
        assert "DecisionSchedule(" not in text
        assert "DecisionSchedule," not in text
        assert "--decision-interval-us" not in text
        assert "--min-decision-interval-us" not in text
        assert "--max-decision-interval-us" not in text
        assert "--no-wake-on-trade" not in text
        assert "--no-wake-on-top-of-book" not in text
        assert "--l1-size-change-fraction" not in text
        assert "next_ts +=" not in text


def test_non_execution_downstream_clis_do_not_generate_decision_ranges():
    downstream = [
        "mmrt/cli/ingest.py",
        "mmrt/cli/build_linear_signals.py",
        "mmrt/cli/audit_linear_execution_features.py",
        "mmrt/cli/train_adverse_selection.py",
        "mmrt/cli/build_adverse_selection_signals.py",
    ]
    for rel in downstream:
        text = _text(rel)
        assert "--start-event-index" not in text
        assert "--max-decisions" not in text


def test_adverse_production_code_has_no_fixed_grid_generation():
    adverse = [
        "mmrt/execution/adverse_selection.py",
        "mmrt/execution/adverse_selection_dataset.py",
        "mmrt/execution/adverse_selection_feature_store.py",
        "mmrt/cli/train_adverse_selection.py",
        "mmrt/cli/build_adverse_selection_signals.py",
    ]
    for rel in adverse:
        text = _text(rel)
        assert "decision_interval_us" not in text
        assert "next_ts +=" not in text


def test_feature_replay_exposes_only_grid_native_replay():
    text = _text("mmrt/execution/feature_replay.py")
    assert "def iter_tape_feature_steps(" not in text
    assert "def iter_decision_feature_chunks(" not in text
    assert "DecisionSchedule" not in text
    assert ".tolist()" not in text
    assert "np.ascontiguousarray(l2_events[" not in text
    for token in (
        'trade_price = list(',
        'trade_amount = list(',
        'trade_side = list(',
        'local_ts = list(',
        'book_ptrs = list(',
        'trade_ptrs = list(',
    ):
        assert token not in text


def test_adverse_disk_builders_do_not_materialize_index_samples():
    adverse = _text("mmrt/execution/adverse_selection.py")
    assert "def build_adverse_selection_feature_dataset(" not in adverse
    assert "def build_adverse_selection_dataset(" not in adverse
    assert "def _build_adverse_selection_feature_rows(" not in adverse
    assert "_precompute_kyle_samples(" not in adverse
    assert "_trade_flow_view_from_tape(" not in adverse
    assert "_valid_l2_view_from_tape(" not in adverse
    assert "_valid_l2_views(" not in adverse

    dataset_body = adverse.split("def build_adverse_selection_dataset_to_disk", 1)[1]
    assert "_precompute_kyle_samples(" not in dataset_body
    assert 'np.ascontiguousarray(events["local_ts_us"]' not in dataset_body
    assert "for i in range(index.kyle_samples.count)" not in dataset_body
    assert "[_KyleSample(" not in dataset_body

    feature_store = _text("mmrt/execution/adverse_selection_feature_store.py")
    assert "_precompute_kyle_samples(" not in feature_store
    assert "for i in range(index.kyle_samples.count)" not in feature_store
    assert "[_KyleSample(" not in feature_store


def test_large_replay_paths_do_not_use_unbounded_python_materialization():
    paths = [
        "mmrt/execution/feature_replay.py",
        "mmrt/execution/adverse_selection.py",
        "mmrt/execution/adverse_selection_feature_store.py",
        "mmrt/execution/adverse_selection_index.py",
    ]
    offenders = []
    for rel in paths:
        text = _text(rel)
        for token in (
            ".tolist()",
            'np.ascontiguousarray(events["local_ts_us"]',
            'np.ascontiguousarray(l2_events["',
            "local_ts = [",
            "event_seq = [",
            "flow = [",
            "trade_price = [",
            "trade_amount = [",
            "trade_side = [",
            "[_KyleSample(",
        ):
            if token in text:
                offenders.append(f"{rel}: {token}")
    assert offenders == []


def test_ingest_trade_counter_is_chunked():
    text = _text("mmrt/cli/ingest.py")
    assert 'np.asarray(events["event_type_code"][:replay_end]' not in text
    assert "def _count_event_type_in_range" in text


def test_decision_grid_has_no_npz_loader_or_flag_names():
    offenders = []
    for path in list((ROOT / "mmrt").rglob("*.py")) + list((ROOT / "tests").rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        for token in (
            "decision_grid" + ".npz",
            "--decision-grid" + "-npz",
            "decision_grid" + "_npz",
            "load_decision_grid" + "_npz",
            "save_decision_grid" + "_npz",
            "output_npz=str(" + "npz)",
        ):
            if token in text:
                offenders.append(f"{path.relative_to(ROOT).as_posix()}: {token}")
    assert offenders == []


def test_execution_env_clock_does_not_use_linear_signal_rows():
    text = _text("mmrt/execution/env.py")
    assert "np.searchsorted(self.linear_signals.decision_event_index" not in text
    assert "target_event_index = int(self.linear_signals.decision_event_index" not in text
    assert "target_local_ts_us = int(self.linear_signals.decision_local_ts_us" not in text
    assert "self.linear_signals.metadata.max_decision_interval_us" not in text
    assert "terminal_due_to_" + "signal" + "_end" not in text
    assert "next_" + "signal" + "_row" not in text
