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
        if "DecisionSchedule(" in text:
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
        assert "--decision-interval-us" not in text
        assert "next_ts +=" not in text


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


def test_execution_env_clock_does_not_use_linear_signal_rows():
    text = _text("mmrt/execution/env.py")
    assert "np.searchsorted(self.linear_signals.decision_event_index" not in text
    assert "target_event_index = int(self.linear_signals.decision_event_index" not in text
    assert "target_local_ts_us = int(self.linear_signals.decision_local_ts_us" not in text
    assert "self.linear_signals.metadata.max_decision_interval_us" not in text
