import json
from pathlib import Path

from mmrt.cli.audit_linear_execution_features import main
from mmrt.execution.decision_grid import save_decision_grid_npz
from mmrt.execution.execution_tape import save_execution_tape
from mmrt.execution.linear_feature_audit import LinearExecutionFeatureAuditConfig, audit_linear_execution_features_from_config
from mmrt.linear import models as lm
from tests.test_execution_linear_signal_builder import _tiny_tape, _train_result
from tests.grid_helpers import decision_grid_for_tape


def _write_tape_and_grid(root):
    tape = _tiny_tape()
    grid = decision_grid_for_tape(tape, max_rows=2)
    save_execution_tape(tape, root, overwrite=True)
    save_decision_grid_npz(root / "decision_grid.npz", grid, overwrite=True)
    return grid


def _write_result(path, cols, grid):
    path.write_text(json.dumps(_train_result({head: cols for head in lm.MODEL_HEADS}, grid=grid).as_dict(), sort_keys=True, allow_nan=False), encoding="utf-8")
    return path


def test_audit_linear_execution_features_cli_tiny_tape(tmp_path):
    root=tmp_path/"tape"; grid=_write_tape_and_grid(root)
    cols=("x_mid_slope_bps_per_sec_1000000us",)
    result=_write_result(tmp_path/"linear.json", cols, grid)
    assert main(["--tape-root", str(root), "--decision-grid-npz", str(root/"decision_grid.npz"), "--linear-train-result-json", str(result), "--overwrite"]) == 0
    assert (root/"linear_execution_feature_audit.json").exists()


def test_audit_detects_collapsed_no_move_distribution(tmp_path):
    root=tmp_path/"tape"; grid=_write_tape_and_grid(root)
    cols=("x_mid_slope_bps_per_sec_1000000us",)
    result=_write_result(tmp_path/"linear.json", cols, grid)
    payload=json.loads(result.read_text())
    payload["model_bundle_state"]["no_move"]["intercept"] = 20.0
    result.write_text(json.dumps(payload), encoding="utf-8")
    out=audit_linear_execution_features_from_config(LinearExecutionFeatureAuditConfig(str(root), str(root/"decision_grid.npz"), str(result)))
    assert "p_no_move_collapsed" in out["warnings"]


def test_audit_reports_feature_zscore_outliers(tmp_path):
    root=tmp_path/"tape"; grid=_write_tape_and_grid(root)
    cols=("x_mid_slope_bps_per_sec_1000000us",)
    result=_write_result(tmp_path/"linear.json", cols, grid)
    payload=json.loads(result.read_text())
    for h in lm.MODEL_HEADS:
        st=payload["preprocess_state"]["states_by_head"][h]
        st["mean"]=[1e9]; st["variance"]=[1e-6]; st["scale"]=[0.001]; st["active_mask"]=[True]
    result.write_text(json.dumps(payload), encoding="utf-8")
    out=audit_linear_execution_features_from_config(LinearExecutionFeatureAuditConfig(str(root), str(root/"decision_grid.npz"), str(result)))
    assert "high_z_fraction" in out["warnings"]


def test_audit_rejects_missing_feature_columns(tmp_path):
    root=tmp_path/"tape"; grid=_write_tape_and_grid(root)
    result=_write_result(tmp_path/"linear.json", ("x_missing_feature",), grid)
    try:
        out=audit_linear_execution_features_from_config(LinearExecutionFeatureAuditConfig(str(root), str(root/"decision_grid.npz"), str(result)))
    except ValueError:
        return
    assert out["combined"]["feature_schema_match"] is False


def test_audit_summary_is_json_strict(tmp_path):
    root=tmp_path/"tape"; grid=_write_tape_and_grid(root)
    result=_write_result(tmp_path/"linear.json", ("x_mid_slope_bps_per_sec_1000000us",), grid)
    out=audit_linear_execution_features_from_config(LinearExecutionFeatureAuditConfig(str(root), str(root/"decision_grid.npz"), str(result)))
    json.dumps(out, allow_nan=False)


def test_linear_execution_feature_audit_does_not_import_rl_or_adverse():
    source = Path("mmrt/cli/audit_linear_execution_features.py").read_text()
    forbidden = ("mmrt.rl", "torch", "gym", "adverse_selection")
    assert all(token not in source for token in forbidden)
