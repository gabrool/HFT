import inspect
import subprocess
import sys

import pytest

pytest.importorskip("pyarrow")

from mmrt.contracts import SplitRole, TimeRangeUS
from mmrt.features import specs
from mmrt.storage import manifest as mf
from mmrt.storage import reader as rd
from mmrt.storage import splits as sp
from mmrt.storage import writer as wr


def feature_values(scale=0.01):
    return tuple(float(i) * scale for i in range(specs.FEATURE_COUNT))


def label_values():
    return (1.0, -2.0, 0.5)


def row(i, *, local_ts_us=None, ts_us=None):
    t = 1_000_000 + i * 500_000
    lts = t if local_ts_us is None else local_ts_us
    return wr.DecisionRow(
        decision_index=i,
        ts_us=t if ts_us is None else ts_us,
        local_ts_us=lts,
        event_seq=i,
        raw_mid=100.0 + i,
        label_entry_ts_us=lts + 1_000,
        label_values=label_values(),
        feature_values=feature_values(),
    )


def make_dataset(tmp_path, *, rows=20, chunk_rows=5, row_fn=row):
    root = tmp_path / "d"
    w = wr.DecisionRowWriter(wr.WriterConfig(dataset_id="d", created_at_utc="2026", dataset_root=str(root), chunk_rows=chunk_rows))
    for i in range(rows):
        w.append(row_fn(i))
    manifest = w.finalize()
    return root, manifest


def test_public_api_boundary():
    assert sp.__all__ == [
        "DEFAULT_SPLIT_BATCH_SIZE",
        "SplitWindow",
        "SplitConfig",
        "SplitPlan",
        "chronological_windows",
        "build_split_plan",
        "apply_split_plan",
        "write_split_manifest",
        "build_and_write_splits",
    ]


def test_no_forbidden_imports():
    cmd = [sys.executable, "-c", "import mmrt.storage.splits as s; print('ok')"]
    out = subprocess.run(cmd, capture_output=True, text=True, check=True)
    assert "ok" in out.stdout
    src = inspect.getsource(sp)
    forbidden = [
        "pan" + "das", "po" + "lars", "to" + "rch", "sklearn", "mmrt.data.tardis_csv", "mmrt.data.event_merge",
        "mmrt.data.quality", "mmrt.features.engine", "mmrt.features.labels", "mmrt.features.transforms", "mmrt.storage.writer",
        "mmrt.linear", "CM" + "SSL", "offline_ingest",
    ]
    assert all(s not in src for s in forbidden)


def test_split_window_validation():
    sp.SplitWindow(role=SplitRole.TRAIN, start_local_ts_us=1, end_local_ts_us=2)
    sp.SplitWindow(role="train", start_local_ts_us=1, end_local_ts_us=2)
    with pytest.raises(ValueError):
        sp.SplitWindow(role="train", start_local_ts_us=0, end_local_ts_us=2)
    with pytest.raises(ValueError):
        sp.SplitWindow(role="train", start_local_ts_us=2, end_local_ts_us=2)
    with pytest.raises(ValueError):
        sp.SplitWindow(role="train", start_local_ts_us=True, end_local_ts_us=2)


def test_split_config_validation():
    a = sp.SplitWindow(role="train", start_local_ts_us=1, end_local_ts_us=10)
    b = sp.SplitWindow(role="val", start_local_ts_us=10, end_local_ts_us=20)
    sp.SplitConfig(windows=(a, b))
    with pytest.raises(ValueError):
        sp.SplitConfig(windows=())
    with pytest.raises(ValueError):
        sp.SplitConfig(windows=(b, a))
    with pytest.raises(ValueError):
        sp.SplitConfig(windows=(a, sp.SplitWindow(role="val", start_local_ts_us=9, end_local_ts_us=15)))


def test_chronological_windows_helper():
    ws = sp.chronological_windows(train=(1, 2), val=(2, 3), test=(3, 4))
    assert tuple(w.role for w in ws) == (SplitRole.TRAIN, SplitRole.VAL, SplitRole.TEST)


def test_build_and_apply(tmp_path):
    root, manifest = make_dataset(tmp_path)
    cfg = sp.SplitConfig(
        windows=sp.chronological_windows(train=(1_000_000, 4_000_000), val=(4_000_000, 7_000_000)),
        purge_before_us=0,
        purge_after_us=0,
        embargo_before_us=0,
        embargo_after_us=0,
    )
    plan = sp.build_split_plan(str(root), cfg)
    assert plan.dataset_id == manifest.dataset_id
    assert plan.roles == (SplitRole.TRAIN, SplitRole.VAL)
    assigned = []
    for e in plan.entries:
        assigned.extend(range(e.start_row, e.end_row))
    assert assigned == sorted(assigned)
    assert len(assigned) == len(set(assigned))

    updated = sp.write_split_manifest(str(root), plan)
    r = rd.open_dataset(str(root))
    t = r.read_split_table("train", columns=(mf.ROW_IDX_COLUMN,))
    assert t.num_rows > 0
    assert len(updated.splits) == len(plan.entries)


def test_build_split_plan_uses_local_ts_not_ts_us(tmp_path):
    def row_fn(i):
        return row(i, ts_us=10_000_000 + i, local_ts_us=1_000_000 + i * 500_000)

    root, _ = make_dataset(tmp_path, row_fn=row_fn)
    cfg = sp.SplitConfig(
        windows=(sp.SplitWindow("train", 1_000_000, 3_000_000),),
        purge_before_us=0,
        purge_after_us=0,
        embargo_before_us=0,
        embargo_after_us=0,
    )
    plan = sp.build_split_plan(str(root), cfg)
    assigned = [i for e in plan.entries for i in range(e.start_row, e.end_row)]
    assert assigned == [0, 1, 2, 3]


def test_source_guards():
    src = inspect.getsource(sp.build_split_plan)
    assert "ROW_IDX_COLUMN" in src and "LOCAL_TS_US_COLUMN" in src
    bad = ["x_columns", "y_columns", "feature_columns", "label_columns"]
    assert all(b not in src for b in bad)
    full = inspect.getsource(sp)
    bad2 = [
        "future_" + "mid", "future_" + "ret", "fit_" + "transform", "Standard" + "Scaler", "P" + "CA", "GRACE_" + "MS",
        "drop_duplicate_" + "trades", "stride_" + "rows", "sort_" + "values", "de" + "dupe", "sh" + "uffle", "rand" + "om",
    ]
    assert all(b not in full for b in bad2)
