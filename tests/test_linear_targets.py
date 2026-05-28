import inspect
import subprocess
import sys

import numpy as np
import pytest

pa = pytest.importorskip("pyarrow")

from mmrt.contracts import TimeRangeUS
from mmrt.linear import targets as tg
from mmrt.storage import manifest as mf


def make_manifest() -> mf.StorageManifest:
    seg = mf.StorageSegment(
        segment_key="seg-000",
        parquet_path="segments/seg-000.parquet",
        row_count=3,
        label_count=3,
        time_range=TimeRangeUS(1, 3),
        local_time_range=TimeRangeUS(1, 3),
        first_row_idx=0,
        last_row_idx=2,
    )
    return mf.make_manifest(dataset_id="ds", created_at_utc="2026-01-01T00:00:00Z", segments=(seg,))


def test_public_api_boundary():
    assert tg.__all__ == [
        "DEFAULT_TARGET_HORIZON_US",
        "DEFAULT_MOVE_DEADBAND_BPS",
        "DEFAULT_TARGET_DTYPE",
        "ALLOWED_TARGET_DTYPES",
        "DIRECTION_INVALID_CLASS",
        "DIRECTION_DOWN_CLASS",
        "DIRECTION_UP_CLASS",
        "LinearTargetConfig",
        "LinearTargetBatch",
        "LinearTargetBuilder",
        "target_column_for_horizon",
        "resolve_target_column",
        "target_column_projection",
        "table_to_return_vector",
        "build_linear_targets",
        "make_target_builder",
    ]
    names = " ".join(tg.__all__).lower()
    for token in ("bybit", "cmssl", "stage", "rocket", "hydra", "aeon", "torch", "sklearn", "pandas", "polars", "preprocess", "model", "evaluate"):
        assert token not in names


def test_no_forbidden_imports():
    code = """
import sys
before=set(sys.modules)
import mmrt.linear.targets
after=set(sys.modules)-before
forbidden=(
 'pan'+'das','po'+'lars','to'+'rch','sk'+'learn','aeon','sktime','num'+'ba',
 'mmrt.features.engine','mmrt.features.la'+'bels','mmrt.features.trans'+'forms',
 'mmrt.data.tardis_csv','mmrt.data.event_merge','mmrt.storage.reader','mmrt.storage.writer','mmrt.storage.splits',
 'CM'+'SSL17','offline_'+'ingest',
)
mods='\\n'.join(sorted(after))
for f in forbidden:
    assert all((m != f and not m.startswith(f + '.')) for m in after), (f, mods)
"""
    subprocess.run([sys.executable, "-c", code], check=True)


def test_config_validation():
    cfg = tg.LinearTargetConfig()
    assert cfg.target_horizon_us == 1_000_000
    assert cfg.move_deadband_bps == 0.0
    assert cfg.output_dtype == "float32"
    assert cfg.dtype == np.dtype("float32")
    assert tg.LinearTargetConfig(output_dtype="float64").dtype == np.dtype("float64")
    for bad in (0, True):
        with pytest.raises(ValueError):
            tg.LinearTargetConfig(target_horizon_us=bad)
    for bad in (-1, np.nan, np.inf, True):
        with pytest.raises(ValueError):
            tg.LinearTargetConfig(move_deadband_bps=bad)
    with pytest.raises(ValueError):
        tg.LinearTargetConfig(output_dtype="float16")
    for attr in ("target_type", "no_move_head", "stage", "mask_quantile", "scaler", "model"):
        assert not hasattr(cfg, attr)


def test_target_column_for_horizon():
    assert tg.target_column_for_horizon(1_000_000) == "y_ret_bps_1000000us"
    for bad in (0, -1, True):
        with pytest.raises(ValueError):
            tg.target_column_for_horizon(bad)


def test_resolve_target_column_from_manifest():
    m = make_manifest()
    cfg = tg.LinearTargetConfig()
    col = tg.resolve_target_column(m, cfg)
    assert col in m.label_columns
    assert tg.target_column_projection(m, cfg) == (col,)
    with pytest.raises(ValueError):
        tg.resolve_target_column(m, tg.LinearTargetConfig(target_horizon_us=123456))
    d = m.to_dict()
    d['label_columns'] = list(m.label_columns[:-1])
    with pytest.raises(ValueError):
        mf.StorageManifest.from_dict(d)


def test_table_to_return_vector_preserves_dtype_and_order():
    col = "y_ret_bps_1000000us"
    tbl = pa.table({col: [1.0, -2.0, 0.0]})
    out = tg.table_to_return_vector(tbl, col)
    assert out.dtype == np.float32
    assert out.flags.c_contiguous
    np.testing.assert_allclose(out, np.array([1.0, -2.0, 0.0], dtype=np.float32))
    out64 = tg.table_to_return_vector(tbl, col, output_dtype="float64")
    assert out64.dtype == np.float64
    z = tg.table_to_return_vector(pa.table({col: []}), col)
    assert z.shape == (0,)


def test_table_to_return_vector_rejects_bad_inputs():
    col = "y_ret_bps_1000000us"
    with pytest.raises(ValueError):
        tg.table_to_return_vector(pa.table({"other": [1.0]}), col)
    with pytest.raises(ValueError):
        tg.table_to_return_vector(pa.table({col: [1.0, np.nan]}), col)
    with pytest.raises(ValueError):
        tg.table_to_return_vector("not-table", col)
    with pytest.raises(ValueError):
        tg.table_to_return_vector(pa.table({col: [1.0]}), "")
    with pytest.raises(ValueError):
        tg.table_to_return_vector(pa.table({col: [1.0]}), col, output_dtype="float16")


def test_build_linear_targets_default_formulas():
    ret = np.array([2.0, -3.0, 0.0], dtype=np.float64)
    b = tg.build_linear_targets(ret)
    np.testing.assert_array_equal(b.y_direction, np.array([1, 0, -1], dtype=np.int8))
    np.testing.assert_array_equal(b.y_no_move, np.array([0.0, 0.0, 1.0], dtype=b.dtype))
    np.testing.assert_array_equal(b.move_mask, np.array([True, True, False]))
    np.testing.assert_array_equal(b.no_move_mask, np.array([False, False, True]))
    np.testing.assert_array_equal(b.up_move_mask, np.array([True, False, False]))
    np.testing.assert_array_equal(b.down_move_mask, np.array([False, True, False]))
    np.testing.assert_allclose(b.y_magnitude_up, np.log1p(np.array([2.0, 0.0, 0.0], dtype=np.float32)))
    np.testing.assert_allclose(b.y_magnitude_down, np.log1p(np.array([0.0, 3.0, 0.0], dtype=np.float32)))
    assert b.y_return_bps.dtype == np.float32
    assert b.n_rows == 3
    assert b.direction_valid_count == 2
    assert b.target_column == "y_ret_bps_1000000us"


def test_build_linear_targets_deadband():
    ret = np.array([-0.5, 0.0, 0.5, 0.5001, -0.5001], dtype=np.float32)
    cfg = tg.LinearTargetConfig(move_deadband_bps=0.5)
    b = tg.build_linear_targets(ret, config=cfg)
    np.testing.assert_array_equal(b.no_move_mask, np.array([True, True, True, False, False]))
    np.testing.assert_array_equal(b.move_mask, np.array([False, False, False, True, True]))
    np.testing.assert_array_equal(b.up_move_mask, np.array([False, False, False, True, False]))
    np.testing.assert_array_equal(b.down_move_mask, np.array([False, False, False, False, True]))
    np.testing.assert_array_equal(b.y_direction, np.array([-1, -1, -1, 1, 0], dtype=np.int8))
    assert np.array_equal(b.no_move_mask, ~b.move_mask)
    assert np.array_equal(b.move_mask, b.up_move_mask | b.down_move_mask)
    assert not np.any(b.up_move_mask & b.down_move_mask)


def test_linear_target_batch_validation_and_copy():
    ret = np.array([1.0, -1.0], dtype=np.float32)
    y_no_move = np.array([0.0, 0.0], dtype=np.float32)
    y_direction = np.array([1, 0], dtype=np.int8)
    y_up = np.array([1.0, 0.0], dtype=np.float32)
    y_down = np.array([0.0, 1.0], dtype=np.float32)
    no_move_mask = np.array([False, False], dtype=bool)
    move_mask = np.array([True, True], dtype=bool)
    up_move_mask = np.array([True, False], dtype=bool)
    down_move_mask = np.array([False, True], dtype=bool)

    b = tg.LinearTargetBatch(
        y_return_bps=ret,
        y_no_move=y_no_move,
        y_direction=y_direction,
        y_magnitude_up=y_up,
        y_magnitude_down=y_down,
        no_move_mask=no_move_mask,
        move_mask=move_mask,
        up_move_mask=up_move_mask,
        down_move_mask=down_move_mask,
        target_column="y_ret_bps_1000000us",
        horizon_us=1_000_000,
    )
    ret[0] = 9.0
    assert b.y_return_bps[0] == 1.0

    with pytest.raises(ValueError):
        tg.LinearTargetBatch(y_return_bps=np.array([1.0], np.float32), y_no_move=y_no_move, y_direction=y_direction, y_magnitude_up=y_up, y_magnitude_down=y_down, no_move_mask=no_move_mask, move_mask=move_mask, up_move_mask=up_move_mask, down_move_mask=down_move_mask, target_column="c", horizon_us=1)
    with pytest.raises(ValueError):
        tg.LinearTargetBatch(y_return_bps=np.array([np.nan, 1.0], np.float32), y_no_move=y_no_move, y_direction=y_direction, y_magnitude_up=y_up, y_magnitude_down=y_down, no_move_mask=no_move_mask, move_mask=move_mask, up_move_mask=up_move_mask, down_move_mask=down_move_mask, target_column="c", horizon_us=1)
    bad_ret = np.arange(4, dtype=np.float32)[::2]
    assert bad_ret.shape == (2,)
    assert not bad_ret.flags.c_contiguous
    with pytest.raises(ValueError):
        tg.LinearTargetBatch(y_return_bps=bad_ret, y_no_move=y_no_move, y_direction=y_direction, y_magnitude_up=y_up, y_magnitude_down=y_down, no_move_mask=no_move_mask, move_mask=move_mask, up_move_mask=up_move_mask, down_move_mask=down_move_mask, target_column="c", horizon_us=1)
    with pytest.raises(ValueError):
        tg.LinearTargetBatch(y_return_bps=np.array([1.0, 1.0], np.float32), y_no_move=np.array([0.0, 2.0], np.float32), y_direction=y_direction, y_magnitude_up=y_up, y_magnitude_down=y_down, no_move_mask=no_move_mask, move_mask=move_mask, up_move_mask=up_move_mask, down_move_mask=down_move_mask, target_column="c", horizon_us=1)
    with pytest.raises(ValueError):
        tg.LinearTargetBatch(y_return_bps=np.array([1.0, 1.0], np.float32), y_no_move=y_no_move, y_direction=y_direction, y_magnitude_up=y_up, y_magnitude_down=y_down, no_move_mask=np.array([True, False]), move_mask=move_mask, up_move_mask=up_move_mask, down_move_mask=down_move_mask, target_column="c", horizon_us=1)
    with pytest.raises(ValueError):
        tg.LinearTargetBatch(y_return_bps=np.array([1.0, 1.0], np.float32), y_no_move=y_no_move, y_direction=y_direction, y_magnitude_up=y_up, y_magnitude_down=y_down, no_move_mask=no_move_mask, move_mask=move_mask, up_move_mask=np.array([True, True]), down_move_mask=down_move_mask, target_column="c", horizon_us=1)
    with pytest.raises(ValueError):
        tg.LinearTargetBatch(y_return_bps=np.array([1.0, 1.0], np.float32), y_no_move=y_no_move, y_direction=np.array([1, 1], np.int8), y_magnitude_up=y_up, y_magnitude_down=y_down, no_move_mask=no_move_mask, move_mask=move_mask, up_move_mask=up_move_mask, down_move_mask=down_move_mask, target_column="c", horizon_us=1)
    with pytest.raises(ValueError):
        tg.LinearTargetBatch(y_return_bps=np.array([1.0, 1.0], np.float32), y_no_move=y_no_move, y_direction=np.array([0, 0], np.int8), y_magnitude_up=y_up, y_magnitude_down=y_down, no_move_mask=no_move_mask, move_mask=move_mask, up_move_mask=up_move_mask, down_move_mask=down_move_mask, target_column="c", horizon_us=1)
    with pytest.raises(ValueError):
        tg.LinearTargetBatch(y_return_bps=np.array([1.0, 1.0], np.float32), y_no_move=y_no_move, y_direction=y_direction, y_magnitude_up=np.array([-1.0, 0.0], np.float32), y_magnitude_down=y_down, no_move_mask=no_move_mask, move_mask=move_mask, up_move_mask=up_move_mask, down_move_mask=down_move_mask, target_column="c", horizon_us=1)


def test_builder_resolves_manifest_and_transforms_table():
    m = make_manifest()
    bld = tg.LinearTargetBuilder(manifest=m)
    assert bld.target_column in m.label_columns
    assert "direction_deadband_bps" not in bld.as_dict()
    tbl = pa.table({bld.target_column: [1.0, -2.0, 0.0]})
    batch = bld.transform_table(tbl)
    np.testing.assert_array_equal(batch.y_direction, np.array([1, 0, -1], dtype=np.int8))
    d = bld.as_dict()
    for k in ("target_horizon_us", "target_column", "move_deadband_bps", "output_dtype", "label_spec"):
        assert k in d
    for k in ("stage", "model", "preprocess"):
        assert k not in d


def test_builder_transform_numpy():
    bld = tg.LinearTargetBuilder()
    b = bld.transform_numpy(np.array([1.0, -1.0]))
    assert b.target_column == "y_ret_bps_1000000us"
    bld2 = tg.LinearTargetBuilder()
    bld2.transform_numpy(np.array([1.0]), target_column="y_ret_bps_1000000us")
    with pytest.raises(ValueError):
        bld2.transform_numpy(np.array([1.0]), target_column="y_ret_bps_2000000us")
    with pytest.raises(ValueError):
        bld.transform_numpy(np.array([np.nan]))


def test_no_recompute_label_or_timestamp_surface():
    src = inspect.getsource(tg)
    forbidden = [
        "LabelBuilder", "PriceHistory", "PriceObservation", "observe_price", "on_decision", "label_now",
        "local_" + "ts_us", "ts_" + "us", "event_seq", "raw_mid", "row_idx", "sort_values", "shuffle", "random",
        "future_" + "mid", "future_" + "ret",
    ]
    for tok in forbidden:
        assert tok not in src


def test_no_old_pipeline_residue():
    src = inspect.getsource(tg)
    forbidden = [
        "BY" + "BIT", "CM" + "SSL", "offline_" + "ingest",
        "stage" + "1", "stage" + "2", "stage" + "3", "stage" + "4", "stage" + "5",
        "Mini" + "Rocket", "Multi" + "Rocket", "Hy" + "dra", "Ae" + "on",
        "torch", "sklearn", "pandas", "polars",
    ]
    for tok in forbidden:
        assert tok not in src


def test_vectorized_no_row_loop_smoke():
    src = inspect.getsource(tg)
    for tok in (".iterrows", "to_pandas", "for i in range(len("):
        assert tok not in src
