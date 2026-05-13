"""Smoke checks for FeatureEngine event-result API contract."""

import sys
import types
from collections import deque

import numpy as np


def _install_optional_dependency_stubs() -> None:
    """Provide import-time stubs for model-only dependencies unused by this smoke test."""

    class _Dummy:
        def __init__(self, *args, **kwargs):
            pass

        def __call__(self, *args, **kwargs):
            return self

        def __getattr__(self, _name):
            return self

        def __getitem__(self, _key):
            return self

        def __setitem__(self, _key, _value):
            pass

    class _Module:
        def __init__(self, *args, **kwargs):
            pass

    class _Parameter:
        def __init__(self, value=None, *args, **kwargs):
            self.value = value

    torch_mod = types.ModuleType("torch")
    torch_mod.Tensor = _Dummy
    torch_mod.cuda = types.SimpleNamespace(empty_cache=lambda: None)
    torch_mod.ones = lambda *args, **kwargs: np.ones(args[0] if args else (), dtype=np.float32)
    torch_mod.tensor = lambda *args, **kwargs: np.asarray(args[0] if args else 0)
    torch_mod.empty = lambda *args, **kwargs: np.empty(args[0] if len(args) == 1 else args, dtype=np.float32)
    torch_mod.randn = lambda *args, **kwargs: np.random.randn(*args)
    torch_mod.exp = np.exp
    torch_mod.log = np.log
    torch_mod.arange = lambda *args, **kwargs: np.arange(*args)
    torch_mod.float32 = np.float32
    torch_mod.no_grad = lambda func=None: (lambda f: f) if func is None else func

    nn_mod = types.ModuleType("torch.nn")
    nn_mod.Module = _Module
    nn_mod.Parameter = _Parameter
    for name in (
        "Linear",
        "Conv1d",
        "SiLU",
        "ReLU",
        "GELU",
        "Dropout",
        "LayerNorm",
        "BatchNorm1d",
        "Sequential",
        "ModuleList",
    ):
        setattr(nn_mod, name, type(name, (_Module,), {}))
    nn_mod.init = types.SimpleNamespace(
        uniform_=lambda *args, **kwargs: None,
        normal_=lambda *args, **kwargs: None,
        zeros_=lambda *args, **kwargs: None,
        constant_=lambda *args, **kwargs: None,
    )
    functional_mod = types.ModuleType("torch.nn.functional")
    utils_mod = types.ModuleType("torch.utils")
    data_mod = types.ModuleType("torch.utils.data")
    data_mod.Dataset = type("Dataset", (_Module,), {})
    data_mod.DataLoader = type("DataLoader", (_Module,), {})
    functorch_mod = types.ModuleType("torch._functorch")
    config_mod = types.ModuleType("torch._functorch.config")
    optim_mod = types.ModuleType("torch.optim")
    optim_mod.Optimizer = type("Optimizer", (_Module,), {"__init__": lambda self, *args, **kwargs: None})
    torch_mod.optim = optim_mod
    torch_mod.nn = nn_mod
    torch_mod.utils = utils_mod
    torch_mod._functorch = functorch_mod

    sys.modules.setdefault("torch", torch_mod)
    sys.modules.setdefault("torch.nn", nn_mod)
    sys.modules.setdefault("torch.nn.functional", functional_mod)
    sys.modules.setdefault("torch.optim", optim_mod)
    sys.modules.setdefault("torch.utils", utils_mod)
    sys.modules.setdefault("torch.utils.data", data_mod)
    sys.modules.setdefault("torch._functorch", functorch_mod)
    sys.modules.setdefault("torch._functorch.config", config_mod)

    tqdm_mod = types.ModuleType("tqdm")
    tqdm_mod.tqdm = lambda iterable=None, *args, **kwargs: iterable if iterable is not None else []
    sys.modules.setdefault("tqdm", tqdm_mod)

    einops_mod = types.ModuleType("einops")
    einops_mod.rearrange = lambda x, *args, **kwargs: x
    einops_mod.repeat = lambda x, *args, **kwargs: x
    sys.modules.setdefault("einops", einops_mod)


    hub_mod = types.ModuleType("huggingface_hub")
    hub_mod.PyTorchModelHubMixin = type("PyTorchModelHubMixin", (), {})
    sys.modules.setdefault("huggingface_hub", hub_mod)

    mamba_modules = {
        "mamba_ssm": {},
        "mamba_ssm.ops": {},
        "mamba_ssm.ops.triton": {},
        "mamba_ssm.ops.triton.selective_state_update": {"selective_state_update": None},
        "mamba_ssm.ops.triton.layernorm_gated": {"RMSNorm": type("RMSNorm", (_Module,), {})},
        "mamba_ssm.distributed": {},
        "mamba_ssm.distributed.tensor_parallel": {
            "ColumnParallelLinear": type("ColumnParallelLinear", (_Module,), {}),
            "RowParallelLinear": type("RowParallelLinear", (_Module,), {}),
        },
        "mamba_ssm.distributed.distributed_utils": {
            "all_reduce": lambda *args, **kwargs: None,
            "reduce_scatter": lambda *args, **kwargs: None,
        },
        "mamba_ssm.ops.triton.ssd_combined": {
            "mamba_chunk_scan_combined": lambda *args, **kwargs: None,
            "mamba_split_conv1d_scan_combined": lambda *args, **kwargs: None,
        },
    }
    for module_name, attrs in mamba_modules.items():
        module = types.ModuleType(module_name)
        for attr_name, attr_value in attrs.items():
            setattr(module, attr_name, attr_value)
        sys.modules.setdefault(module_name, module)


_install_optional_dependency_stubs()

from CMSSL17 import FeatureEngine, FeatureEventResult, LabelBuilder


def assert_not_tuple_unpackable(result: FeatureEventResult) -> None:
    try:
        _a, _b, _c, _d, _e = result
        raise AssertionError("FeatureEventResult must not support tuple unpacking")
    except TypeError:
        pass


def snapshot_ob(ts: int):
    return (
        "ob",
        ts,
        1,
        1,
        ((100.0, 2.0), (99.5, 1.5), (99.0, 1.0), (98.5, 1.0), (98.0, 1.0)),
        ((101.0, 2.5), (101.5, 1.0), (102.0, 1.0), (102.5, 1.0), (103.0, 1.0)),
    )


def delta_ob(ts: int):
    return (
        "ob",
        ts,
        2,
        2,
        ((100.0, 2.25),),
        ((101.0, 2.25),),
    )


def deep_snapshot_ob(ts: int, n_levels: int = 60):
    bids = tuple((100.0 - 0.5 * i, 1.0 + 0.01 * i) for i in range(n_levels))
    asks = tuple((101.0 + 0.5 * i, 1.0 + 0.01 * i) for i in range(n_levels))
    return ("ob", ts, 1, 1, bids, asks)


def crossed_snapshot_ob(ts: int):
    return (
        "ob",
        ts,
        1,
        1,
        ((101.0, 1.0),),
        ((100.0, 1.0),),
    )


def empty_bid_snapshot_ob(ts: int):
    return (
        "ob",
        ts,
        1,
        1,
        tuple(),
        ((101.0, 1.0),),
    )


def locked_snapshot_ob(ts: int):
    return (
        "ob",
        ts,
        1,
        1,
        ((100.0, 1.0),),
        ((100.0, 1.0),),
    )


def malformed_delta_ob(ts: int):
    return (
        "ob",
        ts,
        2,
        2,
        ((float("nan"), 5.0), (-1.0, 1.0), (100.0, 2.25)),
        ((float("inf"), 1.0), (0.0, 1.0), (101.0, 2.25)),
    )


def delete_top_bid_levels_ob(ts: int, n_delete: int):
    bids = tuple((100.0 - 0.5 * i, 0.0) for i in range(n_delete))
    asks = tuple()
    return ("ob", ts, 2, 2, bids, asks)


def delete_top_ask_levels_ob(ts: int, n_delete: int):
    bids = tuple()
    asks = tuple((101.0 + 0.5 * i, 0.0) for i in range(n_delete))
    return ("ob", ts, 3, 2, bids, asks)


def trade(ts: int):
    return (
        "trade",
        ts,
        10,
        100.5,
        0.25,
        1,
        0,
        0,
    )



def test_merge_event_time_trade_wins_exact_timestamp_tie() -> None:
    import offline_ingest

    ob_events = iter([
        ("ob", 1000, 1, 1, ((100.0, 1.0),), ((101.0, 1.0),)),
    ])
    tr_events = iter([
        ("trade", 1000, 10, 100.5, 0.1, 1, 0, 0),
    ])

    merged = list(offline_ingest.merge_event_time(ob_events, tr_events, dq_day=None, strict=True))
    assert [e[0] for e in merged] == ["trade", "ob"]
    assert [e[1] for e in merged] == [1000, 1000]


def test_merge_event_time_lower_timestamp_still_wins() -> None:
    import offline_ingest

    ob_events = iter([("ob", 1000, 1, 1, ((100.0, 1.0),), ((101.0, 1.0),))])
    tr_events = iter([("trade", 999, 10, 100.5, 0.1, 1, 0, 0)])
    merged = list(offline_ingest.merge_event_time(ob_events, tr_events, dq_day=None, strict=True))
    assert [e[0] for e in merged] == ["trade", "ob"]
    assert [e[1] for e in merged] == [999, 1000]


def test_merge_event_time_ob_lower_timestamp_still_wins() -> None:
    import offline_ingest

    ob_events = iter([("ob", 999, 1, 1, ((100.0, 1.0),), ((101.0, 1.0),))])
    tr_events = iter([("trade", 1000, 10, 100.5, 0.1, 1, 0, 0)])
    merged = list(offline_ingest.merge_event_time(ob_events, tr_events, dq_day=None, strict=True))
    assert [e[0] for e in merged] == ["ob", "trade"]
    assert [e[1] for e in merged] == [999, 1000]


def process_decision_results_for_test(results):
    pending = deque()
    labeler = LabelBuilder(delta_ms=0, horizons_ms=[200, 500, 1000])
    rows = []
    last_decision_ts = None

    for result in results:
        if not result.is_decision:
            continue

        ts = int(result.ts_ms)
        if last_decision_ts is not None and ts < last_decision_ts:
            raise RuntimeError("non-monotone")

        row_idx = len(rows)
        rows.append((ts, result.raw_mid))
        pending.append(("week", row_idx, ts))
        labeler.on_decision(ts)

        matured = labeler.on_event(ts, float(result.raw_mid))
        for _yy in matured:
            pending.popleft()

        last_decision_ts = ts

    return rows, pending, labeler


def test_duplicate_ob_timestamps_append_distinct_rows() -> None:
    fe = FeatureEngine()
    r0 = fe.on_fast_event(deep_snapshot_ob(1000, n_levels=60))
    r1 = fe.on_fast_event(("ob", 1000, 2, 2, ((100.0, 2.5),), ((101.0, 2.5),)))

    rows, pending, labeler = process_decision_results_for_test([r0, r1])

    assert len(rows) == 2
    assert rows[0][0] == 1000
    assert rows[1][0] == 1000
    assert len(pending) == 2
    assert len(labeler.wait_delta) == 0
    assert len(labeler.wait_mature) == 2


def test_offline_ingest_no_overwrite_duplicate_timestamp_api() -> None:
    from pathlib import Path

    text = Path("offline_ingest.py").read_text()
    assert "overwrite_latest_feature_row" not in text


def test_snapshot_stores_full_book_but_features_use_top_depth() -> None:
    fe = FeatureEngine()
    result = fe.on_fast_event(deep_snapshot_ob(1_700_000_100_000, n_levels=60))

    assert result.event_type == "ob"
    assert result.is_decision is True
    assert result.features.shape[0] > 0

    assert fe.feature_depth == 20
    assert len(fe.bids) == 60
    assert len(fe.asks) == 60
    assert len(fe.bid_lvls) == fe.feature_depth
    assert len(fe.ask_lvls) == fe.feature_depth

    assert fe.bid_lvls[0][0] == 100.0
    assert fe.ask_lvls[0][0] == 101.0
    assert fe.bid_lvls[-1][0] == 100.0 - 0.5 * 19
    assert fe.ask_lvls[-1][0] == 101.0 + 0.5 * 19


def test_deeper_snapshot_levels_promote_after_top_level_deletes() -> None:
    fe = FeatureEngine()
    fe.on_fast_event(deep_snapshot_ob(1_700_000_200_000, n_levels=60))

    before_visible_bid_prices = [px for px, _ in fe.bid_lvls]
    assert 100.0 - 0.5 * 20 not in before_visible_bid_prices
    assert 100.0 - 0.5 * 20 in fe.bids

    result = fe.on_fast_event(delete_top_bid_levels_ob(1_700_000_200_100, n_delete=5))
    assert result.event_type == "ob"
    assert result.is_decision is True

    after_visible_bid_prices = [px for px, _ in fe.bid_lvls]

    # Old levels 5..24 should now be visible after deleting original top 5.
    assert after_visible_bid_prices[0] == 100.0 - 0.5 * 5
    assert after_visible_bid_prices[-1] == 100.0 - 0.5 * 24

    # Proves level 21+ was preserved and promoted.
    assert 100.0 - 0.5 * 20 in after_visible_bid_prices
    assert len(fe.bid_lvls) == fe.feature_depth


def test_deeper_ask_snapshot_levels_promote_after_top_level_deletes() -> None:
    fe = FeatureEngine()
    fe.on_fast_event(deep_snapshot_ob(1_700_000_300_000, n_levels=60))

    before_visible_ask_prices = [px for px, _ in fe.ask_lvls]
    assert 101.0 + 0.5 * 20 not in before_visible_ask_prices
    assert 101.0 + 0.5 * 20 in fe.asks

    result = fe.on_fast_event(delete_top_ask_levels_ob(1_700_000_300_100, n_delete=5))
    assert result.event_type == "ob"
    assert result.is_decision is True

    after_visible_ask_prices = [px for px, _ in fe.ask_lvls]

    assert after_visible_ask_prices[0] == 101.0 + 0.5 * 5
    assert after_visible_ask_prices[-1] == 101.0 + 0.5 * 24
    assert 101.0 + 0.5 * 20 in after_visible_ask_prices
    assert len(fe.ask_lvls) == fe.feature_depth


def test_book_health_validation_rejects_bad_snapshots() -> None:
    for event in (
        crossed_snapshot_ob(1_700_000_400_000),
        empty_bid_snapshot_ob(1_700_000_400_100),
        locked_snapshot_ob(1_700_000_400_200),
    ):
        fe = FeatureEngine()
        try:
            fe.on_fast_event(event)
            raise AssertionError("Bad snapshot should have raised")
        except Exception as exc:
            assert exc.__class__.__name__ == "BookValidationError"


def test_malformed_delta_levels_are_ignored_but_valid_updates_apply() -> None:
    fe = FeatureEngine()
    fe.on_fast_event(deep_snapshot_ob(1_700_000_500_000, n_levels=60))

    result = fe.on_fast_event(malformed_delta_ob(1_700_000_500_100))
    assert result.event_type == "ob"
    assert result.is_decision is True

    assert fe.bids[100.0] == 2.25
    assert fe.asks[101.0] == 2.25
    assert -1.0 not in fe.bids
    assert 0.0 not in fe.asks


def test_compact_ob_type_code_does_not_default_to_delta() -> None:
    import offline_ingest

    assert offline_ingest._compact_ob_type_code("snapshot") == offline_ingest.OB_TP_SNAPSHOT
    assert offline_ingest._compact_ob_type_code("delta") == offline_ingest.OB_TP_DELTA
    assert offline_ingest._compact_ob_type_code(None) == 0
    assert offline_ingest._compact_ob_type_code("") == 0
    assert offline_ingest._compact_ob_type_code("weird") == 0


def generic_dict_ob(ts: int, ob_type=None):
    event = {
        "ts": ts,
        "data": {
            "b": [[100.0, 2.0], [99.5, 1.5]],
            "a": [[101.0, 2.5], [101.5, 1.0]],
        },
    }
    if ob_type is not None:
        event["type"] = ob_type
    return event


def test_generic_dict_ob_type_parsing_is_explicit() -> None:
    fe = FeatureEngine()

    snapshot = fe.on_event(generic_dict_ob(1_700_000_600_000, "snapshot"))
    assert snapshot.event_type == "ob"
    assert snapshot.is_decision is True
    assert snapshot.features.shape[0] > 0

    delta = fe.on_event(generic_dict_ob(1_700_000_600_100, "delta"))
    assert delta.event_type == "ob"
    assert delta.is_decision is True
    assert delta.features.shape[0] > 0

    for bad_event in (
        generic_dict_ob(1_700_000_600_200),
        generic_dict_ob(1_700_000_600_300, "weird"),
    ):
        bad_fe = FeatureEngine()
        try:
            bad_fe.on_event(bad_event)
            raise AssertionError("Generic dict OB with missing/unknown type should have raised")
        except ValueError as exc:
            assert "Missing/unknown OB type in generic on_event payload" in str(exc)


def test_trade_does_not_pollute_ob_feature_state() -> None:
    fe = FeatureEngine()

    ob0 = fe.on_fast_event(snapshot_ob(1_700_000_000_000))
    assert ob0.is_decision is True
    assert ob0.event_type == "ob"
    assert ob0.features.shape[0] > 0

    z_mean_before = None if fe.z_mean is None else fe.z_mean.copy()
    z_m2_before = None if fe.z_m2 is None else fe.z_m2.copy()
    last_z_ts_before = fe._last_z_ts_ms
    mid_hist_len_before = len(fe._mid_history)
    micro_hist_len_before = len(fe._micro_history)
    ob_snapshots_len_before = len(fe._ob_snapshots)
    ob_snapshot_ts_len_before = len(fe._ob_snapshot_ts_ms)
    return_counts_before = {ms: len(stats.deq) for ms, stats in fe.return_histories.items()}
    last_ob_feature_ts_before = fe._last_ob_feature_ts
    ob_feature_build_count_before = fe.ob_feature_build_count

    prev_bsz_before = fe.prev_bsz
    prev_asz_before = fe.prev_asz
    prev_bsz2_before = fe.prev_bsz2
    prev_asz2_before = fe.prev_asz2
    prev_cum_bid_before = dict(fe.prev_cum_bid_by_level)
    prev_cum_ask_before = dict(fe.prev_cum_ask_by_level)

    tr = fe.on_fast_event(trade(1_700_000_000_010))
    assert tr.event_type == "trade"
    assert tr.is_decision is False
    assert tr.features.shape == (0,)

    assert fe.trade_fast_path_count == 1
    assert fe.last_trade_ts == 1_700_000_000_010
    assert fe.last_trade_price == 100.5
    assert fe._last_any_event_ts == 1_700_000_000_010

    assert fe.ob_feature_build_count == ob_feature_build_count_before
    assert fe._last_ob_feature_ts == last_ob_feature_ts_before
    assert len(fe._mid_history) == mid_hist_len_before
    assert len(fe._micro_history) == micro_hist_len_before
    assert len(fe._ob_snapshots) == ob_snapshots_len_before
    assert len(fe._ob_snapshot_ts_ms) == ob_snapshot_ts_len_before
    assert {ms: len(stats.deq) for ms, stats in fe.return_histories.items()} == return_counts_before

    assert fe.prev_bsz == prev_bsz_before
    assert fe.prev_asz == prev_asz_before
    assert fe.prev_bsz2 == prev_bsz2_before
    assert fe.prev_asz2 == prev_asz2_before
    assert dict(fe.prev_cum_bid_by_level) == prev_cum_bid_before
    assert dict(fe.prev_cum_ask_by_level) == prev_cum_ask_before

    if z_mean_before is None:
        assert fe.z_mean is None
    else:
        assert np.array_equal(fe.z_mean, z_mean_before)

    if z_m2_before is None:
        assert fe.z_m2 is None
    else:
        assert np.array_equal(fe.z_m2, z_m2_before)

    assert fe._last_z_ts_ms == last_z_ts_before

    ob1 = fe.on_fast_event(delta_ob(1_700_000_000_100))
    assert ob1.event_type == "ob"
    assert ob1.is_decision is True
    assert ob1.features.shape[0] > 0
    assert fe.ob_feature_build_count == ob_feature_build_count_before + 1
    assert fe._last_ob_feature_ts == 1_700_000_000_100
    assert fe._last_any_event_ts == 1_700_000_000_100
    assert len(fe._mid_history) == mid_hist_len_before + 1


def test_feature_transform_contract_is_raw_no_projection() -> None:
    import CMSSL17

    fe = FeatureEngine()
    raw_names = list(fe.feature_names())
    forbidden = "p" + "ca"
    assert len(raw_names) > 0
    assert CMSSL17.FEATURE_TRANSFORM == "raw_zscore_plus_aux_no_" + forbidden + "_v1"
    assert "p" + "ca250" not in CMSSL17.FEATURE_SCHEMA.lower()
    assert "final256" not in CMSSL17.FEATURE_SCHEMA.lower()
    assert "p" + "ca250" not in CMSSL17.CHECKPOINT_SCHEMA.lower()
    assert "final256" not in CMSSL17.CHECKPOINT_SCHEMA.lower()


def test_offline_ingest_raw_feature_dims() -> None:
    import CMSSL17
    import offline_ingest

    raw_names = list(FeatureEngine().feature_names())
    assert offline_ingest.RAW_FEATURE_DIM_CORE == len(raw_names)
    assert offline_ingest.RAW_FEATURE_DIM_TOTAL == len(raw_names) + CMSSL17.AUX_DIM
    assert offline_ingest.RAW_FEATURE_NAMES == raw_names
    assert offline_ingest.FEATURE_TRANSFORM == CMSSL17.FEATURE_TRANSFORM


def test_pruned_feature_schema_contract() -> None:
    fe = FeatureEngine()
    names = list(fe.feature_names())

    assert len(names) == 235
    assert fe.core_feature_dim() == 235
    assert fe.feature_dim() == 241

    removed = {
        "time_hour_sin",
        "time_hour_cos",
        "time_dow_sin",
        "time_dow_cos",
        "session_is_weekend",
        "session_is_asia",
        "session_is_europe",
        "session_is_us",
        "session_is_europe_us_overlap",

        "mid_trend_r2_200ms",
        "mid_position_in_range_200ms",
        "mid_breakout_up_200ms",
        "mid_breakout_down_200ms",
        "sign_persistence_200ms",
        "up_return_fraction_200ms",
        "return_autocorr_lag1_200ms",

        "cum_bid_l1",
        "cum_ask_l1",
        "cum_bid_l20",
        "cum_ask_l20",
        "obi_l20",
        "ofi_l20",

        "ofi_l1_over_spread_bps",
        "ofi_l3_over_spread_bps",
        "ofi_l5_over_spread_bps",
        "ofi_l10_over_spread_bps",

        "ofi_l1_sum_200ms",
        "ofi_l3_sum_500ms",
        "ofi_l10_sum_1000ms",
        "ofi_l1_accel_200_minus_1000ms",

        "micro_l3_minus_mid_over_spread",
        "micro_l5_minus_mid_over_spread",
        "micro_l10_minus_mid_over_spread",

        "bid_notional_within_1bps",
        "ask_notional_within_1bps",
        "notional_imbalance_within_10bps",

        "book_slope_bid_top5",
        "book_convexity_ask_10bps",

        "spread_delta_bps_200ms",
        "spread_delta_bps_500ms",
        "spread_delta_bps_1000ms",

        "vwap_vs_micro_bps_200ms",
        "signed_trade_premium_bps_count_weighted_500ms",
        "buy_trade_premium_bps_1000ms",
        "sell_trade_premium_bps_1000ms",

        "large_trade_cluster_count_200ms",
        "time_since_large_buy_ms",
        "last_large_sell_notional_usd",
        "return_since_last_large_buy_bps",

        "signed_flow_per_bp_move_200ms",
        "price_response_to_buy_flow_500ms",
        "price_response_to_sell_flow_1000ms",

        "variance_ratio_500ms_over_200ms",
        "variance_ratio_1000ms_over_500ms",

        "resid_spread_bps_200ms",
        "resid_micro_minus_mid_bps_500ms",
        "resid_trade_imbalance_notional_1000ms_1000ms",
    }

    for name in removed:
        assert name not in names, name

    retained = {
        "mid_ret_bps_200ms",
        "micro_ret_bps_500ms",
        "mid_slope_bps_per_sec_1000ms",
        "mid_range_bps_1000ms",
        "spread_bps",
        "bsz1",
        "asz1",
        "micro_minus_mid_bps",
        "micro_minus_mid_over_spread",
        "time_since_trade_ms",
        "time_since_mid_change_ms",
        "obi_l10",
        "ofi_l10",
        "ofi_l10_over_depth_l10",
        "ofi_l10_over_depth_5bps",
        "ofi_l10_sum_over_depth_1000ms",
        "ofi_l10_accel_500_minus_1000ms",
        "micro_l10_minus_mid_bps",
        "vamp_l10_minus_mid_bps",
        "bid_depth_within_10bps",
        "ask_depth_within_10bps",
        "depth_imbalance_within_10bps",
        "vwap_vs_mid_bps_1000ms",
        "signed_trade_premium_bps_volume_weighted_1000ms",
        "signed_notional_flow_usd_1000ms",
        "trade_imbalance_notional_1000ms",
        "max_signed_trade_notional_usd_1000ms",
        "top5_trade_notional_sum_usd_1000ms",
        "buy_flow_without_price_up_1000ms",
        "sell_flow_without_price_down_1000ms",
        "absorption_bid_1000ms",
        "absorption_ask_1000ms",
        "return_std_bps_1000ms",
        "regime_realized_vol_bps_1000ms",
        "spread_z_1000ms",
        "depth_imbalance_5bps_slope_1000ms",
        "ofi_l1_pressure_ewma_1000ms",
    }

    for name in retained:
        assert name in names, name


def test_pruned_feature_vector_matches_names() -> None:
    fe = FeatureEngine()
    result = fe.on_fast_event(deep_snapshot_ob(1_700_000_700_000, n_levels=60))
    assert result.event_type == "ob"
    assert result.features.shape == (235,)
    assert len(fe.feature_names()) == 235


def test_no_empty_feature_family_scaffolding_remains() -> None:
    import CMSSL17

    forbidden_attrs = [
        "INTERACTION_WINDOWS_MS",
        "TRADE_BURST_WINDOWS_MS",
        "LARGE_TRADE_CONTINUATION_WINDOWS_MS",
        "MACD_TRIPLETS_MS",
        "VPIN_BUCKET_SECS",
        "BOOK_SHAPE_BANDS",
        "SLIPPAGE_NOTIONAL_USD",
        "LARGE_TRADE_NOTIONAL_USD",
        "TRADE_IMBALANCE_DIFF_PAIRS_MS",
        "NET_FLOW_DIFF_PAIRS_MS",
        "OFI_IMBALANCE_DIFF_PAIRS_MS",
    ]

    for attr in forbidden_attrs:
        assert not hasattr(CMSSL17, attr), attr


def test_hot_path_pruned_feature_count_still_unchanged() -> None:
    fe = FeatureEngine()
    names = list(fe.feature_names())
    assert len(names) == 235
    assert fe.core_feature_dim() == 235
    assert fe.feature_dim() == 241

    result = fe.on_fast_event(deep_snapshot_ob(1_700_000_800_000, n_levels=60))
    assert result.event_type == "ob"
    assert result.is_decision is True
    assert result.features.shape == (235,)


def test_removed_hot_path_scaffolding_strings_absent() -> None:
    from pathlib import Path

    text = Path("CMSSL17.py").read_text()

    forbidden = [
        "band_depth_stats",
        "shape_stats",
        "slippage_by_notional",
        "slippage_asymmetry_features",
        "vpin_features",
        "macd_features",
        "rolling_obi_stats",
        "state.slope()",
        "state.persistence(",
        "trade_sign_history",
        "trade_burst_states",
        "vpin_state",
        "macd_state",
        "threshold_counts",
        "large_buy_count_ge_",
        "large_sell_count_ge_",
        "large_trade_imbalance_ge_",
    ]

    for token in forbidden:
        assert token not in text, token


def test_large_trade_and_cvd_windows_are_flow_only() -> None:
    import CMSSL17

    fe = FeatureEngine()
    assert tuple(fe.large_trade_windows) == tuple(CMSSL17.FLOW_WINDOWS_MS)
    assert tuple(fe.cvd_windows) == tuple(CMSSL17.FLOW_WINDOWS_MS)
    assert set(fe.large_trade_states.keys()) == set(CMSSL17.FLOW_WINDOWS_MS)
    assert set(fe.cvd_window_states.keys()) == set(CMSSL17.FLOW_WINDOWS_MS)
    assert 3000 not in fe.large_trade_states
    assert 3000 not in fe.cvd_window_states

def main() -> None:
    fe = FeatureEngine()

    ob_event = (
        "ob",
        1_700_000_000_000,
        1,
        1,
        ((100.0, 2.0), (99.5, 1.5)),
        ((101.0, 2.5), (101.5, 1.0)),
    )
    result = fe.on_fast_event(ob_event)
    assert isinstance(result, FeatureEventResult)
    assert result.event_type == "ob"
    assert result.is_decision is True
    assert isinstance(result.ts_ms, int)
    assert isinstance(result.dt_ms, float)
    assert isinstance(result.raw_mid, float)
    assert isinstance(result.features, np.ndarray)

    trade_event = (
        "trade",
        1_700_000_000_010,
        2,
        100.5,
        0.25,
        1,
        0,
        0,
    )
    result = fe.on_fast_event(trade_event)
    assert isinstance(result, FeatureEventResult)
    assert result.event_type == "trade"
    assert result.is_decision is False
    assert isinstance(result.features, np.ndarray)
    assert_not_tuple_unpackable(result)

    test_trade_does_not_pollute_ob_feature_state()
    test_snapshot_stores_full_book_but_features_use_top_depth()
    test_deeper_snapshot_levels_promote_after_top_level_deletes()
    test_deeper_ask_snapshot_levels_promote_after_top_level_deletes()
    test_book_health_validation_rejects_bad_snapshots()
    test_malformed_delta_levels_are_ignored_but_valid_updates_apply()
    test_compact_ob_type_code_does_not_default_to_delta()
    test_generic_dict_ob_type_parsing_is_explicit()
    test_feature_transform_contract_is_raw_no_projection()
    test_offline_ingest_raw_feature_dims()
    test_merge_event_time_trade_wins_exact_timestamp_tie()
    test_merge_event_time_lower_timestamp_still_wins()
    test_merge_event_time_ob_lower_timestamp_still_wins()
    test_duplicate_ob_timestamps_append_distinct_rows()
    test_offline_ingest_no_overwrite_duplicate_timestamp_api()
    test_pruned_feature_schema_contract()
    test_pruned_feature_vector_matches_names()
    test_no_empty_feature_family_scaffolding_remains()
    test_hot_path_pruned_feature_count_still_unchanged()
    test_removed_hot_path_scaffolding_strings_absent()
    test_large_trade_and_cvd_windows_are_flow_only()


if __name__ == "__main__":
    main()
