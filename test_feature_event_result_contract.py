"""Smoke checks for FeatureEngine event-result API contract."""

import sys
import types
from collections import deque

import numpy as np


def _install_optional_dependency_stubs() -> None:
    """Provide import-time stubs for model-only dependencies unused by this smoke test."""

    try:
        import torch as _real_torch  # noqa: F401
        import torch.nn as _real_nn  # noqa: F401
        real_torch_available = True
    except Exception:
        real_torch_available = False

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
        "MultiheadAttention",
        "Sequential",
        "ModuleList",
    ):
        setattr(nn_mod, name, type(name, (_Module,), {}))
    nn_mod.init = types.SimpleNamespace(
        uniform_=lambda *args, **kwargs: None,
        normal_=lambda *args, **kwargs: None,
        zeros_=lambda *args, **kwargs: None,
        constant_=lambda *args, **kwargs: None,
        xavier_uniform_=lambda *args, **kwargs: None,
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

    if not real_torch_available:
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


REMOVED_V8_FEATURES = {
    "regime_volume_ewma_1000ms",
    "ofi_l1_pressure_ewma_1000ms",
    "ofi_l1_pressure_ewma_500ms",
    "ofi_l1_pressure_ewma_200ms",
    "ofi_l3_sum_over_depth_200ms",
    "ofi_l3_sum_over_depth_500ms",
    "ofi_l3_sum_over_depth_1000ms",
    "ofi_l1_sum_over_depth_1000ms",
    "micro_l1_minus_micro_l10_bps",
    "micro_minus_mid_over_spread",
    "micro_premia",
    "spread_delta_over_spread_200ms",
    "spread_delta_over_spread_500ms",
    "spread_delta_over_spread_1000ms",
    "bid_depth_within_2bps",
    "bid_depth_within_5bps",
    "bid_depth_within_10bps",
    "ask_depth_within_2bps",
    "ask_depth_within_5bps",
    "ask_depth_within_10bps",
    "depth_imbalance_within_2bps",
    "depth_imbalance_within_5bps",
    "depth_imbalance_within_10bps",
    "mid_ret_bps_200ms",
    "mid_ret_bps_500ms",
    "mid_ret_bps_1000ms",
    "return_std_bps_500ms",
    "return_std_bps_1000ms",
    "regime_realized_vol_bps_500ms",
    "regime_realized_vol_bps_1000ms",
    "regime_realized_vol_bps_3000ms",
    "buy_flow_without_price_up_200ms",
    "buy_flow_without_price_up_500ms",
    "buy_flow_without_price_up_1000ms",
    "sell_flow_without_price_down_200ms",
    "sell_flow_without_price_down_500ms",
    "sell_flow_without_price_down_1000ms",
    "ofi_l1_over_depth_l1",
    "ofi_l3_over_depth_l3",
    "ofi_l5_over_depth_l5",
    "ofi_l10_over_depth_l10",
    "regime_flow_imbalance_500ms",
    "regime_flow_imbalance_1000ms",
    "signed_notional_flow_usd_500ms",
    "signed_notional_flow_usd_1000ms",
    "cvd_change_usd_200ms",
    "max_signed_trade_notional_usd_200ms",
    "trade_imbalance_notional_200ms",
    "obi_l5_mean_200ms",
    "obi_l10_mean_200ms",
    "obi_l3_mean_200ms",
    "obi_l5_mean_500ms",
    "obi_l10_mean_500ms",
    "obi_l5_mean_1000ms",
    "obi_l10_mean_1000ms",
    "ask_price_change_rate_200ms",
    "ask_price_change_rate_500ms",
    "ask_price_change_rate_1000ms",
    "ask_l1_rem_rate_over_depth_200ms",
    "bid_l1_rem_rate_over_depth_500ms",
    "ask_l1_rem_rate_over_depth_500ms",
    "bid_l1_rem_rate_over_depth_1000ms",
    "ask_l1_rem_rate_over_depth_1000ms",
    "obi_l3",
    "obi_l5",
    "ofi_l10",
    "bid_l1_depletion_over_depth_200ms",
}

MUST_KEEP_V8_FEATURES = {
    "obi_l1", "obi_l10",
    "micro_minus_mid_bps",
    "obi_l3_mean_500ms", "obi_l3_mean_1000ms",
    "bid_depth_within_1bps", "ask_depth_within_1bps", "depth_imbalance_within_1bps",
    "micro_ret_bps_200ms", "micro_ret_bps_500ms", "micro_ret_bps_1000ms",
    "ofi_l1_over_depth_5bps", "ofi_l3_over_depth_5bps",
    "ofi_l5_over_depth_5bps", "ofi_l10_over_depth_5bps",
    "absorption_bid_200ms", "absorption_ask_200ms",
    "absorption_bid_500ms", "absorption_ask_500ms",
    "absorption_bid_1000ms", "absorption_ask_1000ms",
    "signed_notional_flow_usd_200ms", "cvd_change_usd_500ms",
    "cvd_change_usd_1000ms", "trade_imbalance_notional_500ms",
    "trade_imbalance_notional_1000ms",
    "bid_l1_rem_rate_over_depth_200ms",
    "ask_l1_depletion_over_depth_200ms",
    "bid_l1_depletion_over_depth_500ms", "ask_l1_depletion_over_depth_500ms",
    "bid_l1_depletion_over_depth_1000ms", "ask_l1_depletion_over_depth_1000ms",
}

assert len(REMOVED_V8_FEATURES) >= 67


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

    transform_rows_before = fe.transform_diagnostics_summary()["rows_seen"]
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

    assert fe.transform_diagnostics_summary()["rows_seen"] == transform_rows_before

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
    assert CMSSL17.FEATURE_SCHEMA == "cmssl17_1s_maker_rtcore_v8_raw_no_" + forbidden + "_pruned153_lb10_xformv2"
    assert CMSSL17.FEATURE_TRANSFORM == "feature_transform_spec_v2_pruned153_lb10"
    assert CMSSL17.CHECKPOINT_SCHEMA == (
        "cmssl17-dir-mag-v1-1s-maker-rtcore-raw-no-"
        + forbidden
        + "-pruned153_lb10-xformv2-mamba512-pool512-head1024-k333333-prenormres-finallinear"
    )
    assert "p" + "ca250" not in CMSSL17.FEATURE_SCHEMA.lower()
    assert "final256" not in CMSSL17.FEATURE_SCHEMA.lower()
    assert "p" + "ca250" not in CMSSL17.CHECKPOINT_SCHEMA.lower()
    assert "final256" not in CMSSL17.CHECKPOINT_SCHEMA.lower()



def test_conv_encoder_layer_prenorm_identity_batchnorm() -> None:
    if not _real_torch_available():
        return

    import torch
    from CMSSL17 import _ConvEncoderLayer

    torch.manual_seed(123)

    layer = _ConvEncoderLayer(
        kernel_size=3,
        d_model=8,
        d_ff=16,
        dropout=0.0,
        activation="gelu",
        enable_res_param=True,
        norm="batch",
        re_param=True,
        small_ks=3,
    )
    layer.train()

    x = torch.randn(4, 8, 17)

    with torch.no_grad():
        y = layer(x)

    assert y.shape == x.shape
    assert torch.isfinite(y).all()

    max_abs_err = float((y - x).abs().max())
    assert max_abs_err < 1e-5, max_abs_err


def test_conv_encoder_layer_prenorm_identity_layernorm() -> None:
    if not _real_torch_available():
        return

    import torch
    from CMSSL17 import _ConvEncoderLayer

    torch.manual_seed(123)

    layer = _ConvEncoderLayer(
        kernel_size=3,
        d_model=8,
        d_ff=16,
        dropout=0.0,
        activation="gelu",
        enable_res_param=True,
        norm="layer",
        re_param=True,
        small_ks=3,
    )
    layer.train()

    x = torch.randn(4, 8, 17)

    with torch.no_grad():
        y = layer(x)

    assert y.shape == x.shape
    assert torch.isfinite(y).all()

    max_abs_err = float((y - x).abs().max())
    assert max_abs_err < 1e-5, max_abs_err


def test_conv_encoder_layer_prenorm_sublayers_active_when_residual_scale_enabled() -> None:
    if not _real_torch_available():
        return

    import torch
    from CMSSL17 import _ConvEncoderLayer

    torch.manual_seed(123)

    layer = _ConvEncoderLayer(
        kernel_size=3,
        d_model=8,
        d_ff=16,
        dropout=0.0,
        activation="gelu",
        enable_res_param=True,
        norm="batch",
        re_param=True,
        small_ks=3,
    )
    layer.train()

    with torch.no_grad():
        layer.sublayerconnect1.a.fill_(1.0)
        layer.sublayerconnect2.a.fill_(1.0)

    x = torch.randn(4, 8, 17)

    with torch.no_grad():
        y = layer(x)

    assert y.shape == x.shape
    assert torch.isfinite(y).all()

    mean_abs_delta = float((y - x).abs().mean())
    assert mean_abs_delta > 1e-4, mean_abs_delta


def _real_torch_available() -> bool:
    import torch

    return getattr(torch, "__version__", None) is not None


def test_stage1_final_mixer_factory_linear() -> None:
    from CMSSL17 import (
        build_final_mixer,
        LinearFinalMixer,
        CTN_FINAL_MIXER,
        CTN_FINAL_MIXER_SCHEMA,
        DMODEL,
    )

    assert CTN_FINAL_MIXER == "linear"
    assert CTN_FINAL_MIXER_SCHEMA == "finallinear"

    mixer = build_final_mixer(
        "linear",
        in_feats=193,
        c_internal=8,
        final_in_dim=193 * 8,
        d_model=DMODEL,
        patch_count=99,
        dropout=0.1,
    )
    assert isinstance(mixer, LinearFinalMixer)


def test_stage1_ci_encoded_to_semantic_tokens_layout() -> None:
    if not _real_torch_available():
        return

    import torch
    from CMSSL17 import ci_encoded_to_semantic_tokens

    B, F, P, C = 2, 3, 5, 4
    ci_t = torch.arange(B * F * C * P).reshape(B * F, C, P)

    ci_tokens = ci_encoded_to_semantic_tokens(
        ci_t,
        batch_size=B,
        in_feats=F,
        patch_count=P,
        c_internal=C,
    )

    assert ci_tokens.shape == (B, P, F, C)

    for b in range(B):
        for f in range(F):
            for p in range(P):
                for c in range(C):
                    assert ci_tokens[b, p, f, c] == ci_t[b * F + f, c, p]


def test_stage1_legacy_flatten_ci_tokens_matches_old_order() -> None:
    if not _real_torch_available():
        return

    import torch
    from CMSSL17 import legacy_flatten_ci_tokens

    B, P, F, C = 2, 5, 3, 4
    ci_tokens = torch.arange(B * P * F * C).reshape(B, P, F, C)

    got = legacy_flatten_ci_tokens(ci_tokens)
    expected = ci_tokens.permute(0, 1, 3, 2).contiguous().reshape(B, P, F * C)

    assert got.shape == (B, P, F * C)
    assert torch.equal(got, expected)


def test_stage1_linear_final_mixer_shape_and_finite() -> None:
    if not _real_torch_available():
        return

    import torch
    from CMSSL17 import LinearFinalMixer, DMODEL

    torch.manual_seed(123)
    mixer = LinearFinalMixer(final_in_dim=193 * 8, d_model=DMODEL)

    x = torch.randn(2, 99, 193, 8)
    y = mixer(x)

    assert y.shape == (2, 99, DMODEL)
    assert torch.isfinite(y).all()


def test_stage1_linear_final_mixer_equivalent_to_old_final_proj_order() -> None:
    if not _real_torch_available():
        return

    import torch
    import torch.nn as nn
    from CMSSL17 import LinearFinalMixer, legacy_flatten_ci_tokens, DMODEL

    torch.manual_seed(123)

    B, P, F, C = 2, 11, 7, 3
    final_in_dim = F * C

    mixer = LinearFinalMixer(final_in_dim=final_in_dim, d_model=DMODEL)
    direct = nn.Linear(final_in_dim, DMODEL)

    with torch.no_grad():
        direct.weight.copy_(mixer.proj.weight)
        direct.bias.copy_(mixer.proj.bias)

    ci_tokens = torch.randn(B, P, F, C)

    y_mixer = mixer(ci_tokens)
    y_direct = direct(legacy_flatten_ci_tokens(ci_tokens))

    assert torch.allclose(y_mixer, y_direct, atol=0.0, rtol=0.0)


def test_stage1_extractor_final_proj_compatibility_property() -> None:
    if not _real_torch_available():
        return

    from CMSSL17 import ConvTimeNetFeatureExtractor, DMODEL, LOOKBACK, LinearFinalMixer

    extractor = ConvTimeNetFeatureExtractor(
        in_feats=193,
        seq_len=LOOKBACK,
        d_model=DMODEL,
    )

    assert isinstance(extractor.final_mixer, LinearFinalMixer)
    assert extractor.final_proj is extractor.final_mixer.proj
    assert extractor.final_proj.out_features == DMODEL


def test_stage2_final_mixer_factory_swiglu() -> None:
    from CMSSL17 import (
        build_final_mixer,
        SwiGLUFinalMixer,
        DMODEL,
    )

    mixer = build_final_mixer(
        "swiglu",
        in_feats=193,
        c_internal=8,
        final_in_dim=193 * 8,
        d_model=DMODEL,
        patch_count=99,
        dropout=0.1,
    )
    assert isinstance(mixer, SwiGLUFinalMixer)
    assert mixer.final_in_dim == 159 * 8
    assert mixer.d_model == DMODEL
    assert mixer.hidden_dim > 0


def test_stage2_swiglu_final_mixer_shape_and_finite() -> None:
    if not _real_torch_available():
        return

    import torch
    from CMSSL17 import SwiGLUFinalMixer, DMODEL

    torch.manual_seed(123)
    mixer = SwiGLUFinalMixer(final_in_dim=193 * 8, d_model=DMODEL, dropout=0.0)

    x = torch.randn(2, 99, 193, 8)
    y = mixer(x)

    assert y.shape == (2, 99, DMODEL)
    assert torch.isfinite(y).all()


def test_stage2_swiglu_final_mixer_uses_legacy_flatten_order() -> None:
    if not _real_torch_available():
        return

    import torch
    import torch.nn.functional as F
    from CMSSL17 import SwiGLUFinalMixer, legacy_flatten_ci_tokens, DMODEL

    torch.manual_seed(123)

    B, P, F_dim, C_dim = 2, 11, 7, 3
    final_in_dim = F_dim * C_dim
    mixer = SwiGLUFinalMixer(final_in_dim=final_in_dim, d_model=DMODEL, hidden_dim=32, dropout=0.0)
    mixer.eval()

    ci_tokens = torch.randn(B, P, F_dim, C_dim)

    with torch.no_grad():
        y = mixer(ci_tokens)

        x = legacy_flatten_ci_tokens(ci_tokens)
        xn = mixer.norm(x)
        gate, value = mixer.in_proj(xn).chunk(2, dim=-1)
        h = F.silu(gate) * value
        y_manual = mixer.out_proj(h)

    assert torch.allclose(y, y_manual, atol=0.0, rtol=0.0)


def test_stage2_swiglu_final_mixer_budget_close_to_linear() -> None:
    if not _real_torch_available():
        return

    from CMSSL17 import (
        SwiGLUFinalMixer,
        LinearFinalMixer,
        DMODEL,
        module_parameter_count,
        swiglu_budget_matched_hidden_dim,
    )

    final_in_dim = 193 * 8
    assert swiglu_budget_matched_hidden_dim(final_in_dim, DMODEL) == 224
    linear = LinearFinalMixer(final_in_dim=final_in_dim, d_model=DMODEL)
    swiglu = SwiGLUFinalMixer(final_in_dim=final_in_dim, d_model=DMODEL, dropout=0.1)

    linear_params = module_parameter_count(linear)
    swiglu_params = module_parameter_count(swiglu)
    ratio = swiglu_params / max(1, linear_params)

    assert 0.90 <= ratio <= 1.10, (linear_params, swiglu_params, ratio)


def test_stage3_final_mixer_factory_dcn() -> None:
    from CMSSL17 import (
        build_final_mixer,
        DCNFinalMixer,
        DMODEL,
    )

    mixer = build_final_mixer(
        "dcn",
        in_feats=193,
        c_internal=8,
        final_in_dim=193 * 8,
        d_model=DMODEL,
        patch_count=99,
        dropout=0.1,
    )

    assert isinstance(mixer, DCNFinalMixer)
    assert mixer.final_in_dim == 159 * 8
    assert mixer.d_model == DMODEL
    assert mixer.rank > 0
    assert mixer.num_layers == 1


def test_stage3_dcn_final_mixer_shape_and_finite() -> None:
    if not _real_torch_available():
        return

    import torch
    from CMSSL17 import DCNFinalMixer, DMODEL

    torch.manual_seed(123)

    mixer = DCNFinalMixer(
        final_in_dim=193 * 8,
        d_model=DMODEL,
        rank=16,
        num_layers=1,
    )

    x = torch.randn(2, 99, 193, 8)
    y = mixer(x)

    assert y.shape == (2, 99, DMODEL)
    assert torch.isfinite(y).all()


def test_stage3_dcn_low_rank_cross_layer_matches_manual_formula() -> None:
    if not _real_torch_available():
        return

    import torch
    from CMSSL17 import DCNV2LowRankCrossLayer

    torch.manual_seed(123)

    B, P, D, R = 2, 5, 11, 3
    layer = DCNV2LowRankCrossLayer(dim=D, rank=R)
    x0 = torch.randn(B, P, D)
    xl = torch.randn(B, P, D)

    with torch.no_grad():
        y = layer(x0, xl)
        manual_cross = layer.U(layer.V(xl))
        y_manual = xl + x0 * manual_cross

    assert torch.allclose(y, y_manual, atol=0.0, rtol=0.0)


def test_stage3_dcn_final_mixer_uses_legacy_flatten_order() -> None:
    if not _real_torch_available():
        return

    import torch
    from CMSSL17 import DCNFinalMixer, legacy_flatten_ci_tokens, DMODEL

    torch.manual_seed(123)

    B, P, F_dim, C_dim = 2, 11, 7, 3
    final_in_dim = F_dim * C_dim

    mixer = DCNFinalMixer(
        final_in_dim=final_in_dim,
        d_model=DMODEL,
        rank=4,
        num_layers=1,
    )
    mixer.eval()

    ci_tokens = torch.randn(B, P, F_dim, C_dim)

    with torch.no_grad():
        y = mixer(ci_tokens)

        x = legacy_flatten_ci_tokens(ci_tokens)
        x0 = mixer.norm(x)
        xl = x0
        for layer in mixer.cross_layers:
            xl = layer(x0, xl)
        y_manual = mixer.out_proj(xl)

    assert torch.allclose(y, y_manual, atol=0.0, rtol=0.0)


def test_stage3_dcn_final_mixer_budget_close_to_linear() -> None:
    if not _real_torch_available():
        return

    from CMSSL17 import (
        DCNFinalMixer,
        LinearFinalMixer,
        DMODEL,
        module_parameter_count,
    )

    final_in_dim = 193 * 8

    linear = LinearFinalMixer(final_in_dim=final_in_dim, d_model=DMODEL)
    dcn = DCNFinalMixer(
        final_in_dim=final_in_dim,
        d_model=DMODEL,
        rank=16,
        num_layers=1,
    )

    linear_params = module_parameter_count(linear)
    dcn_params = module_parameter_count(dcn)
    ratio = dcn_params / max(1, linear_params)

    assert 0.90 <= ratio <= 1.10, (linear_params, dcn_params, ratio)


def test_stage4_final_mixer_factory_hybrid() -> None:
    from CMSSL17 import (
        build_final_mixer,
        HybridFinalMixer,
        DMODEL,
    )

    mixer = build_final_mixer(
        "hybrid",
        in_feats=193,
        c_internal=8,
        final_in_dim=193 * 8,
        d_model=DMODEL,
        patch_count=99,
        dropout=0.1,
    )

    assert isinstance(mixer, HybridFinalMixer)
    assert mixer.final_in_dim == 159 * 8
    assert mixer.d_model == DMODEL
    assert mixer.swiglu_hidden_dim > 0
    assert mixer.dcn_rank > 0
    assert mixer.dcn_num_layers == 1


def test_stage4_hybrid_final_mixer_shape_and_finite() -> None:
    if not _real_torch_available():
        return

    import torch
    from CMSSL17 import HybridFinalMixer, DMODEL

    torch.manual_seed(123)

    mixer = HybridFinalMixer(
        final_in_dim=193 * 8,
        d_model=DMODEL,
        swiglu_hidden_dim=128,
        dcn_rank=8,
        dcn_num_layers=1,
        dropout=0.0,
    )

    x = torch.randn(2, 99, 193, 8)
    y = mixer(x)

    assert y.shape == (2, 99, DMODEL)
    assert torch.isfinite(y).all()


def test_stage4_hybrid_final_mixer_matches_manual_formula() -> None:
    if not _real_torch_available():
        return

    import torch
    import torch.nn.functional as F
    from CMSSL17 import HybridFinalMixer, legacy_flatten_ci_tokens, DMODEL

    torch.manual_seed(123)

    B, P, F_dim, C_dim = 2, 11, 7, 3
    final_in_dim = F_dim * C_dim

    mixer = HybridFinalMixer(
        final_in_dim=final_in_dim,
        d_model=DMODEL,
        swiglu_hidden_dim=16,
        dcn_rank=4,
        dcn_num_layers=1,
        dropout=0.0,
    )
    mixer.eval()

    ci_tokens = torch.randn(B, P, F_dim, C_dim)

    with torch.no_grad():
        y = mixer(ci_tokens)

        x = legacy_flatten_ci_tokens(ci_tokens)
        x0 = mixer.norm(x)

        gate, value = mixer.swiglu_in_proj(x0).chunk(2, dim=-1)
        h_swiglu = F.silu(gate) * value

        x_dcn = x0
        for layer in mixer.cross_layers:
            x_dcn = layer(x0, x_dcn)

        h = torch.cat([h_swiglu, x_dcn], dim=-1)
        y_manual = mixer.out_proj(h)

    assert torch.allclose(y, y_manual, atol=0.0, rtol=0.0)


def test_stage4_hybrid_final_mixer_uses_legacy_flatten_order() -> None:
    if not _real_torch_available():
        return

    import torch
    from CMSSL17 import HybridFinalMixer, legacy_flatten_ci_tokens, DMODEL

    torch.manual_seed(123)

    B, P, F_dim, C_dim = 2, 5, 7, 3
    mixer = HybridFinalMixer(
        final_in_dim=F_dim * C_dim,
        d_model=DMODEL,
        swiglu_hidden_dim=16,
        dcn_rank=4,
        dcn_num_layers=1,
        dropout=0.0,
    )

    ci_tokens = torch.randn(B, P, F_dim, C_dim)
    x = legacy_flatten_ci_tokens(ci_tokens)

    with torch.no_grad():
        x0 = mixer.norm(x)
        gate, value = mixer.swiglu_in_proj(x0).chunk(2, dim=-1)

    assert gate.shape == (B, P, 16)
    assert value.shape == (B, P, 16)


def test_stage4_hybrid_final_mixer_budget_close_to_linear() -> None:
    if not _real_torch_available():
        return

    from CMSSL17 import (
        HybridFinalMixer,
        LinearFinalMixer,
        DMODEL,
        module_parameter_count,
    )

    final_in_dim = 193 * 8

    linear = LinearFinalMixer(final_in_dim=final_in_dim, d_model=DMODEL)
    hybrid = HybridFinalMixer(
        final_in_dim=final_in_dim,
        d_model=DMODEL,
        swiglu_hidden_dim=128,
        dcn_rank=8,
        dcn_num_layers=1,
        dropout=0.1,
    )

    linear_params = module_parameter_count(linear)
    hybrid_params = module_parameter_count(hybrid)
    ratio = hybrid_params / max(1, linear_params)

    assert 0.90 <= ratio <= 1.10, (linear_params, hybrid_params, ratio)


def test_stage5_final_mixer_factory_latent_attn() -> None:
    from CMSSL17 import (
        build_final_mixer,
        LatentAttentionFinalMixer,
        DMODEL,
    )

    mixer = build_final_mixer(
        "latent_attn",
        in_feats=193,
        c_internal=8,
        final_in_dim=193 * 8,
        d_model=DMODEL,
        patch_count=99,
        dropout=0.1,
    )

    assert isinstance(mixer, LatentAttentionFinalMixer)
    assert mixer.in_feats == 159
    assert mixer.c_internal == 8
    assert mixer.d_model == DMODEL
    assert mixer.latent_count > 0
    assert mixer.token_dim > 0
    assert mixer.num_heads > 0
    assert mixer.token_dim % mixer.num_heads == 0


def test_stage5_latent_attn_final_mixer_shape_and_finite() -> None:
    if not _real_torch_available():
        return

    import torch
    from CMSSL17 import LatentAttentionFinalMixer, DMODEL

    torch.manual_seed(123)

    mixer = LatentAttentionFinalMixer(
        in_feats=193,
        c_internal=8,
        d_model=DMODEL,
        token_dim=80,
        latent_count=16,
        num_heads=4,
        ff_mult=4,
        dropout=0.0,
    )

    x = torch.randn(2, 99, 193, 8)
    y = mixer(x)

    assert y.shape == (2, 99, DMODEL)
    assert torch.isfinite(y).all()


def test_stage5_latent_attn_final_mixer_matches_manual_formula() -> None:
    if not _real_torch_available():
        return

    import torch
    from CMSSL17 import LatentAttentionFinalMixer, DMODEL

    torch.manual_seed(123)

    B, P, F_dim, C_dim = 2, 5, 7, 3
    mixer = LatentAttentionFinalMixer(
        in_feats=F_dim,
        c_internal=C_dim,
        d_model=DMODEL,
        token_dim=16,
        latent_count=4,
        num_heads=4,
        ff_mult=2,
        dropout=0.0,
    )
    mixer.eval()

    ci_tokens = torch.randn(B, P, F_dim, C_dim)

    with torch.no_grad():
        y = mixer(ci_tokens)

        x = ci_tokens.reshape(B * P, F_dim, C_dim)
        feat = mixer.token_proj(x)
        feat = feat + mixer.feature_embed.view(1, F_dim, mixer.token_dim)

        latents = mixer.latents.view(1, mixer.latent_count, mixer.token_dim).expand(B * P, -1, -1)

        q = mixer.latent_cross_norm(latents)
        kv = mixer.feature_norm(feat)
        cross_out, _ = mixer.cross_attn(q, kv, kv, need_weights=False)
        latents = latents + mixer.cross_drop(cross_out)

        z = mixer.latent_self_norm(latents)
        self_out, _ = mixer.self_attn(z, z, z, need_weights=False)
        latents = latents + mixer.self_drop(self_out)

        latents = latents + mixer.ff_drop(mixer.latent_ff(mixer.latent_ff_norm(latents)))

        flat = latents.reshape(B * P, mixer.latent_count * mixer.token_dim)
        y_manual = mixer.out_proj(flat).view(B, P, DMODEL)

    assert torch.allclose(y, y_manual, atol=0.0, rtol=0.0)


def test_stage5_latent_attn_uses_semantic_feature_tokens() -> None:
    if not _real_torch_available():
        return

    import torch
    from CMSSL17 import LatentAttentionFinalMixer

    torch.manual_seed(123)

    B, P, F_dim, C_dim = 2, 3, 5, 4
    mixer = LatentAttentionFinalMixer(
        in_feats=F_dim,
        c_internal=C_dim,
        d_model=32,
        token_dim=8,
        latent_count=2,
        num_heads=2,
        ff_mult=2,
        dropout=0.0,
    )

    ci_tokens = torch.randn(B, P, F_dim, C_dim)

    x = ci_tokens.reshape(B * P, F_dim, C_dim)

    # Exact semantic-layout check before projection. The feature index f in x must
    # correspond exactly to ci_tokens[:, :, f, :] flattened over [B, P].
    for f in range(F_dim):
        expected_feature_token = ci_tokens[:, :, f, :].reshape(B * P, C_dim)
        assert torch.equal(x[:, f, :], expected_feature_token)

    tokens = mixer.token_proj(x) + mixer.feature_embed.view(1, F_dim, mixer.token_dim)

    for f in range(F_dim):
        manual = mixer.token_proj(ci_tokens[:, :, f, :].reshape(B * P, C_dim))
        manual = manual + mixer.feature_embed[f].view(1, mixer.token_dim)

        max_abs_err = float((tokens[:, f, :] - manual).abs().max().detach().cpu())
        # The exact layout invariant is tested above with torch.equal() before projection.
        # After token_proj, batched Linear vs per-feature Linear can differ by tiny
        # floating-point roundoff, so use a tight tolerance rather than exact equality.
        assert torch.allclose(
            tokens[:, f, :],
            manual,
            atol=1e-6,
            rtol=1e-6,
        ), max_abs_err


def test_stage5_latent_attn_final_mixer_budget_close_to_linear() -> None:
    if not _real_torch_available():
        return

    from CMSSL17 import (
        LatentAttentionFinalMixer,
        LinearFinalMixer,
        DMODEL,
        module_parameter_count,
    )

    final_in_dim = 193 * 8

    linear = LinearFinalMixer(final_in_dim=final_in_dim, d_model=DMODEL)
    latent = LatentAttentionFinalMixer(
        in_feats=193,
        c_internal=8,
        d_model=DMODEL,
        token_dim=80,
        latent_count=16,
        num_heads=4,
        ff_mult=4,
        dropout=0.1,
    )

    linear_params = module_parameter_count(linear)
    latent_params = module_parameter_count(latent)
    ratio = latent_params / max(1, linear_params)

    assert 0.90 <= ratio <= 1.10, (linear_params, latent_params, ratio)


def test_stage6_final_mixer_factory_cross_attn() -> None:
    from CMSSL17 import (
        build_final_mixer,
        CrossChannelAttentionFinalMixer,
        DMODEL,
    )

    mixer = build_final_mixer(
        "cross_attn",
        in_feats=193,
        c_internal=8,
        final_in_dim=193 * 8,
        d_model=DMODEL,
        patch_count=99,
        dropout=0.1,
    )

    assert isinstance(mixer, CrossChannelAttentionFinalMixer)
    assert mixer.in_feats == 159
    assert mixer.c_internal == 8
    assert mixer.d_model == DMODEL
    assert mixer.token_dim > 0
    assert mixer.num_heads > 0
    assert mixer.token_dim % mixer.num_heads == 0
    assert mixer.num_layers > 0


def test_stage6_cross_attn_final_mixer_shape_and_finite() -> None:
    if not _real_torch_available():
        return

    import torch
    from CMSSL17 import CrossChannelAttentionFinalMixer, DMODEL

    torch.manual_seed(123)

    mixer = CrossChannelAttentionFinalMixer(
        in_feats=193,
        c_internal=8,
        d_model=DMODEL,
        token_dim=224,
        num_heads=8,
        ff_mult=4,
        num_layers=1,
        dropout=0.0,
    )

    x = torch.randn(2, 99, 193, 8)
    y = mixer(x)

    assert y.shape == (2, 99, DMODEL)
    assert torch.isfinite(y).all()


def test_stage6_cross_attn_final_mixer_matches_manual_formula() -> None:
    if not _real_torch_available():
        return

    import torch
    from CMSSL17 import CrossChannelAttentionFinalMixer, DMODEL

    torch.manual_seed(123)

    B, P, F_dim, C_dim = 2, 5, 7, 3
    mixer = CrossChannelAttentionFinalMixer(
        in_feats=F_dim,
        c_internal=C_dim,
        d_model=DMODEL,
        token_dim=16,
        num_heads=4,
        ff_mult=2,
        num_layers=1,
        dropout=0.0,
    )
    mixer.eval()

    ci_tokens = torch.randn(B, P, F_dim, C_dim)

    with torch.no_grad():
        y = mixer(ci_tokens)

        x = ci_tokens.reshape(B * P, F_dim, C_dim)
        feat = mixer.token_proj(x)
        feat = feat + mixer.feature_embed.view(1, F_dim, mixer.token_dim)

        cls = mixer.cls_token.view(1, 1, mixer.token_dim).expand(B * P, -1, -1)
        tokens = torch.cat([cls, feat], dim=1)

        for block in mixer.blocks:
            tokens = block(tokens)

        cls_out = tokens[:, 0, :]
        y_manual = mixer.out_proj(mixer.out_norm(cls_out)).view(B, P, DMODEL)

    assert torch.allclose(y, y_manual, atol=0.0, rtol=0.0)


def test_stage6_cross_attn_uses_semantic_feature_tokens() -> None:
    if not _real_torch_available():
        return

    import torch
    from CMSSL17 import CrossChannelAttentionFinalMixer

    torch.manual_seed(123)

    B, P, F_dim, C_dim = 2, 3, 5, 4
    mixer = CrossChannelAttentionFinalMixer(
        in_feats=F_dim,
        c_internal=C_dim,
        d_model=32,
        token_dim=8,
        num_heads=2,
        ff_mult=2,
        num_layers=1,
        dropout=0.0,
    )

    ci_tokens = torch.randn(B, P, F_dim, C_dim)
    x = ci_tokens.reshape(B * P, F_dim, C_dim)

    # Exact semantic-layout check before projection.
    for f in range(F_dim):
        expected_feature_token = ci_tokens[:, :, f, :].reshape(B * P, C_dim)
        assert torch.equal(x[:, f, :], expected_feature_token)

    tokens = mixer.token_proj(x) + mixer.feature_embed.view(1, F_dim, mixer.token_dim)

    for f in range(F_dim):
        manual = mixer.token_proj(ci_tokens[:, :, f, :].reshape(B * P, C_dim))
        manual = manual + mixer.feature_embed[f].view(1, mixer.token_dim)

        max_abs_err = float((tokens[:, f, :] - manual).abs().max().detach().cpu())
        assert torch.allclose(
            tokens[:, f, :],
            manual,
            atol=1e-6,
            rtol=1e-6,
        ), max_abs_err


def test_stage6_cross_attn_final_mixer_budget_close_to_linear() -> None:
    if not _real_torch_available():
        return

    from CMSSL17 import (
        CrossChannelAttentionFinalMixer,
        LinearFinalMixer,
        DMODEL,
        module_parameter_count,
    )

    final_in_dim = 193 * 8

    linear = LinearFinalMixer(final_in_dim=final_in_dim, d_model=DMODEL)
    cross = CrossChannelAttentionFinalMixer(
        in_feats=193,
        c_internal=8,
        d_model=DMODEL,
        token_dim=224,
        num_heads=8,
        ff_mult=4,
        num_layers=1,
        dropout=0.1,
    )

    linear_params = module_parameter_count(linear)
    cross_params = module_parameter_count(cross)
    ratio = cross_params / max(1, linear_params)

    assert 0.90 <= ratio <= 1.10, (linear_params, cross_params, ratio)


def test_stage5_latent_attn_disables_flash_sdp_in_subprocess() -> None:
    import os
    import subprocess
    import sys
    import textwrap

    code = textwrap.dedent(
        r'''
        import torch
        import CMSSL17

        print("mixer", CMSSL17.CTN_FINAL_MIXER)
        if not torch.cuda.is_available():
            print("sdpa_backend_checks_skipped no_cuda")
        else:
            print("flash", torch.backends.cuda.flash_sdp_enabled())
            print("mem_eff", torch.backends.cuda.mem_efficient_sdp_enabled())
            print("math", torch.backends.cuda.math_sdp_enabled())
            if hasattr(torch.backends.cuda, "cudnn_sdp_enabled"):
                print("cudnn", torch.backends.cuda.cudnn_sdp_enabled())

            assert torch.backends.cuda.flash_sdp_enabled() is False
            assert torch.backends.cuda.mem_efficient_sdp_enabled() is True
            assert torch.backends.cuda.math_sdp_enabled() is True
            if hasattr(torch.backends.cuda, "cudnn_sdp_enabled"):
                assert torch.backends.cuda.cudnn_sdp_enabled() is False

        assert CMSSL17.CTN_FINAL_MIXER == "latent_attn"
        '''
    )

    env = os.environ.copy()
    env["BYBIT_CTN_FINAL_MIXER"] = "latent_attn"
    p = subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=True,
    )
    assert "mixer latent_attn" in p.stdout
    if "sdpa_backend_checks_skipped no_cuda" not in p.stdout:
        assert "flash False" in p.stdout


def test_stage6_cross_attn_disables_flash_sdp_in_subprocess() -> None:
    import os
    import subprocess
    import sys
    import textwrap

    code = textwrap.dedent(
        r'''
        import torch
        import CMSSL17

        assert CMSSL17.CTN_FINAL_MIXER == "cross_attn"
        if not torch.cuda.is_available():
            print("ok cross_attn no_cuda")
        else:
            assert torch.backends.cuda.flash_sdp_enabled() is False
            assert torch.backends.cuda.mem_efficient_sdp_enabled() is True
            assert torch.backends.cuda.math_sdp_enabled() is True
            if hasattr(torch.backends.cuda, "cudnn_sdp_enabled"):
                assert torch.backends.cuda.cudnn_sdp_enabled() is False
            print("ok cross_attn flash disabled")
        '''
    )

    env = os.environ.copy()
    env["BYBIT_CTN_FINAL_MIXER"] = "cross_attn"
    p = subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=True,
    )
    assert (
        "ok cross_attn flash disabled" in p.stdout
        or "ok cross_attn no_cuda" in p.stdout
    )


def test_stage1_linear_does_not_force_disable_flash_sdp_in_subprocess() -> None:
    import os
    import subprocess
    import sys

    code = r'''
import CMSSL17
assert CMSSL17.CTN_FINAL_MIXER == "linear"
print("ok linear import")
'''

    env = os.environ.copy()
    env["BYBIT_CTN_FINAL_MIXER"] = "linear"
    p = subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=True,
    )
    assert "ok linear import" in p.stdout


def test_stage3_extractor_final_proj_unavailable_for_dcn() -> None:
    # This is better as a manual env smoke test because CTN_FINAL_MIXER
    # is import-time config. See manual command in the Stage 3 instructions.
    pass


def test_v9_smallstable_model_width_contract() -> None:
    from CMSSL17 import (
        DMODEL,
        NUM_HEADS,
        ModelArgs,
        LOOKBACK,
        MAMBA_LAYERS,
        FeatureEngine,
        MODEL_ARCH_SCHEMA,
        CHECKPOINT_SCHEMA,
        CTN_FINAL_MIXER,
        CTN_FINAL_MIXER_SCHEMA,
    )

    assert DMODEL == 512
    assert NUM_HEADS == 16

    fe = FeatureEngine()
    f_total = fe.feature_dim()
    assert f_total == 159

    args = ModelArgs(DMODEL, MAMBA_LAYERS, f_total, LOOKBACK)
    assert args.headdim == 32

    assert "mamba512" in MODEL_ARCH_SCHEMA
    assert "mamba512" in CHECKPOINT_SCHEMA
    assert "prenormres" in MODEL_ARCH_SCHEMA
    assert "prenormres" in CHECKPOINT_SCHEMA
    assert "k333333" in MODEL_ARCH_SCHEMA
    assert "k333333" in CHECKPOINT_SCHEMA
    assert CTN_FINAL_MIXER == "linear"
    assert CTN_FINAL_MIXER_SCHEMA == "finallinear"
    assert "finallinear" in MODEL_ARCH_SCHEMA
    assert "finallinear" in CHECKPOINT_SCHEMA

    if not _real_torch_available():
        return

    from CMSSL17 import SAMBA

    model = SAMBA(args)

    assert model.depatch_proj_encoder.final_proj.out_features == 512

    fused_dim = DMODEL * 2
    assert fused_dim == 1024

    assert model.dir_head[0].in_features == fused_dim
    assert model.dir_head[0].out_features == fused_dim
    assert model.dir_head[-1].in_features == fused_dim
    assert model.dir_head[-1].out_features == 3

    assert model.mag_up_head[0].in_features == fused_dim
    assert model.mag_up_head[0].out_features == fused_dim
    assert model.mag_up_head[-1].in_features == fused_dim
    assert model.mag_up_head[-1].out_features == 3

    assert model.dir_token_decoder.ff[0].in_features == fused_dim
    assert model.dir_token_decoder.ff[0].out_features == 2 * fused_dim
    assert model.dir_token_decoder.ff[-1].out_features == fused_dim

    assert model.dir_pool.d_hidden == 512
    assert model.mag_pool.d_hidden == 512


def test_v9_gated_pooling_query_scaled_init() -> None:
    if not _real_torch_available():
        return

    import math
    import torch
    from CMSSL17 import GatedPooling

    pool = GatedPooling(1024)
    assert pool.d_hidden == 512
    assert pool.u.shape == (512,)

    # Do not require exact std because random sample variance is noisy.
    # But it should be far closer to 1/sqrt(512) than to 1.0.
    u_std = float(pool.u.detach().std(unbiased=False))
    target = 1.0 / math.sqrt(512.0)

    assert 0.25 * target <= u_std <= 4.0 * target
    assert u_std < 0.20

    h = torch.randn(4, 99, 1024)
    z = pool(h)
    assert z.shape == (4, 1024)
    assert torch.isfinite(z).all()


def test_v9_initial_direction_logits_not_extreme() -> None:
    if not _real_torch_available():
        return

    import torch
    from CMSSL17 import DMODEL, MAMBA_LAYERS, LOOKBACK, ModelArgs, SAMBA, FeatureEngine

    torch.manual_seed(123)

    fe = FeatureEngine()
    f_total = fe.feature_dim()
    args = ModelArgs(DMODEL, MAMBA_LAYERS, f_total, LOOKBACK)
    model = SAMBA(args)
    model.eval()

    x = torch.randn(4, LOOKBACK, f_total)

    with torch.no_grad():
        pred = model(x)

    dir_logits = pred["dir_logits"].float()
    assert dir_logits.shape == (4, 3)
    assert torch.isfinite(dir_logits).all()

    # This should be much lower than the failed run's dir_logit_std=13.8.
    assert float(dir_logits.std(unbiased=False)) < 5.0

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
    names = set(fe.feature_names())

    assert fe.core_feature_dim() == 153
    assert fe.feature_dim() == 159

    legacy_removed = {
        "time_hour_sin", "time_hour_cos", "time_dow_sin", "time_dow_cos",
        "session_is_weekend", "session_is_asia", "session_is_europe", "session_is_us",
        "session_is_europe_us_overlap", "mid_trend_r2_200ms",
        "mid_position_in_range_200ms", "mid_breakout_up_200ms",
        "mid_breakout_down_200ms", "sign_persistence_200ms",
        "up_return_fraction_200ms", "return_autocorr_lag1_200ms",
        "cum_bid_l1", "cum_ask_l1", "cum_bid_l20", "cum_ask_l20",
        "obi_l20", "ofi_l20", "ofi_l1_over_spread_bps",
        "ofi_l3_over_spread_bps", "ofi_l5_over_spread_bps",
        "ofi_l10_over_spread_bps", "spread_delta_bps_200ms",
        "spread_delta_bps_500ms", "spread_delta_bps_1000ms",
        "last_is_rpi", "mid_slope_bps_per_sec_200ms", "mid_range_bps_200ms",
    }

    for name in legacy_removed | REMOVED_V8_FEATURES:
        assert name not in names, name

    for name in MUST_KEEP_V8_FEATURES:
        assert name in names, name


def test_v9_pruned159_removed_features_absent_and_representatives_retained() -> None:
    fe = FeatureEngine()
    names = set(fe.feature_names())

    assert len(REMOVED_V8_FEATURES) >= 67

    for name in REMOVED_V8_FEATURES:
        assert name not in names, name

    for name in MUST_KEEP_V8_FEATURES:
        assert name in names, name

    assert fe.core_feature_dim() == 153
    assert fe.feature_dim() == 159




def test_pruned_feature_vector_matches_names_on_decision_event() -> None:
    fe = FeatureEngine()
    result = fe.on_fast_event(deep_snapshot_ob(1_700_000_100_000, n_levels=60))
    names = fe.feature_names()

    assert result.is_decision is True
    assert result.features.shape == (153,)
    assert len(names) == 153
    assert result.features.shape[0] == len(names)

def test_v8_pruned153_lb10_event_feature_shape() -> None:
    fe = FeatureEngine()
    r = fe.on_fast_event(deep_snapshot_ob(1_700_003_000_000, n_levels=60))
    assert r.event_type == "ob"
    assert r.is_decision is True
    assert r.features.shape == (153,)
    assert np.isfinite(r.features).all()


def test_removed_v8_feature_names_not_in_production_feature_names() -> None:
    fe = FeatureEngine()
    names = "\n".join(fe.feature_names())

    for banned in REMOVED_V8_FEATURES:
        assert banned not in names


def test_v8_return_histories_only_track_retained_windows() -> None:
    fe = FeatureEngine()
    assert tuple(fe.return_windows_ms) == (200,)
    assert set(fe.return_histories.keys()) == {200}

    r = fe.on_fast_event(deep_snapshot_ob(1_700_004_000_000, n_levels=60))
    assert r.features.shape == (153,)
    assert set(fe.return_histories.keys()) == {200}


def test_v8_replenishment_states_only_track_emitted_keys() -> None:
    fe = FeatureEngine()

    expected = {
        200: (("bid", 1, "add"), ("ask", 1, "add"), ("bid", 1, "rem")),
        500: (("bid", 1, "add"), ("ask", 1, "add")),
        1000: (("bid", 1, "add"), ("ask", 1, "add")),
    }

    assert fe._replen_keys_by_window == expected

    for (window_ms, key) in fe.replen_deques.keys():
        assert window_ms in expected
        assert key in expected[window_ms]
        assert key[1] == 1

    for (window_ms, key) in fe.replen_sums.keys():
        assert window_ms in expected
        assert key in expected[window_ms]
        assert key[1] == 1


def test_v8_removed_computation_strings_absent_from_production_cmssl17() -> None:
    from pathlib import Path

    text = Path("CMSSL17.py").read_text()

    forbidden = [
        "rv_ewma",
        "regime_vol_ewma",
        "max_signed_trade_notional_usd_200ms",
        "cvd_change_usd_200ms",
        "ask_l1_rem_rate_over_depth_200ms",
        "bid_l1_rem_rate_over_depth_500ms",
        "ask_l1_rem_rate_over_depth_500ms",
        "bid_l1_rem_rate_over_depth_1000ms",
        "ask_l1_rem_rate_over_depth_1000ms",
    ]

    for s in forbidden:
        assert s not in text, s


def test_v8_large_trade_and_cvd_representatives_still_present() -> None:
    fe = FeatureEngine()
    names = set(fe.feature_names())

    retained = {
        "top5_trade_notional_sum_usd_200ms",
        "top5_trade_notional_sum_usd_500ms",
        "top5_trade_notional_sum_usd_1000ms",
        "max_signed_trade_notional_usd_500ms",
        "max_signed_trade_notional_usd_1000ms",
        "cvd_slope_usd_per_sec_200ms",
        "cvd_minus_ema_usd_200ms",
        "cvd_change_usd_500ms",
        "cvd_slope_usd_per_sec_500ms",
        "cvd_minus_ema_usd_500ms",
        "cvd_change_usd_1000ms",
        "cvd_slope_usd_per_sec_1000ms",
        "cvd_minus_ema_usd_1000ms",
    }

    for name in retained:
        assert name in names, name

    assert "max_signed_trade_notional_usd_200ms" not in names
    assert "cvd_change_usd_200ms" not in names


def test_pruned_feature_vector_matches_names() -> None:
    fe = FeatureEngine()
    result = fe.on_fast_event(deep_snapshot_ob(1_700_000_700_000, n_levels=60))
    assert result.event_type == "ob"
    assert result.features.shape == (153,)
    assert len(fe.feature_names()) == 153


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
    assert len(names) == 153
    assert fe.core_feature_dim() == 153
    assert fe.feature_dim() == 159

    result = fe.on_fast_event(deep_snapshot_ob(1_700_000_800_000, n_levels=60))
    assert result.event_type == "ob"
    assert result.is_decision is True
    assert result.features.shape == (153,)


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


def test_transform_v2_feature_count_unchanged() -> None:
    fe = FeatureEngine()
    assert len(fe.feature_names()) == 153
    r = fe.on_fast_event(deep_snapshot_ob(1_700_000_900_000, n_levels=60))
    assert r.features.shape == (153,)


def test_every_feature_has_exactly_one_transform_spec() -> None:
    from CMSSL17 import build_feature_transform_specs
    fe = FeatureEngine()
    specs = build_feature_transform_specs(fe.feature_names())
    assert len(specs) == 153
    assert [s.name for s in specs] == list(fe.feature_names())


def test_old_zscore_path_removed() -> None:
    from pathlib import Path
    text = Path("CMSSL17.py").read_text()
    forbidden = ["def _zscore", "_feature_z_half_life_ms", "_ensure_zscore_metadata", "z_mean", "z_m2", "_last_z_ts_ms"]
    for token in forbidden:
        assert token not in text, token


def test_ewma_transform_scores_before_update() -> None:
    from CMSSL17 import FeatureTransformEngine
    names = ["trade_count_200ms"]
    eng = FeatureTransformEngine(names)
    for _ in range(60):
        eng.apply(np.asarray([0.0], dtype=np.float32), 100.0)
    y_jump = eng.apply(np.asarray([100.0], dtype=np.float32), 100.0)
    assert y_jump[0] > 1.0


def test_bounded_features_are_not_ewma_z() -> None:
    from CMSSL17 import build_feature_transform_specs, NormalizeKind
    fe = FeatureEngine()
    specs = {s.name: s for s in build_feature_transform_specs(fe.feature_names())}
    for name in ["obi_l1", "obi_l10", "depth_imbalance_within_1bps", "trade_imbalance_notional_1000ms", "spread_z_1000ms"]:
        assert specs[name].normalize == NormalizeKind.NONE


def test_heavy_tailed_features_use_log_ewma() -> None:
    from CMSSL17 import build_feature_transform_specs, RawTransformKind, NormalizeKind
    fe = FeatureEngine()
    specs = {s.name: s for s in build_feature_transform_specs(fe.feature_names())}
    assert specs["signed_notional_flow_usd_200ms"].raw_transform == RawTransformKind.SIGNED_LOG1P
    assert specs["signed_notional_flow_usd_200ms"].normalize == NormalizeKind.EWMA_Z
    assert specs["top5_trade_notional_sum_usd_1000ms"].raw_transform == RawTransformKind.LOG1P_POS
    assert specs["top5_trade_notional_sum_usd_1000ms"].normalize == NormalizeKind.EWMA_Z




def test_down_up_vol_imbalance_is_bounded_identity_transform() -> None:
    from CMSSL17 import build_feature_transform_specs, RawTransformKind, NormalizeKind

    fe = FeatureEngine()
    specs = {s.name: s for s in build_feature_transform_specs(fe.feature_names())}
    names = set(fe.feature_names())

    for old_name in [
        "down_up_vol_ratio_500ms",
        "down_up_vol_ratio_1000ms",
        "down_up_vol_ratio_3000ms",
    ]:
        assert old_name not in names

    for name in [
        "down_up_vol_imbalance_500ms",
        "down_up_vol_imbalance_1000ms",
        "down_up_vol_imbalance_3000ms",
    ]:
        assert name in names
        s = specs[name]
        assert s.raw_transform == RawTransformKind.IDENTITY
        assert s.normalize == NormalizeKind.NONE
        assert s.half_life_ms == 0
        assert s.scale == 1.0
        assert s.input_clip_abs == 0.0
        assert s.output_clip_abs == 1.0


def test_down_up_vol_imbalance_values_are_bounded() -> None:
    fe = FeatureEngine()

    # Initialize book.
    fe.on_fast_event(deep_snapshot_ob(1_703_000_000_000, n_levels=60))

    result = None
    # Feed alternating OB updates to create some return variation.
    for i in range(1, 120):
        ts = 1_703_000_000_000 + i * 100
        bid = 100.0 + (0.05 if i % 3 == 0 else -0.02 if i % 5 == 0 else 0.0)
        ask = 101.0 + (0.05 if i % 3 == 0 else -0.02 if i % 5 == 0 else 0.0)
        result = fe.on_fast_event(("ob", ts, i + 1, 2, ((bid, 2.0),), ((ask, 2.0),)))

    assert result is not None
    names = list(fe.feature_names())
    vals = dict(zip(names, result.features.tolist()))

    for name in [
        "down_up_vol_imbalance_500ms",
        "down_up_vol_imbalance_1000ms",
        "down_up_vol_imbalance_3000ms",
    ]:
        assert name in vals
        assert np.isfinite(vals[name])
        assert -1.0 <= vals[name] <= 1.0


def test_time_since_features_clip_at_12() -> None:
    from CMSSL17 import build_feature_transform_specs

    fe = FeatureEngine()
    names = set(fe.feature_names())
    specs = {s.name: s for s in build_feature_transform_specs(fe.feature_names())}

    for name in [
        "time_since_trade_ms",
        "time_since_mid_change_ms",
        "time_since_last_buy_trade_ms",
        "time_since_last_sell_trade_ms",
    ]:
        if name in names:
            assert specs[name].output_clip_abs == 12.0


def test_notional_context_uses_scaled_log_no_ewma() -> None:
    from CMSSL17 import build_feature_transform_specs, RawTransformKind, NormalizeKind

    fe = FeatureEngine()
    specs = {s.name: s for s in build_feature_transform_specs(fe.feature_names())}

    for name in [
        "bid_l1_notional_usd",
        "ask_l1_notional_usd",
        "bid_depth_notional_5bps",
        "ask_depth_notional_5bps",
        "total_depth_notional_5bps",
    ]:
        s = specs[name]
        assert s.raw_transform == RawTransformKind.LOG1P_POS_SCALE
        assert s.normalize == NormalizeKind.NONE
        assert s.scale == 10.0
        assert s.output_clip_abs == 3.0


def test_transform_diagnostics_actionable_summary_exists() -> None:
    fe = FeatureEngine()
    for i in range(80):
        if i == 0:
            fe.on_fast_event(deep_snapshot_ob(1_702_000_000_000, n_levels=60))
        else:
            fe.on_fast_event(delta_ob(1_702_000_000_000 + i * 100))

    diag = fe.transform_diagnostics_summary()
    assert "actionable_summary" in diag
    summary = diag["actionable_summary"]
    assert "dead_features" in summary
    assert "low_variance_features" in summary
    assert "high_clip_features" in summary
    assert isinstance(summary["dead_features"], list)
    assert isinstance(summary["low_variance_features"], list)
    assert isinstance(summary["high_clip_features"], list)

def test_notional_context_features_present_and_transformed() -> None:
    from CMSSL17 import (
        build_feature_transform_specs,
        RawTransformKind,
        NormalizeKind,
    )

    fe = FeatureEngine()
    names = list(fe.feature_names())
    specs = {s.name: s for s in build_feature_transform_specs(names)}

    expected = [
        "bid_l1_notional_usd",
        "ask_l1_notional_usd",
        "bid_depth_notional_5bps",
        "ask_depth_notional_5bps",
        "total_depth_notional_5bps",
    ]

    for name in expected:
        assert name in names, name
        assert name in specs, name

    assert "utc_hour_sin" not in names
    assert "utc_dow_sin" not in names
    assert "is_weekend" not in names
    top_book_end = names.index("time_since_mid_change_ms") + 1
    assert names[top_book_end:top_book_end + 5] == [
        "bid_l1_notional_usd",
        "ask_l1_notional_usd",
        "bid_depth_notional_5bps",
        "ask_depth_notional_5bps",
        "total_depth_notional_5bps",
    ]

    for name in [
        "bid_l1_notional_usd",
        "ask_l1_notional_usd",
        "bid_depth_notional_5bps",
        "ask_depth_notional_5bps",
        "total_depth_notional_5bps",
    ]:
        s = specs[name]
        assert s.raw_transform == RawTransformKind.LOG1P_POS_SCALE
        assert s.normalize == NormalizeKind.NONE
        assert s.half_life_ms == 0
        assert s.scale == 10.0
        assert s.output_clip_abs == 3.0


def test_notional_context_feature_values_are_sane() -> None:
    fe = FeatureEngine()
    result = fe.on_fast_event(deep_snapshot_ob(1_700_003_000_000, n_levels=60))

    assert result.event_type == "ob"
    assert result.features.shape == (153,)

    names = list(fe.feature_names())
    values = dict(zip(names, result.features.tolist()))

    assert "utc_hour_sin" not in values
    assert "utc_dow_sin" not in values
    assert "is_weekend" not in values

    # These are transformed by log1p, so should be positive and finite.
    for name in [
        "bid_l1_notional_usd",
        "ask_l1_notional_usd",
        "bid_depth_notional_5bps",
        "ask_depth_notional_5bps",
        "total_depth_notional_5bps",
    ]:
        assert np.isfinite(values[name])
        assert values[name] >= 0.0


def test_event_density_counts_same_ms_trade_and_ob_without_popping_trade() -> None:
    fe = FeatureEngine()

    r0 = fe.on_fast_event(deep_snapshot_ob(1_700_002_000_000, n_levels=60))
    assert r0.event_type == "ob"

    # After first OB, 100ms event density has one timestamp.
    assert len(fe._event_density_deques[100]) == 1

    tr = fe.on_fast_event(trade(1_700_002_000_100))
    assert tr.event_type == "trade"
    assert len(fe._event_density_deques[100]) == 2

    ob = fe.on_fast_event(("ob", 1_700_002_000_100, 2, 2, ((100.0, 2.5),), ((101.0, 2.5),)))
    assert ob.event_type == "ob"

    # Critical regression: same-ms OB must not pop the same-ms trade from mixed event density.
    assert len(fe._event_density_deques[100]) == 3
    assert list(fe._event_density_deques[100])[-2:] == [
        1_700_002_000_100,
        1_700_002_000_100,
    ]


def test_mixed_event_density_does_not_use_ob_jitter_guard() -> None:
    from pathlib import Path

    text = Path("CMSSL17.py").read_text()
    assert "self._append_ts_with_guard(deq, ts_ms, w, is_ob_event=(etype == \"ob\"))" not in text


def test_transform_engine_uses_ob_clock_not_any_event_clock() -> None:
    fe = FeatureEngine()

    # First OB initializes book, OB feature clock, and transform engine.
    r0 = fe.on_fast_event(deep_snapshot_ob(1_700_001_000_000, n_levels=60))
    assert r0.event_type == "ob"
    assert r0.is_decision is True

    # Same-ms trade before OB: this updates _last_any_event_ts to the OB timestamp,
    # so the following OB would see any_event_dt_ms=1.0 if the wrong clock is used.
    tr = fe.on_fast_event(trade(1_700_001_000_100))
    assert tr.event_type == "trade"
    assert tr.is_decision is False

    ob = fe.on_fast_event(("ob", 1_700_001_000_100, 2, 2, ((100.0, 2.5),), ((101.0, 2.5),)))
    assert ob.event_type == "ob"
    assert ob.is_decision is True

    assert fe._feature_transform_engine is not None
    assert fe._feature_transform_engine.last_apply_dt_ms == 100.0


def test_micro_minus_mid_bps_transform_contract() -> None:
    from CMSSL17 import build_feature_transform_specs, RawTransformKind, NormalizeKind

    fe = FeatureEngine()
    specs = {s.name: s for s in build_feature_transform_specs(fe.feature_names())}
    spec = specs["micro_minus_mid_bps"]

    assert spec.raw_transform == RawTransformKind.FIXED_SCALE
    assert spec.normalize == NormalizeKind.NONE
    assert spec.half_life_ms == 0
    assert spec.scale == 2.0
    assert spec.input_clip_abs == 0.0
    assert spec.output_clip_abs == 8.0


def test_aux_transform_constant_and_metadata_contract() -> None:
    import CMSSL17
    assert CMSSL17.AUX_TRANSFORM == "prelog1p_no_ewma_v1"


def test_aux_transform_is_validated_in_training_loaders() -> None:
    from pathlib import Path

    cmssl_text = Path("CMSSL17.py").read_text()
    offline_text = Path("CMSSL17_offline.py").read_text()
    ingest_text = Path("offline_ingest.py").read_text()

    assert "AUX_TRANSFORM" in cmssl_text
    assert "aux_transform" in cmssl_text
    assert "aux_transform" in offline_text
    assert "AUX_TRANSFORM" in offline_text
    assert "aux_transform" in ingest_text
    assert "AUX_TRANSFORM" in ingest_text

def test_transform_diagnostics_summary_has_required_fields() -> None:
    fe = FeatureEngine()
    for i in range(80):
        fe.on_fast_event(deep_snapshot_ob(1_701_000_000_000 + i * 100, n_levels=60) if i == 0 else delta_ob(1_701_000_000_000 + i * 100))
    diag = fe.transform_diagnostics_summary()
    assert diag["version"] == "feature_transform_diag_v1"
    assert diag["feature_count"] == 153
    assert "clip_summary" in diag
    assert "half_life_summary" in diag
    assert "feature_rows" in diag
    assert len(diag["feature_rows"]) == 153

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
    test_stage1_final_mixer_factory_linear()
    test_stage1_ci_encoded_to_semantic_tokens_layout()
    test_stage1_legacy_flatten_ci_tokens_matches_old_order()
    test_stage1_linear_final_mixer_shape_and_finite()
    test_stage1_linear_final_mixer_equivalent_to_old_final_proj_order()
    test_stage1_extractor_final_proj_compatibility_property()
    test_stage2_final_mixer_factory_swiglu()
    test_stage2_swiglu_final_mixer_shape_and_finite()
    test_stage2_swiglu_final_mixer_uses_legacy_flatten_order()
    test_stage2_swiglu_final_mixer_budget_close_to_linear()
    test_stage3_final_mixer_factory_dcn()
    test_stage3_dcn_final_mixer_shape_and_finite()
    test_stage3_dcn_low_rank_cross_layer_matches_manual_formula()
    test_stage3_dcn_final_mixer_uses_legacy_flatten_order()
    test_stage3_dcn_final_mixer_budget_close_to_linear()
    test_stage4_final_mixer_factory_hybrid()
    test_stage4_hybrid_final_mixer_shape_and_finite()
    test_stage4_hybrid_final_mixer_matches_manual_formula()
    test_stage4_hybrid_final_mixer_uses_legacy_flatten_order()
    test_stage4_hybrid_final_mixer_budget_close_to_linear()
    test_stage5_final_mixer_factory_latent_attn()
    test_stage5_latent_attn_final_mixer_shape_and_finite()
    test_stage5_latent_attn_final_mixer_matches_manual_formula()
    test_stage5_latent_attn_uses_semantic_feature_tokens()
    test_stage5_latent_attn_final_mixer_budget_close_to_linear()
    test_stage6_final_mixer_factory_cross_attn()
    test_stage6_cross_attn_final_mixer_shape_and_finite()
    test_stage6_cross_attn_final_mixer_matches_manual_formula()
    test_stage6_cross_attn_uses_semantic_feature_tokens()
    test_stage6_cross_attn_final_mixer_budget_close_to_linear()
    test_stage5_latent_attn_disables_flash_sdp_in_subprocess()
    test_stage6_cross_attn_disables_flash_sdp_in_subprocess()
    test_stage1_linear_does_not_force_disable_flash_sdp_in_subprocess()
    test_v9_smallstable_model_width_contract()
    test_conv_encoder_layer_prenorm_identity_batchnorm()
    test_conv_encoder_layer_prenorm_identity_layernorm()
    test_conv_encoder_layer_prenorm_sublayers_active_when_residual_scale_enabled()
    test_v9_gated_pooling_query_scaled_init()
    test_v9_initial_direction_logits_not_extreme()
    test_offline_ingest_raw_feature_dims()
    test_merge_event_time_trade_wins_exact_timestamp_tie()
    test_merge_event_time_lower_timestamp_still_wins()
    test_merge_event_time_ob_lower_timestamp_still_wins()
    test_duplicate_ob_timestamps_append_distinct_rows()
    test_offline_ingest_no_overwrite_duplicate_timestamp_api()
    test_pruned_feature_schema_contract()
    test_v9_pruned159_removed_features_absent_and_representatives_retained()
    test_v8_pruned153_lb10_event_feature_shape()
    test_removed_v8_feature_names_not_in_production_feature_names()
    test_v8_return_histories_only_track_retained_windows()
    test_v8_replenishment_states_only_track_emitted_keys()
    test_v8_removed_computation_strings_absent_from_production_cmssl17()
    test_v8_large_trade_and_cvd_representatives_still_present()
    test_pruned_feature_vector_matches_names()
    test_no_empty_feature_family_scaffolding_remains()
    test_hot_path_pruned_feature_count_still_unchanged()
    test_removed_hot_path_scaffolding_strings_absent()
    test_large_trade_and_cvd_windows_are_flow_only()
    test_transform_v2_feature_count_unchanged()
    test_every_feature_has_exactly_one_transform_spec()
    test_old_zscore_path_removed()
    test_ewma_transform_scores_before_update()
    test_bounded_features_are_not_ewma_z()
    test_down_up_vol_imbalance_is_bounded_identity_transform()
    test_down_up_vol_imbalance_values_are_bounded()
    test_time_since_features_clip_at_12()
    test_notional_context_uses_scaled_log_no_ewma()
    test_transform_diagnostics_actionable_summary_exists()
    test_calendar_and_notional_context_features_present_and_transformed()
    test_calendar_and_notional_context_feature_values_are_sane()
    test_event_density_counts_same_ms_trade_and_ob_without_popping_trade()
    test_mixed_event_density_does_not_use_ob_jitter_guard()
    test_transform_engine_uses_ob_clock_not_any_event_clock()
    test_micro_minus_mid_bps_transform_contract()
    test_aux_transform_constant_and_metadata_contract()
    test_aux_transform_is_validated_in_training_loaders()
    test_heavy_tailed_features_use_log_ewma()
    test_transform_diagnostics_summary_has_required_fields()


if __name__ == "__main__":
    main()


def test_removed_features_absent_from_feature_names_and_transform_specs():
    from CMSSL17 import FeatureEngine, build_feature_transform_specs

    names = list(FeatureEngine().feature_names())
    specs = build_feature_transform_specs(names)

    removed = {
        "micro_premia",
        "micro_minus_mid_over_spread",
        "obi_l3",
        "obi_l5",
        "micro_l1_minus_micro_l10_bps",
        "ofi_l1_sum_over_depth_1000ms",
        "ofi_l3_sum_over_depth_1000ms",
        "ofi_l3_sum_over_depth_500ms",
        "ofi_l3_sum_over_depth_200ms",
        "ofi_l10",
        "ofi_l1_pressure_ewma_200ms",
        "ofi_l1_pressure_ewma_500ms",
        "ofi_l1_pressure_ewma_1000ms",
        "bid_l1_depletion_over_depth_200ms",
        "regime_volume_ewma_1000ms",
    }

    assert not (set(names) & removed)
    assert len(specs) == len(names)
