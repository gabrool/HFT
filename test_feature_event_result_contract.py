"""Smoke checks for FeatureEngine event-result API contract."""

import sys
import types

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

    sklearn_mod = types.ModuleType("sklearn")
    decomposition_mod = types.ModuleType("sklearn.decomposition")
    decomposition_mod.PCA = type("PCA", (_Module,), {})
    sys.modules.setdefault("sklearn", sklearn_mod)
    sys.modules.setdefault("sklearn.decomposition", decomposition_mod)

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

from CMSSL17 import FeatureEngine, FeatureEventResult


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


if __name__ == "__main__":
    main()
