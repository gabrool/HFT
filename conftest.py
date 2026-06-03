import sys
import types


def install_optional_dependency_stubs() -> None:
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

    class _Module:
        def __init__(self, *args, **kwargs):
            pass

    class _Parameter:
        def __init__(self, value=None, *args, **kwargs):
            self.value = value

    torch_mod = types.ModuleType("torch")
    torch_mod.Tensor = _Dummy
    torch_mod.cuda = types.SimpleNamespace(empty_cache=lambda: None)
    torch_mod.ones = lambda *args, **kwargs: 0
    torch_mod.tensor = lambda *args, **kwargs: 0
    torch_mod.empty = lambda *args, **kwargs: 0
    torch_mod.randn = lambda *args, **kwargs: 0
    torch_mod.exp = lambda x: x
    torch_mod.log = lambda x: x
    torch_mod.arange = lambda *args, **kwargs: []
    torch_mod.float32 = float
    torch_mod.no_grad = lambda func=None: (lambda f: f) if func is None else func

    nn_mod = types.ModuleType("torch.nn")
    nn_mod.Module = _Module
    nn_mod.Parameter = _Parameter
    for name in ("Linear", "Conv1d", "SiLU", "ReLU", "GELU", "Dropout", "LayerNorm", "BatchNorm1d", "MultiheadAttention", "Sequential", "ModuleList"):
        setattr(nn_mod, name, type(name, (_Module,), {}))

    functional_mod = types.ModuleType("torch.nn.functional")
    utils_mod = types.ModuleType("torch.utils")
    data_mod = types.ModuleType("torch.utils.data")
    data_mod.Dataset = type("Dataset", (_Module,), {})
    data_mod.DataLoader = type("DataLoader", (_Module,), {})
    functorch_mod = types.ModuleType("torch._functorch")
    config_mod = types.ModuleType("torch._functorch.config")
    optim_mod = types.ModuleType("torch.optim")
    optim_mod.Optimizer = type("Optimizer", (_Module,), {"__init__": lambda self, *args, **kwargs: None})

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


def pytest_configure(config):
    install_optional_dependency_stubs()
