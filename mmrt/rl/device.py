"""Torch device helpers for execution PPO."""

from __future__ import annotations

from typing import Any

import torch

__all__ = [
    "canonicalize_torch_device",
    "resolve_torch_device",
    "torch_device_summary",
    "cuda_memory_summary",
]


def canonicalize_torch_device(device: torch.device) -> torch.device:
    if not isinstance(device, torch.device):
        raise ValueError("device must be a torch.device")
    if device.type == "cuda" and device.index is None and torch.cuda.is_available():
        return torch.device("cuda", torch.cuda.current_device())
    return device


def resolve_torch_device(device: str | torch.device | None) -> torch.device:
    if device is None:
        return torch.device("cpu")
    if isinstance(device, torch.device):
        return canonicalize_torch_device(device)
    if not isinstance(device, str) or not device.strip():
        raise ValueError("device must be None, str, or torch.device")
    normalized = device.strip().lower()
    if normalized == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda", torch.cuda.current_device())
        return torch.device("cpu")
    return canonicalize_torch_device(torch.device(device.strip()))


def torch_device_summary(*, requested_device: str | torch.device | None, resolved_device: torch.device) -> dict[str, Any]:
    if not isinstance(resolved_device, torch.device):
        raise ValueError("resolved_device must be a torch.device")
    resolved_device = canonicalize_torch_device(resolved_device)
    cuda_available = bool(torch.cuda.is_available())
    summary: dict[str, Any] = {
        "requested_device": None if requested_device is None else str(requested_device),
        "resolved_device": str(resolved_device),
        "torch_version": str(torch.__version__),
        "cuda_available": cuda_available,
        "cuda_device_name": None,
    }
    if cuda_available:
        try:
            summary["cuda_device_name"] = torch.cuda.get_device_name(resolved_device)
        except Exception:
            summary["cuda_device_name"] = torch.cuda.get_device_name(0)
    return summary


def cuda_memory_summary(device: torch.device) -> dict[str, int] | None:
    if not isinstance(device, torch.device):
        raise ValueError("device must be a torch.device")
    device = canonicalize_torch_device(device)
    if device.type != "cuda" or not torch.cuda.is_available():
        return None
    return {
        "memory_allocated_bytes": int(torch.cuda.memory_allocated(device)),
        "memory_reserved_bytes": int(torch.cuda.memory_reserved(device)),
        "max_memory_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "max_memory_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
    }
