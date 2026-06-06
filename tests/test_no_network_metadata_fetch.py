from __future__ import annotations

import ast
from pathlib import Path


PRODUCTION_ROOT = Path("mmrt")
FORBIDDEN_NETWORK_IMPORTS = ("requests", "httpx", "aiohttp", "urllib.request")
FORBIDDEN_BINANCE_URLS = (
    "https://fapi.binance.com",
    "https://api.binance.com",
    "/fapi/v1/exchangeInfo",
)


def test_production_code_does_not_fetch_exchange_metadata_from_network() -> None:
    offenders: list[str] = []
    for path in PRODUCTION_ROOT.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        tree = ast.parse(text, filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    name = alias.name
                    if name in FORBIDDEN_NETWORK_IMPORTS or any(name.startswith(f"{forbidden}.") for forbidden in FORBIDDEN_NETWORK_IMPORTS):
                        offenders.append(f"{path}:{node.lineno}: forbidden import {name}")
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                if module in FORBIDDEN_NETWORK_IMPORTS or any(module.startswith(f"{forbidden}.") for forbidden in FORBIDDEN_NETWORK_IMPORTS):
                    offenders.append(f"{path}:{node.lineno}: forbidden import {module}")
        for forbidden in FORBIDDEN_BINANCE_URLS:
            if forbidden in text:
                offenders.append(f"{path}: forbidden Binance metadata URL {forbidden}")
    assert offenders == []
