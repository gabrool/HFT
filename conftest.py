"""Pytest compatibility shims for the Windows Codex test runner."""

from __future__ import annotations

import os
from pathlib import Path


if os.name == "nt":
    _ORIGINAL_PATH_MKDIR = Path.mkdir

    def _mkdir_without_private_windows_mode(
        self: Path,
        mode: int = 0o777,
        parents: bool = False,
        exist_ok: bool = False,
    ) -> None:
        if mode == 0o700:
            mode = 0o777
        return _ORIGINAL_PATH_MKDIR(
            self,
            mode=mode,
            parents=parents,
            exist_ok=exist_ok,
        )

    Path.mkdir = _mkdir_without_private_windows_mode
