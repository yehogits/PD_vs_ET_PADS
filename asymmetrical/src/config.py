# Configuration Module: Centralizes file paths and directory structures ensuring code runs identically on any machine.

from __future__ import annotations
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

class Paths:
    
    # Static container for project paths.
    
    # Base Directories
    data = PROJECT_ROOT / "data"
    reports = PROJECT_ROOT / "reports"

    # Data Sub-directories
    data_raw = data / "raw"
    data_processed = data / "processed"

    @classmethod
    def make_directories(cls):
        # Creates the directory tree if it doesn't exist.
        for path in [cls.data_raw, cls.data_processed, cls.reports]:
            path.mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_here(cls) -> type[Paths]:
        return cls