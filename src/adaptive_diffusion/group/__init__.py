# -*- coding: utf-8 -*-
"""Grouped adaptive diffusion implementations.

This subpackage collects and re-exports individual algorithms:
- Ostromoukhov black-white diffusion
- Zhou-Fang black-white diffusion
"""

from __future__ import annotations

from .ostromoukhov import ostromoukhov_bw
from .zhou_fang import zhou_fang_bw

__all__ = [
    "ostromoukhov_bw",
    "zhou_fang_bw",
]
