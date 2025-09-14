# -*- coding: utf-8 -*-
"""Kernel-loading utilities for error diffusion algorithms.

Provides functions to load coefficient tables and strength parameters
used by Ostromoukhov and Zhou-Fang dithering methods.
"""

from __future__ import annotations

from .load_kernel import (
    load_ostro_coeffs,
    load_zf_strength,
)

__all__ = [
    "load_ostro_coeffs",
    "load_zf_strength",
]
