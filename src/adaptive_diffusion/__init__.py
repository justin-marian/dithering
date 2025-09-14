# -*- coding: utf-8 -*-
"""Adaptive diffusion dithering algorithms.

This subpackage provides implementations of Ostromoukhov's and Zhou-Fang's
adaptive diffusion methods for black-white dithering.
"""

from __future__ import annotations

from .adaptive_dif import adaptive_diff_bw

__all__ = [
    "adaptive_diff_bw",
]
