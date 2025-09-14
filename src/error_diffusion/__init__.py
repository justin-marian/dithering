# -*- coding: utf-8 -*-
"""Error diffusion dithering algorithms.

This subpackage provides implementations of error-diffusion
methods for black-white image processing.
"""

from __future__ import annotations

from .err_diff import error_diff_bw

__all__ = [
    "error_diff_bw",
]
