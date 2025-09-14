# -*- coding: utf-8 -*-
"""Naive threshold-based dithering methods.

This subpackage provides simple binarization approaches without
error diffusion, including global, mean, percentile, and Otsu thresholding.
"""

from __future__ import annotations

from .threshold import threshold_bw

__all__ = [
    "threshold_bw",
]
