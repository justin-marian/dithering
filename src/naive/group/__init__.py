# -*- coding: utf-8 -*-
"""Thresholding methods for naive dithering.

This package-level module exposes multiple threshold selection strategies
(global, mean, percentile, and Otsu) under a unified namespace.
"""

from __future__ import annotations

from .globalg import global_threshold
from .mean import mean_threshold
from .otsu import otsu_threshold
from .percentile import percentile_threshold

__all__ = [
    "global_threshold",
    "mean_threshold",
    "percentile_threshold",
    "otsu_threshold",
]
