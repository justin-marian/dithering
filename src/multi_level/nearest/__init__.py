# -*- coding: utf-8 -*-
"""Nearest-color utilities for palette-based dithering.

This subpackage exposes helper functions for finding the closest
palette entry to a given pixel, used in multi-level dithering.
"""

from __future__ import annotations

from .near_color import nearest_color

__all__ = [
    "nearest_color",
]
