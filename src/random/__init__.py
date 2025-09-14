# -*- coding: utf-8 -*-
"""Random dithering package initializer.

Exposes the `random_bw` function for noise-based dithering.
"""

from __future__ import annotations

from .noise import random_bw

__all__ = [
    "random_bw",
]
