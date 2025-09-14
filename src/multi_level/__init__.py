# -*- coding: utf-8 -*-
"""Multi-level dithering package.

This subpackage provides functions for multi-level (grayscale palette)
dithering, including palette-based error diffusion.
"""

from __future__ import annotations

from .palette import palette_bw

__all__ = [
    "palette_bw",
]
