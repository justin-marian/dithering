# -*- coding: utf-8 -*-
"""Top-level package for the dithering project.

Exposes submodules implementing various dithering algorithms.
"""

from __future__ import annotations

from . import adaptive_diffusion, error_diffusion, multi_level, naive, ordered, random

__all__ = [
    "adaptive_diffusion",
    "error_diffusion",
    "multi_level",
    "naive",
    "ordered",
    "random",
]
