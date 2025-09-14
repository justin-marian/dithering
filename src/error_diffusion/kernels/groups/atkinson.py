# -*- coding: utf-8 -*-
"""Atkinson error-diffusion kernel.

Defines the Atkinson dithering kernel (Bill Atkinson, 1980s)
and its alias mapping.
"""

from __future__ import annotations

from typing import Dict

from .ktype import Kernel

ATKINSON: Dict[str, Kernel] = {
    "atkinson": ([
        ((0, +1), 1),  ((0, +2), 1),
        ((+1, -1), 1), ((+1, 0), 1), ((+1, +1), 1),
        ((+2, 0), 1),
    ], 8),  # https://en.wikipedia.org/wiki/Atkinson_dithering
}

ATKINSON_ALIASES: Dict[str, str] = {
    "A": "atkinson",
}
