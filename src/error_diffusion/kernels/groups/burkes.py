# -*- coding: utf-8 -*-
"""Burkes error-diffusion kernel.

Defines the Burkes dithering kernel and its alias mapping.
"""

from __future__ import annotations

from typing import Dict

from ktype import Kernel

BURKES: Dict[str, Kernel] = {
    "burkes": ([
        ((0, +1), 8),  ((0, +2), 4),
        ((+1, -2), 2), ((+1, -1), 4), ((+1, 0), 8),
        ((+1, +1), 4), ((+1, +2), 2),
    ], 32),  # https://tannerhelland.com/2012/12/28/dithering-eleven-algorithms-source-code.html
}

BURKES_ALIASES: Dict[str, str] = {
    "B": "burkes",
}
