# -*- coding: utf-8 -*-
"""Research error-diffusion kernels.

Provides kernels proposed in research literature such as:
- Bell
- Stevenson-Arce

Also defines their alias mappings.
"""

from __future__ import annotations

from typing import Dict

from .ktype import Kernel

RESEARCH: Dict[str, Kernel] = {
    "bell": ([
        ((0, +1), 8),  ((0, +2), 4),
        ((+1, -2), 2), ((+1, -1), 4), ((+1, 0), 8),
        ((+1, +1), 4), ((+1, +2), 2),
        ((+2, -1), 1), ((+2, 0), 2),  ((+2, +1), 1),
    ], 36),  # https://en.wikipedia.org/wiki/Error_diffusion

    "stevenson_arce": ([
        ((0, +1), 32), ((0, +2), 12),
        ((+1, -2), 5),  ((+1, -1), 12), ((+1, 0), 26),
        ((+1, +1), 12), ((+1, +2), 5),
        ((+2, -2), 2),  ((+2, -1), 5),  ((+2, 0), 12),
        ((+2, +1), 5),  ((+2, +2), 2),
    ], 200),  # https://en.wikipedia.org/wiki/Error_diffusion
}

RESEARCH_ALIASES: Dict[str, str] = {
    "BELL": "bell",
    "SA": "stevenson_arce",
}
