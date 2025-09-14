# -*- coding: utf-8 -*-
"""Sierra error-diffusion kernels.

Provides several variants of the Sierra dithering kernel:
- Sierra (original)
- Two-row Sierra
- Sierra Lite
- Sierra 2
- Filter Lite

Each kernel is defined with its weight distribution and denominator.
Aliases are also provided for shorthand references.
"""

from __future__ import annotations

from typing import Dict

from .ktype import Kernel

SIERRA: Dict[str, Kernel] = {
    "sierra": ([
        ((0, +1), 5),  ((0, +2), 3),
        ((+1, -2), 2), ((+1, -1), 4), ((+1, 0), 5),
        ((+1, +1), 4), ((+1, +2), 2),
        ((+2, -1), 2), ((+2, 0), 3),  ((+2, +1), 2),
    ], 32),  # https://tannerhelland.com/2012/12/28/dithering-eleven-algorithms-source-code.html

    "two_row_sierra": ([
        ((0, +1), 4),  ((0, +2), 3),
        ((+1, -2), 1), ((+1, -1), 2), ((+1, 0), 3),
        ((+1, +1), 2), ((+1, +2), 1),
    ], 16),  # same source

    "sierra_lite": ([
        ((0, +1), 2),
        ((+1, -1), 1), ((+1, 0), 1),
    ], 4),  # same source

    "sierra_2": ([
        ((0, +1), 4),  ((0, +2), 3),
        ((+1, -2), 1), ((+1, -1), 2), ((+1, 0), 3),
        ((+1, +1), 2), ((+1, +2), 1),
    ], 16),  # same source

    "filter_lite": ([
        ((0, +1), 2),
        ((+1, 0), 1),
    ], 4),  # https://numbersandshapes.net/posts/simple_error_diffusion/
}

SIERRA_ALIASES: Dict[str, str] = {
    "SI": "sierra",
    "2RS": "two_row_sierra",
    "SL": "sierra_lite",
    "S2": "sierra_2",
    "FL": "filter_lite",
}
