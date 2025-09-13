# -*- coding: utf-8 -*-

from __future__ import annotations

from typing import Dict

from .ktype import Kernel

SIERRA: Dict[str, Kernel] = {
    "sierra": ([
        ((0, +1), 5),  ((0, +2), 3),
        ((+1, -2), 2), ((+1, -1), 4), ((+1, 0), 5), ((+1, +1), 4), ((+1, +2), 2),
        ((+2, -1), 2), ((+2, 0), 3),  ((+2, +1), 2),
    ], 32),  # https://tannerhelland.com/2012/12/28/dithering-eleven-algorithms-source-code.html

    "two_row_sierra": ([
        ((0, +1), 4),  ((0, +2), 3),
        ((+1, -2), 1), ((+1, -1), 2), ((+1, 0), 3), ((+1, +1), 2), ((+1, +2), 1),
    ], 16),  # https://tannerhelland.com/2012/12/28/dithering-eleven-algorithms-source-code.html

    "sierra_lite": ([
        ((0, +1), 2),
        ((+1, -1), 1), ((+1, 0), 1),
    ], 4),  # https://tannerhelland.com/2012/12/28/dithering-eleven-algorithms-source-code.html

    "sierra_2": ([
        ((0, +1), 4),  ((0, +2), 3),
        ((+1, -2), 1), ((+1, -1), 2), ((+1, 0), 3), ((+1, +1), 2), ((+1, +2), 1),
    ], 16),  # https://tannerhelland.com/2012/12/28/dithering-eleven-algorithms-source-code.html

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
