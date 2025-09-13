# -*- coding: utf-8 -*-

from __future__ import annotations

from typing import Dict

from .ktype import Kernel

OPTIMIZED: Dict[str, Kernel] = {
    "minimized_average_error": ([
        ((0, +1), 8),  ((0, +2), 4),
        ((+1, -2), 2), ((+1, -1), 4), ((+1, 0), 8), ((+1, +1), 4), ((+1, +2), 2),
        ((+2, -2), 1), ((+2, -1), 2), ((+2, 0), 4), ((+2, +1), 2), ((+2, +2), 1),
    ], 42),  # https://en.wikipedia.org/wiki/Dither

    "pigeon": ([
        ((0, +1), 2),
        ((+1, -1), 1), ((+1, 0), 2), ((+1, +1), 1),
    ], 8),  # https://en.wikipedia.org/wiki/Error_diffusion
}

OPTIMIZED_ALIASES: Dict[str, str] = {
    "MAE": "minimized_average_error",
    "P": "pigeon",
}
