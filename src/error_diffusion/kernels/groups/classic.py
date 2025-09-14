# -*- coding: utf-8 -*-
"""Classic error-diffusion kernels.

Defines the Floyd-Steinberg, Jarvis-Judice-Ninke, and Stucki
dithering kernels, along with their common alias mappings.
"""

from __future__ import annotations

from typing import Dict

from ktype import Kernel

CLASSIC: Dict[str, Kernel] = {
    "floyd_steinberg": ([
        ((0, +1), 7),
        ((+1, -1), 3), ((+1, 0), 5), ((+1, +1), 1),
    ], 16),  # https://en.wikipedia.org/wiki/Floyd%E2%80%93Steinberg_dithering

    "jarvis_judice_ninke": ([
        ((0, +1), 7),  ((0, +2), 5),
        ((+1, -2), 3), ((+1, -1), 5), ((+1, 0), 7),
        ((+1, +1), 5), ((+1, +2), 3),
        ((+2, -2), 1), ((+2, -1), 3), ((+2, 0), 5),
        ((+2, +1), 3), ((+2, +2), 1),
    ], 48),  # https://en.wikipedia.org/wiki/Error_diffusion

    "stucki": ([
        ((0, +1), 8),  ((0, +2), 4),
        ((+1, -2), 2), ((+1, -1), 4), ((+1, 0), 8),
        ((+1, +1), 4), ((+1, +2), 2),
        ((+2, -2), 1), ((+2, -1), 2), ((+2, 0), 4),
        ((+2, +1), 2), ((+2, +2), 1),
    ], 42),  # https://en.wikipedia.org/wiki/Dither
}

CLASSIC_ALIASES: Dict[str, str] = {
    "FS": "floyd_steinberg",
    "JJN": "jarvis_judice_ninke",
    "ST": "stucki",
}
