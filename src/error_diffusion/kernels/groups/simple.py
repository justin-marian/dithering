# -*- coding: utf-8 -*-
"""Simple error-diffusion kernels.

Provides lightweight or minimal kernels used for fast dithering:
- Simple 2D
- False Floyd-Steinberg
- Three-pixel
- Four-pixel
- Micro

Each kernel is defined with its diffusion offsets and denominator.
Aliases are also provided for shorthand references.
"""

from __future__ import annotations

from typing import Dict

from .ktype import Kernel

SIMPLE: Dict[str, Kernel] = {
    "simple_2d": ([
        ((0, +1), 1),
        ((+1, 0), 1),
    ], 2),  # https://en.wikipedia.org/wiki/Error_diffusion

    "false_floyd_steinberg": ([
        ((0, +1), 3),
        ((+1, -1), 1), ((+1, 0), 1),
    ], 8),  # https://tannerhelland.com/2012/12/28/dithering-eleven-algorithms-source-code.html

    "three_pixel": ([
        ((0, +1), 2),
        ((+1, -1), 1), ((+1, 0), 1),
    ], 4),  # https://numbersandshapes.net/posts/error_diffusion/

    "four_pixel": ([
        ((0, +1), 2),
        ((+1, -1), 1), ((+1, 0), 2), ((+1, +1), 1),
    ], 6),  # https://surma.dev/things/ditherpunk/

    "micro": ([
        ((0, +1), 1),
    ], 1),  # https://en.wikipedia.org/wiki/Error_diffusion
}

SIMPLE_ALIASES: Dict[str, str] = {
    "S2D": "simple_2d",
    "FFS": "false_floyd_steinberg",
    "3P": "three_pixel",
    "4P": "four_pixel",
    "M": "micro",
}
