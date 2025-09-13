# -*- coding: utf-8 -*-

from __future__ import annotations

from typing import Dict

from .ktype import Kernel

ARTISTIC: Dict[str, Kernel] = {
    "threshold": ([
        ((0, +1), 3),
        ((+1, -1), 2), ((+1, 0), 3), ((+1, +1), 2),
    ], 10),  # https://en.wikipedia.org/wiki/Dither

    "gradient": ([
        ((0, +1), 6), ((0, +2), 4), ((0, +3), 2),
        ((+1, 0), 2),
    ], 14),  # https://jdobr.es/blog/dithering/

    "radial": ([
        ((0, +1), 4), ((0, +2), 2),
        ((-1, +1), 1),
        ((+1, -1), 2), ((+1, 0), 4), ((+1, +1), 2),
        ((+2, 0), 1),
    ], 16),  # https://surma.dev/things/ditherpunk/

    "spiral": ([
        ((0, +1), 5),
        ((+1, +1), 3),
        ((+1, 0), 2),
        ((+1, -1), 1),
    ], 11),  # https://surma.dev/things/ditherpunk/

    "asymmetric": ([
        ((0, +1), 5), ((0, +2), 2),
        ((+1, -1), 1), ((+1, 0), 3), ((+1, +1), 4), ((+1, +2), 1),
    ], 16),  # https://jdobr.es/blog/dithering/

    "weighted_diagonal": ([
        ((0, +1), 2),
        ((+1, -1), 3), ((+1, 0), 1), ((+1, +1), 3),
    ], 9),  # https://numbersandshapes.net/posts/error_diffusion/
}

ARTISTIC_ALIASES: Dict[str, str] = {
    "T": "threshold",
    "G": "gradient",
    "RAD": "radial",
    "SP": "spiral",
    "ASYM": "asymmetric",
    "WD": "weighted_diagonal",
}
