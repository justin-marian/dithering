# -*- coding: utf-8 -*-

from __future__ import annotations

from typing import Dict

from .ktype import Kernel

DIRECTIONAL: Dict[str, Kernel] = {
    "diagonal": ([
        ((0, +1), 1),
        ((+1, +1), 1),
    ], 2),  # https://en.wikipedia.org/wiki/Error_diffusion

    "x_diagonal": ([
        ((+1, -1), 1),
        ((+1, +1), 1),
    ], 2),  # https://en.wikipedia.org/wiki/Error_diffusion

    "horizontal": ([
        ((0, +1), 1),
    ], 1),  # https://en.wikipedia.org/wiki/Error_diffusion

    "vertical": ([
        ((+1, 0), 1),
    ], 1),  # https://en.wikipedia.org/wiki/Error_diffusion

    "cross": ([
        ((0, +1), 1),
        ((0, -1), 1),
        ((-1, 0), 1),
        ((+1, 0), 1),
    ], 4),  # https://en.wikipedia.org/wiki/Error_diffusion
}

DIRECTIONAL_ALIASES: Dict[str, str] = {
    "DIAG": "diagonal",
    "XDIAG": "x_diagonal",
    "H": "horizontal",
    "V": "vertical",
    "CROSS": "cross",
}
