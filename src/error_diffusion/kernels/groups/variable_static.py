# -*- coding: utf-8 -*-
"""Variable static error-diffusion kernels.

Provides static (non-adaptive) versions of variable-coefficient kernels:
- Ostromoukhov static
- Zhou-Fang static

Each kernel is defined with fixed diffusion weights and denominator.
Aliases are also provided for shorthand references.
"""

from __future__ import annotations

from typing import Dict

from .ktype import Kernel

VARIABLE_STATIC: Dict[str, Kernel] = {
    "ostromoukhov_static": ([
        ((0, +1), 13),
        ((+1, -1), 0), ((+1, 0), 5), ((+1, +1), 2),
    ], 20),  # https://dl.acm.org/doi/10.1145/383259.383326

    "zhou_fang_static": ([
        ((0, +1), 7),
        ((+1, -1), 1), ((+1, 0), 3), ((+1, +1), 1),
    ], 12),  # https://en.wikipedia.org/wiki/Error_diffusion
}

VARIABLE_STATIC_ALIASES: Dict[str, str] = {
    "OST": "ostromoukhov_static",
    "ZF": "zhou_fang_static",
}
