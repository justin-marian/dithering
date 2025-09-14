# -*- coding: utf-8 -*-
"""Fan error-diffusion kernels.

Defines the Fan and Shiau-Fan dithering kernels, widely cited in the
literature on halftoning and error diffusion, along with their alias mappings.
"""

from __future__ import annotations

from typing import Dict

from .ktype import Kernel

FAN: Dict[str, Kernel] = {
    "fan": ([
        ((0, +1), 7),
        ((+1, -1), 1), ((+1, 0), 3), ((+1, +1), 5),
    ], 16),  # https://doi.org/10.1117/12.236968

    "shiau_fan": ([
        ((0, +1), 4),
        ((+1, -1), 1), ((+1, 0), 1), ((+1, +1), 2),
    ], 8),  # https://doi.org/10.1117/12.236968

    "shiau_fan2": ([
        ((0, +1), 8),
        ((+1, -1), 1), ((+1, 0), 2), ((+1, +1), 4),
    ], 16),  # https://doi.org/10.1117/12.236968
}

FAN_ALIASES: Dict[str, str] = {
    "FAN": "fan",
    "SF": "shiau_fan",
    "SF2": "shiau_fan2",
}
