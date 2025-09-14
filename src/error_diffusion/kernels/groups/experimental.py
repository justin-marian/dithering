# -*- coding: utf-8 -*-
"""Experimental error-diffusion kernels.

Defines a set of non-standard or exploratory dithering kernels
(e.g., ripple, cluster, serpentine, diamond, hexagonal)
and their alias mappings.
"""

from __future__ import annotations

from typing import Dict

from .ktype import Kernel

EXPERIMENTAL: Dict[str, Kernel] = {
    "ripple": ([
        ((0, +1), 4), ((0, +2), 2), ((0, +3), 1),
        ((+1, -1), 1), ((+1, 0), 2), ((+1, +1), 1),
        ((+2, 0), 1),
    ], 12),  # https://jdobr.es/blog/dithering/

    "cluster": ([
        ((0, +1), 3), ((0, +2), 3),
        ((+1, 0), 3), ((+1, +1), 3),
    ], 12),  # https://en.wikipedia.org/wiki/Ordered_dithering

    "serpentine": ([
        ((0, +1), 6),
        ((+1, -1), 2), ((+1, 0), 4), ((+1, +1), 2),
        ((+2, -1), 1), ((+2, +1), 1),
    ], 16),  # https://en.wikipedia.org/wiki/Floyd%E2%80%93Steinberg_dithering

    "diamond": ([
        ((0, +1), 4),
        ((-1, 0), 2),
        ((+1, -1), 1), ((+1, 0), 2), ((+1, +1), 1),
    ], 10),  # https://surma.dev/things/ditherpunk/

    "hexagonal": ([
        ((0, +1), 3), ((0, +2), 1),
        ((-1, +1), 1),
        ((+1, -1), 1), ((+1, 0), 3), ((+1, +1), 3),
    ], 12),  # https://surma.dev/things/ditherpunk/
}

EXPERIMENTAL_ALIASES: Dict[str, str] = {
    "RIP": "ripple",
    "CL": "cluster",
    "SERP": "serpentine",
    "DIA": "diamond",
    "HEX": "hexagonal",
}
