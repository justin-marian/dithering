# -*- coding: utf-8 -*-
"""Kernel group definitions for error-diffusion dithering.

This module collects canonical kernels and their alias mappings
across different research families (classic, sierra, burkes, etc.).
"""

from __future__ import annotations

from .artistic import ARTISTIC, ARTISTIC_ALIASES
from .atkinson import ATKINSON, ATKINSON_ALIASES
from .burkes import BURKES, BURKES_ALIASES
from .classic import CLASSIC, CLASSIC_ALIASES
from .directional import DIRECTIONAL, DIRECTIONAL_ALIASES
from .experimental import EXPERIMENTAL, EXPERIMENTAL_ALIASES
from .fan import FAN, FAN_ALIASES
from .ktype import Kernel
from .optimized import OPTIMIZED, OPTIMIZED_ALIASES
from .research import RESEARCH, RESEARCH_ALIASES
from .sierra import SIERRA, SIERRA_ALIASES
from .simple import SIMPLE, SIMPLE_ALIASES
from .variable_static import VARIABLE_STATIC, VARIABLE_STATIC_ALIASES

__all__ = [
    "ARTISTIC", "ARTISTIC_ALIASES",
    "ATKINSON", "ATKINSON_ALIASES",
    "DIRECTIONAL", "DIRECTIONAL_ALIASES",
    "OPTIMIZED", "OPTIMIZED_ALIASES",
    "SIMPLE", "SIMPLE_ALIASES",
    "VARIABLE_STATIC", "VARIABLE_STATIC_ALIASES",
    "CLASSIC", "CLASSIC_ALIASES",
    "SIERRA", "SIERRA_ALIASES",
    "BURKES", "BURKES_ALIASES",
    "RESEARCH", "RESEARCH_ALIASES",
    "FAN", "FAN_ALIASES",
    "EXPERIMENTAL", "EXPERIMENTAL_ALIASES",
    "Kernel",
]
