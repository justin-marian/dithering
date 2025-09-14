# -*- coding: utf-8 -*-
"""Noise distribution generators for random dithering."""

from __future__ import annotations

from .normal import normal_distribution
from .uniform import uniform_distrib

__all__ = [
    "normal_distribution",
    "uniform_distrib",
]
