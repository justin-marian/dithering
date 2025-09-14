# -*- coding: utf-8 -*-
"""Noise-based dithering modes (additive and jitter)."""

from __future__ import annotations

from .additive import addition
from .jitter import jitter

__all__ = [
    "addition",
    "jitter",
]
