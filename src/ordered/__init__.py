# -*- coding: utf-8 -*-
"""Ordered dithering algorithms.

This package provides implementations of ordered dithering methods,
such as Bayer matrices and halftone spot functions.
"""

from __future__ import annotations

from .ordered import ordered_bw

__all__ = [
    "ordered_bw",
]
