# -*- coding: utf-8 -*-
"""Uniform-distribution noise generator for random dithering."""

from __future__ import annotations

from typing import Optional

import numpy as np


def uniform_distrib(
    low: float,
    high: float,
    size: tuple[int, int],
    seed: Optional[int] = None
) -> np.ndarray:
    """
    Generate uniformly distributed noise.

    Parameters
    ----------
    low : float
        Lower bound of the distribution range.
    high : float
        Upper bound of the distribution range.
    size : tuple[int, int]
        Output shape of the noise array.
    seed : Optional[int], default=None
        Seed for the random number generator (for reproducibility).

    Returns
    -------
    np.ndarray
        2D array of uniformly distributed noise values (float32).
    """
    rng = np.random.default_rng(seed)
    return rng.uniform(low, high, size=size).astype(np.float32)
