# -*- coding: utf-8 -*-
"""Normal-distribution noise generator for random dithering."""

from __future__ import annotations

from typing import Optional

import numpy as np


def normal_distribution(
    mean: float,
    stddev: float,
    size: tuple[int, int],
    seed: Optional[int] = None
) -> np.ndarray:
    """
    Generate Gaussian-distributed noise.

    Parameters
    ----------
    mean : float
        Mean of the normal distribution.
    stddev : float
        Standard deviation of the distribution.
    size : tuple[int, int]
        Output shape of the noise array.
    seed : Optional[int], default=None
        Seed for the random number generator (for reproducibility).

    Returns
    -------
    np.ndarray
        2D array of normally distributed noise values (float32).
    """
    rng = np.random.default_rng(seed)
    return rng.normal(mean, stddev, size=size).astype(np.float32)
