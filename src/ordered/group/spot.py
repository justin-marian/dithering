# -*- coding: utf-8 -*-
"""Spot-function threshold map generator for halftone dithering."""

from __future__ import annotations

from typing import Literal

import numpy as np


def spot_threshold(
    size: int = 8,
    angle_deg: float = 45.0,
    spot: Literal["cos+cos", "cosx", "cosx+2cosy"] = "cos+cos",
) -> np.ndarray:
    """
    Generate a halftone spot threshold map.

    Parameters
    ----------
    size : int, default=8
        Tile size in pixels (must be >= 2).
    angle_deg : float, default=45.0
        Rotation angle of the spot function in degrees.
    spot : {"cos+cos", "cosx", "cosx+2cosy"}, default="cos+cos"
        Spot function type.

    Returns
    -------
    np.ndarray
        Threshold map of shape (size, size) with values in [0, 1].

    Raises
    ------
    ValueError
        If `size` < 2 or if `spot` type is unknown.
    """
    if size < 2:
        raise ValueError("size must be >= 2")

    theta = np.deg2rad(angle_deg)
    cos_theta, sin_theta = np.cos(theta), np.sin(theta)

    # Coordinates centered for symmetry
    y, x = np.mgrid[0:size, 0:size].astype(np.float32)
    xc = x - (size / 2.0)
    yc = y - (size / 2.0)

    # Rotate coords (u, v), one cycle across the tile
    u = (xc * cos_theta + yc * sin_theta) / size
    v = (-xc * sin_theta + yc * cos_theta) / size

    if spot == "cos+cos":
        spot_vals = np.cos(2 * np.pi * u) + np.cos(2 * np.pi * v)
    elif spot == "cosx":
        spot_vals = np.cos(2 * np.pi * u)
    elif spot == "cosx+2cosy":
        spot_vals = np.cos(2 * np.pi * u) + 2 * np.cos(2 * np.pi * v)
    else:
        raise ValueError(f"Unknown spot function: {spot}")

    # Rank the values to create threshold map
    flat_idx = np.argsort(spot_vals, axis=None)
    ranks = np.empty_like(flat_idx)
    ranks[flat_idx] = np.arange(flat_idx.size, dtype=np.int32)

    # Normalize to [0, 1]
    threshold_map = ranks.reshape((size, size)).astype(np.float32)
    return (threshold_map + 0.5) / (size * size)
