# -*- coding: utf-8 -*-

from __future__ import annotations

from typing import Literal

import numpy as np


def spot_threshold(
    size: int = 8,
    angle_deg: float = 45.0,
    spot: Literal["cos+cos", "cosx", "cosx+2cosy"] = "cos+cos"
) -> np.ndarray:
    if size < 2:
        raise ValueError("size must be >= 2")
    theta = np.deg2rad(angle_deg)
    cos, sin = np.cos(theta), np.sin(theta)

    # Coordinates centered for symmetry
    y, x = np.mgrid[0:size, 0:size].astype(np.float32)
    xc = x - (size / 2.0)
    yc = y - (size / 2.0)

    # Rotate coords (u,v) one cycle across the tile
    # One period per tile → frequency = 1/size
    u = ( xc * cos + yc * sin) / size
    v = (-xc * sin + yc * cos) / size
    if spot == "cos+cos":
        S = np.cos(2 * np.pi * u) + np.cos(2 * np.pi * v)
    elif spot == "cosx":
        S = np.cos(2 * np.pi * u)
    elif spot == "cosx+2cosy":
        S = np.cos(2 * np.pi * u) + 2 * np.cos(2 * np.pi * v)
    else:
        raise ValueError("Unknown spot function")

    # Rank the values to create threshold map
    flat_idx = np.argsort(S, axis=None)
    ranks = np.empty_like(flat_idx)
    ranks[flat_idx] = np.arange(flat_idx.size, dtype=np.int32)

    # Normalize to [0,1]
    T = ranks.reshape((size, size)).astype(np.float32)
    return (T + 0.5) / (size * size)
