# -*- coding: utf-8 -*-
"""Kernel loaders for adaptive diffusion dithering.

Provides functions to load and normalize coefficient/strength tables
used by Ostromoukhov (2001) and Zhou-Fang (2007) algorithms.
"""

from __future__ import annotations

import pathlib
from typing import Union

import numpy as np


def load_ostro_coeffs(
    filepath: Union[str, pathlib.Path],
    dtype: type = np.float32,
) -> np.ndarray:
    """Load Ostromoukhov coefficient table.

    Parameters
    ----------
    filepath : str | pathlib.Path
        Path to the whitespace-delimited text file.
    dtype : type, optional
        Numpy dtype for the array. Default np.float32.

    Returns
    -------
    np.ndarray
        2D array of shape (256, 3), rows normalized to sum to 1.0.
    """
    arr = np.loadtxt(filepath, dtype=dtype)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError(f"Unexpected weights shape {arr.shape} in {filepath}")

    s = arr.sum(axis=1, keepdims=True)
    s[s == 0] = 1.0
    arr = (arr / s).astype(np.float32)
    return arr


def load_zf_strength(
    filepath: Union[str, pathlib.Path],
    dtype: type = np.float32,
) -> np.ndarray:
    """Load Zhou-Fang strength table.

    Parameters
    ----------
    filepath : str | pathlib.Path
        Path to the whitespace-delimited text file.
    dtype : type, optional
        Numpy dtype for the array. Default np.float32.

    Returns
    -------
    np.ndarray
        1D array of length 256 with scalar strength values.
    """
    arr = np.loadtxt(filepath, dtype=np.float64, ndmin=2)
    if arr.shape[1] != 3:
        raise ValueError(f"Unexpected strengths shape {arr.shape} in {filepath}")

    weights = arr[:, 2].astype(np.float64)
    s0 = float(np.clip(np.mean(weights), 0.0, 1.0))
    return np.full((256,), s0, dtype=dtype)
