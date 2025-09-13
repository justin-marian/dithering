# -*- coding: utf-8 -*-

from __future__ import annotations

import pathlib
from typing import Union

import numpy as np


def load_ostro_coeffs(
    filepath: Union[str, pathlib.Path],
    dtype: type = np.float32
) -> np.ndarray:
    """
    Load coefficients from a whitespace-delimited text file.
    - filepath: path to the text file
    - usecols: columns to read (None = all)
    - dtype: data type of the loaded array
    Returns a 2D numpy array of shape (N=256, 3) with normalized coefficients.
    """
    arr = np.loadtxt(filepath, dtype=dtype)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError(f"Unexpected weights shape {arr.shape} in {filepath}")

    # Normalize rows to sum to 1.0
    s = arr.sum(axis=1, keepdims=True)
    s[s == 0] = 1.0
    arr = (arr / s).astype(np.float32)
    return arr

def load_zf_strength(
    filepath: Union[str, pathlib.Path],
    dtype: type = np.float32
) -> np.ndarray:
    """
    Load Zhou-Fang strength values from a whitespace-delimited text file.
    - filepath: path to the text file
    - dtype: data type of the loaded array
    Returns a 1D numpy array of length N=256 with strength values.
    """
    arr = np.loadtxt(filepath, dtype=np.float64, ndmin=2)
    if arr.shape[1] != 3:
        raise ValueError(f"Unexpected strengths shape {arr.shape} in {filepath}")

    weights = arr[:, 2].astype(np.float64)
    s0 = float(np.clip(np.mean(weights), 0.0, 1.0))
    return np.full((256,), s0, dtype=dtype)
