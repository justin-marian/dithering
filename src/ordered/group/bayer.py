# -*- coding: utf-8 -*-
"""Bayer matrices and ordered dithering (B/W)."""

from __future__ import annotations

from typing import Literal, Union

import numpy as np

from utils import binarize, grayscale

# Predefined Bayer matrices (normalized to [0,1))
BAYER_2: np.ndarray = (1 / 4) * np.array(
    [[0, 2],
     [3, 1]],
    dtype=np.float32,
)

BAYER_4: np.ndarray = (1 / 16) * np.array(
    [
        [0, 8, 2, 10],
        [12, 4, 14, 6],
        [3, 11, 1, 9],
        [15, 7, 13, 5],
    ],
    dtype=np.float32,
)

BAYER_8: np.ndarray = (1 / 64) * np.array(
    [
        [0, 32, 8, 40, 2, 34, 10, 42],
        [48, 16, 56, 24, 50, 18, 58, 26],
        [12, 44, 4, 36, 14, 46, 6, 38],
        [60, 28, 52, 20, 62, 30, 54, 22],
        [3, 35, 11, 43, 1, 33, 9, 41],
        [51, 19, 59, 27, 49, 17, 57, 25],
        [15, 47, 7, 39, 13, 45, 5, 37],
        [63, 31, 55, 23, 61, 29, 53, 21],
    ],
    dtype=np.float32,
)


def bayer_matrix(n: int) -> np.ndarray:
    """
    Generate an nxn Bayer matrix normalized to [0, 1).

    Parameters
    ----------
    n : int
        Size of the Bayer matrix. Must be a power of two (2, 4, 8, 16, ...).

    Returns
    -------
    np.ndarray
        An nxn float32 matrix with values in [0, 1).

    Raises
    ------
    ValueError
        If `n` is not a power of two or is less than 2.
    """
    if n == 2:
        return BAYER_2
    if n == 4:
        return BAYER_4
    if n == 8:
        return BAYER_8

    if n < 2 or (n & (n - 1)) != 0:
        raise ValueError("Bayer matrix size must be a power of two (2, 4, 8, 16, ...)")

    # Recursive construction for larger sizes (16, 32, 64, ...)
    smaller = bayer_matrix(n // 2)
    
    # Combine four smaller matrices into one larger matrix
    top = np.hstack((4 * smaller + 0, 4 * smaller + 2))
    bottom = np.hstack((4 * smaller + 3, 4 * smaller + 1))
    return (1 / (n * n)) * np.vstack((top, bottom))


def bayer_bw(
    img: np.ndarray,
    *,
    matrix: np.ndarray = BAYER_8,
    dtype: Union[Literal["u8"], Literal["f32"], np.dtype, type] = "u8",
) -> np.ndarray:
    """
    Ordered dithering (Bayer) to 1-bit.

    Parameters
    ----------
    img : np.ndarray
        Grayscale image (H, W). Values can be uint8 [0..255] or float in [0..1].
    matrix : np.ndarray, default BAYER_8
        Bayer matrix normalized to [0, 1).
    dtype : {"u8","f32"} | np.dtype | type, default "u8"
        Output dtype. 'u8' → {0,255}; float dtypes → {0.0,1.0}; other ints → {0,max(dtype)}.

    Returns
    -------
    np.ndarray
        Binarized image with values mapped according to `dtype`.

    Raises
    ------
    ValueError
        If `img` is not grayscale, or `matrix` is not square, or its size is not a power of two.
    """
    if img.ndim != 2:
        raise ValueError("Input image must be grayscale (H, W)")

    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Matrix must be a square 2D array")

    n = matrix.shape[0]
    if n < 2 or (n & (n - 1)) != 0:
        raise ValueError("Matrix size must be a power of two (2, 4, 8, 16, ...)")

    h, w = img.shape
    tiles_y = (h + n - 1) // n
    tiles_x = (w + n - 1) // n

    gray_img = grayscale(img, dtype=dtype)  # float32 in gray domain
    thresh = np.tile(matrix, (tiles_y, tiles_x))[:h, :w] * 255.0
    dithered_u8 = np.where(gray_img >= thresh, 255, 0).astype(np.uint8)
    return binarize(dithered_u8, dtype)
