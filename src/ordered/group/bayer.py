# -*- coding: utf-8 -*-

from __future__ import annotations

from typing import Literal, Union

import numpy as np

from ...utils import binarize, grayscale

# Order Dithering Matrices - Bayer matrices for ordered dithering
BAYER_2: np.ndarray = (1 / 4) * np.array([
    [0, 2],
    [3, 1]
], dtype=np.float32)

BAYER_4: np.ndarray = (1 / 16) * np.array([
    [ 0,  8,  2, 10],
    [12,  4, 14,  6],
    [ 3, 11,  1,  9],
    [15,  7, 13,  5]
], dtype=np.float32)

BAYER_8: np.ndarray = (1 / 64) * np.array([
    [ 0, 32,  8, 40,  2, 34, 10, 42],
    [48, 16, 56, 24, 50, 18, 58, 26],
    [12, 44,  4, 36, 14, 46,  6, 38],
    [60, 28, 52, 20, 62, 30, 54, 22],
    [ 3, 35, 11, 43,  1, 33,  9, 41],
    [51, 19, 59, 27, 49, 17, 57, 25],
    [15, 47,  7, 39, 13, 45,  5, 37],
    [63, 31, 55, 23, 61, 29, 53, 21]
], dtype=np.float32)

def bayer_matrix(n: int) -> np.ndarray:
    if n == 2:
        return BAYER_2
    elif n == 4:
        return BAYER_4
    elif n == 8:
        return BAYER_8
    
    if n < 2 or (n & (n - 1)) != 0:
        raise ValueError("Bayer matrix size must be a power of two (2, 4, 8, 16, ...)")
    
    # Recursive construction for larger sizes
    smaller = bayer_matrix(n // 2)
    top = np.hstack((4 * smaller + 0, 4 * smaller + 2))
    bottom = np.hstack((4 * smaller + 3, 4 * smaller + 1))
    return (1 / (n * n)) * np.vstack((top, bottom))

def bayer_bw(
    img: np.ndarray,
    *,
    matrix: np.ndarray = BAYER_8,
    dtype: Union[Literal['u8'], Literal['f32'], np.dtype, type] = 'u8'
) -> np.ndarray:
    """
    Ordered dithering (Bayer) to 1-bit.

    - Accepts grayscale or RGB/RGBA inputs, uint8 [0..255] or float [0..1].
    - Uses utils.grayscale() to normalize to float32 [0..255].
    - Vectorized (no Python loops).
    - Returns in requested dtype:
        * 'u8'/uint8  -> {0,255}
        * float dtypes -> {0.0,1.0}
        * other ints  -> {0,max(dtype)}
    """
    n = matrix.shape[0]

    if img.ndim != 2:
        raise ValueError("Input image must be grayscale (HxW)")
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Matrix must be a square 2D array")
    if (n & (n - 1)) != 0 or n < 2:
        raise ValueError("Matrix size must be a power of two (2, 4, 8, ...)")

    height, width = img.shape
    height_blocks = (height + n - 1) // n
    width_blocks = (width + n - 1) // n
    # Tile the Bayer matrix to cover the image area
    gray_img = grayscale(img, dtype=dtype)
    thresh = np.tile(matrix, ((height_blocks, width_blocks)))[:height, :width] * 255.0
    # Normalize threshold to [0,255]
    dithered_img = np.where(gray_img >= thresh, 255, 0).astype(np.uint8)
    return binarize(dithered_img, dtype)
