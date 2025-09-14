# -*- coding: utf-8 -*-

from __future__ import annotations

from typing import Literal, Union

import numpy as np

from ..utils.prep_img import process_dtype_arg


def to_grayscale(
    img: np.ndarray, 
    dtype: Union[Literal['u8'], Literal['f32'], np.dtype, type] = 'u8'
) -> np.ndarray:
    """
    Convert grayscale/RGB/RGBA to luminance using BT.601.
    Output dtype/range:
      - integer dtype -> full integer range of that dtype
      - floating dtype -> [0.0, 1.0]
    """
    # ITU-R BT.601 luminance coefficients
    luminance = np.array([0.2989, 0.5870, 0.1140], dtype=np.float32)
    output_dtype, is_int_out, _ = process_dtype_arg(dtype)

    a = np.asarray(img)
    if a.ndim == 3 and a.shape[2] == 1:
        a = a[..., 0]

    if np.issubdtype(a.dtype, np.integer):
        in_max = np.iinfo(a.dtype).max
        a_f = a.astype(np.float32) / float(in_max if in_max > 0 else 255.0)
    elif np.issubdtype(a.dtype, np.floating):
        a_f = a.astype(np.float32)
        if a_f.size:
            vmax = float(np.nanmax(a_f))
            if vmax > 1.0:
                a_f = a_f / 255.0
    else:
        a_f = a.astype(np.float32) / 255.0

    if a_f.ndim == 2:
        y01 = np.clip(a_f, 0.0, 1.0)
    elif a_f.ndim == 3 and a_f.shape[2] >= 3:
        rgb = a_f[..., :3]
        y01 = np.clip(rgb @ luminance, 0.0, 1.0)
    else:
        raise ValueError(
            f"Unsupported image format: shape {a_f.shape}. "
            "Expected 2D grayscale, HxWx3 RGB, or HxWx4 RGBA."
        )

    if is_int_out:
        out_max = np.iinfo(output_dtype).max
        return np.clip(np.rint(y01 * out_max), 0, out_max).astype(output_dtype)
    else:
        return np.clip(y01, 0.0, 1.0).astype(output_dtype)

def to_grayscale_u8(img: np.ndarray) -> np.ndarray:
    """Wrapper for to_grayscale with uint8 output."""
    return to_grayscale(img, dtype='u8')

def to_grayscale_f32(img: np.ndarray) -> np.ndarray:
    """Wrapper for to_grayscale with float32 output."""
    return to_grayscale(img, dtype='f32')

def grayscale(
    img: np.ndarray,
    dtype: Union[Literal['u8'], Literal['f32'], np.dtype, type] = 'u8'
) -> np.ndarray:
    """
    Convert image to grayscale float32 [0..255] for internal processing.
    Input can be grayscale/RGB/RGBA with uint8 [0..255] or float32 [0.0..1.0] values.
    Output is always float32 [0..255].
    """
    checker_type = isinstance(dtype, (np.dtype, type))
    checker_float = checker_type and np.issubdtype(np.dtype(dtype), np.floating)
    if dtype == 'f32' or (checker_type and checker_float):
        gray_img = to_grayscale_f32(img)
        gray_img = np.clip(np.round(gray_img * 255.0), 0, 255).astype(np.uint8)
    else:
        gray_img = to_grayscale_u8(img)
        gray_img = gray_img.astype(np.float32)
    return gray_img

def map_threshold_graydomain(
    threshold: Union[int, float],
    dtype: Union[Literal['u8'], Literal['f32'], np.dtype, type]
) -> int:
    """
    Map threshold to what the kernel expects (uint8 input domain 0..255).
    - For integer dtypes, threshold is in [0..255].
    - For float dtypes, threshold is in [0.0..1.0] and mapped to [0..255].
    """
    checker_type = isinstance(dtype, (np.dtype, type))
    checker_float = checker_type and np.issubdtype(np.dtype(dtype), np.floating)
    if dtype == 'f32' or (checker_type and checker_float):
        thr_u8 = int(round(float(threshold) * 255.0))
    else:
        thr_u8 = int(round(float(threshold)))
    return max(0, min(255, thr_u8))

def binarize(
    bw_u8: np.ndarray,
    dtype: Union[Literal['u8'], Literal['f32'], np.dtype, type]
) -> np.ndarray:
    """
    Map {0,255} to requested dtype:
      - float-like: {0.0,1.0}
      - uint8: {0,255}
      - other int dtype: {0,max(dtype)}
    """
    # map to requested dtype
    checker_type = isinstance(dtype, (np.dtype, type))
    checker_float = np.issubdtype(np.dtype(dtype), np.floating)
    checker_int = np.issubdtype(np.dtype(dtype), np.integer)

    if dtype == 'f32' or (checker_type and checker_float):
        out_dtype = np.float32 if dtype == 'f32' else np.dtype(dtype).type
        return (bw_u8.astype(np.float32) / 255.0).astype(out_dtype)

    # transform to integer dtype
    if dtype == 'u8' or (checker_type and checker_int):
        out_dtype = np.uint8 if dtype == 'u8' else np.dtype(dtype).type
        return (bw_u8.astype(out_dtype) // 255).astype(out_dtype)

    # other integer dtype
    out_dtype = np.dtype(dtype).type
    maxv = np.iinfo(out_dtype).max
    return ((bw_u8.astype(out_dtype) // 255) * maxv).astype(out_dtype)
