# -*- coding: utf-8 -*-

from __future__ import annotations

from typing import Literal, Tuple, Union

import numpy as np


def to_uint8_image(arr: np.ndarray) -> np.ndarray:
    """
    Return a uint8 image (H,W) or (H,W,3/4) suitable for saving.
    Handles all common cases robustly, including uint8 arrays with {0,1}.
    """
    a = np.asarray(arr)

    # bool: map {False,True} to {0,255}
    if a.dtype == np.bool_:
        return (a.astype(np.uint8) * 255)

    # uint8: if it looks like {0,1}, scale to {0,255}
    if a.dtype == np.uint8:
        amax = int(a.max()) if a.size else 0
        if amax <= 1:
            return (a * 255).astype(np.uint8)
        return a

    # Floats: either [0..1] or [0..255]; scale appropriately
    if np.issubdtype(a.dtype, np.floating):
        amax = float(a.max()) if a.size else 0.0
        if amax <= 1.0:
            a = np.clip(a, 0.0, 1.0) * 255.0
        else:
            a = np.clip(a, 0.0, 255.0)
        return a.round().astype(np.uint8)

    # Other ints: clip to [0,255]
    return np.clip(a, 0, 255).astype(np.uint8)

def process_dtype_arg(
    dtype: Union[Literal['u8'], Literal['f32'], np.dtype, type]
) -> Tuple[type, bool, Tuple[float, float]]:
    """
    Returns img_arr tuple of (numpy dtype, normalize_input, output_range)
    - 'u8' -> (np.uint8, True, (0, 255))
    - 'f32' -> (np.float32, False, (0.0, 1.0))
    - numpy integer dtype -> (dtype, True, (0, 255))
    - numpy float dtype -> (dtype, False, (0.0, 1.0))
    """
    if dtype == 'u8':
        return np.uint8, True, (0, 255)
    if dtype == 'f32':
        return np.float32, False, (0.0, 1.0)
    
    if isinstance(dtype, (np.dtype, type)):
        dt = np.dtype(dtype)
        if np.issubdtype(dt, np.integer):
            return dt.type, True, (0, 255)
        if np.issubdtype(dt, np.floating):
            return np.float32 if dt == np.float64 else dt.type, False, (0.0, 1.0)

    return exit(f"Unsupported dtype: {dtype}")

def tuple_prepare_img(
    img: np.ndarray,
    dtype: Union[Literal['u8'], Literal['f32'], np.dtype, type]
) -> Tuple[np.ndarray, type, bool, Tuple[float, float]]:
    """
    Prepare image array and return (img_arr, out_dtype, normalize_input, output_range).
    - img_arr: ndarray of shape (H, W) or (H, W, C) with dtype np.uint8 or np.float32
    - out_dtype: requested output dtype
    - normalize_input: whether input was integer and normalized to [0..255]
    - output_range: (min, max) of output dtype
    """
    if img.ndim != 3 or img.shape[2] < 3:
        raise ValueError("Input image must be HxWx3 or HxWx4.")

    # Keep RGB, drop alpha if present
    a = np.asarray(img)[..., :3]

    # Convert to float32 in [0..255]
    if np.issubdtype(a.dtype, np.integer):
        in_max = float(np.iinfo(a.dtype).max)
        a_f = a.astype(np.float32)
        if in_max != 255.0:
            a_f *= (255.0 / in_max)
    elif np.issubdtype(a.dtype, np.floating):
        a_f = a.astype(np.float32)
        if a_f.size:
            vmax = float(np.nanmax(a_f))
            if vmax <= 1.0:
                a_f *= 255.0
    else:
        # Fallback: treat as bytes-like
        a_f = a.astype(np.float32)

    # Clip just in case and ensure contiguous float32
    a_f = np.clip(a_f, 0.0, 255.0).astype(np.float32, copy=False)
    a_f = np.ascontiguousarray(a_f)

    # Resolve desired output dtype / range semantics
    out_dtype, is_int_out, out_range = process_dtype_arg(dtype)
    return a_f, out_dtype, is_int_out, out_range
