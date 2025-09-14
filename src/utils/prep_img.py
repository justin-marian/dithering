# -*- coding: utf-8 -*-
"""Image preparation helpers.

Utilities to:
- Convert arbitrary arrays to uint8 for saving (`to_uint8_image`)
- Normalize/interpret requested output dtype semantics (`process_dtype_arg`)
- Prepare RGB images as float32 in [0..255] with metadata (`tuple_prepare_img`)
"""

from __future__ import annotations

from typing import Literal, Tuple, Union

import numpy as np


def to_uint8_image(arr: np.ndarray) -> np.ndarray:
    """
    Convert an array to a uint8 image suitable for saving.

    Handles common cases, including:
    - bool arrays: {False, True} → {0, 255}
    - uint8 arrays with {0, 1}: scaled to {0, 255}
    - float arrays in [0..1] or [0..255]: scaled/clipped appropriately
    - other integer arrays: clipped to [0..255]

    Parameters
    ----------
    arr : np.ndarray
        Input array of shape (H, W) or (H, W, 3/4).

    Returns
    -------
    np.ndarray
        Array with dtype uint8.
    """
    a = np.asarray(arr)

    # bool: map {False, True} to {0, 255}
    if a.dtype == np.bool_:
        return a.astype(np.uint8) * 255

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
    dtype: Union[Literal["u8"], Literal["f32"], np.dtype, type]
) -> Tuple[type, bool, Tuple[float, float]]:
    """
    Interpret an output dtype specifier.

    Returns a tuple (out_dtype, normalize_input, output_range):
      - 'u8'              → (np.uint8, True, (0, 255))
      - 'f32'             → (np.float32, False, (0.0, 1.0))
      - numpy integer     → (dtype, True, (0, 255))
      - numpy float       → (dtype or np.float32, False, (0.0, 1.0))

    Parameters
    ----------
    dtype : {"u8","f32"} | np.dtype | type
        Desired output dtype semantics.

    Returns
    -------
    Tuple[type, bool, Tuple[float, float]]
        (out_dtype, normalize_input, output_range)

    Raises
    ------
    TypeError
        If `dtype` cannot be interpreted.
    """
    if dtype == "u8":
        return np.uint8, True, (0, 255)
    if dtype == "f32":
        return np.float32, False, (0.0, 1.0)

    if isinstance(dtype, (np.dtype, type)):
        dt = np.dtype(dtype)
        if np.issubdtype(dt, np.integer):
            return dt.type, True, (0, 255)
        if np.issubdtype(dt, np.floating):
            return (np.float32 if dt == np.float64 else dt.type), False, (0.0, 1.0)

    raise TypeError(f"Unsupported dtype: {dtype!r}")


def tuple_prepare_img(
    img: np.ndarray,
    dtype: Union[Literal["u8"], Literal["f32"], np.dtype, type],
) -> Tuple[np.ndarray, type, bool, Tuple[float, float]]:
    """
    Prepare an RGB(A) image for processing.

    Converts input to float32 in [0..255] and returns:
      (img_arr, out_dtype, normalize_input, output_range)

    Parameters
    ----------
    img : np.ndarray
        Input image of shape (H, W, 3/4), any numeric dtype.
    dtype : {"u8","f32"} | np.dtype | type
        Desired output dtype semantics.

    Returns
    -------
    Tuple[np.ndarray, type, bool, Tuple[float, float]]
        - img_arr : float32, shape (H, W, 3), values in [0..255]
        - out_dtype : resolved output dtype (type)
        - normalize_input : whether integer input was normalized to [0..255]
        - output_range : (min, max) range for the output dtype

    Raises
    ------
    ValueError
        If the input image is not (H, W, 3/4).
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
            a_f *= 255.0 / in_max
    elif np.issubdtype(a.dtype, np.floating):
        a_f = a.astype(np.float32)
        if a_f.size:
            vmax = float(np.nanmax(a_f))
            if vmax <= 1.0:
                a_f *= 255.0
    else:
        # Fallback: treat as bytes-like
        a_f = a.astype(np.float32)

    # Clip and ensure contiguous float32
    a_f = np.clip(a_f, 0.0, 255.0).astype(np.float32, copy=False)
    a_f = np.ascontiguousarray(a_f)

    # Resolve desired output dtype / range semantics
    out_dtype, is_int_out, out_range = process_dtype_arg(dtype)
    return a_f, out_dtype, is_int_out, out_range
