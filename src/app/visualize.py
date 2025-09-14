# -*- coding: utf-8 -*-
"""Visualization utilities for dithering outputs.

Provides:
- png_bytes: robust PNG encoder using Pillow
- show_images: grid display + optional saving to disk
"""

from __future__ import annotations

import io
import math
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from utils import to_uint8_image


def png_bytes(im: np.ndarray) -> bytes:
    """Robust PNG encoder using Pillow. Works for grayscale/RGB/RGBA."""
    u8 = to_uint8_image(im)
    if u8.ndim == 2:
        mode = "L"
    elif u8.ndim == 3 and u8.shape[2] == 3:
        mode = "RGB"
    elif u8.ndim == 3 and u8.shape[2] == 4:
        mode = "RGBA"
    else:
        raise ValueError(f"Unsupported image shape for saving: {u8.shape}")

    img = Image.fromarray(u8, mode=mode)
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def save_png(im: np.ndarray, outdir: Path, stem: str, task: str, name: str) -> None:
    """Save a single image as a PNG file."""
    outdir.mkdir(parents=True, exist_ok=True)
    outpath = outdir / f"{stem}_{task}_{name}.png"
    with open(outpath, "wb") as f:
        f.write(png_bytes(im))
    print(f"Saved: {outpath}")
    

def show_images(
    images: List[np.ndarray],
    names: List[str],
    task: str,
    *,
    save: bool,
    outdir: Path,
    stem: str,
) -> None:
    """Display a list of images in a grid and optionally save them."""
    n = len(images)
    cols = min(4, max(1, n))
    rows = max(1, math.ceil(n / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    _ = fig  # avoid unused var if needed in future

    axes = np.atleast_1d(axes).ravel()
    plt.suptitle(f"Task: {task.replace('_', ' ')}", fontsize=16)

    used = min(n, len(axes))
    for idx in range(used):
        im = images[idx]
        name = names[idx] if idx < len(names) else f"img_{idx}"
        ax = axes[idx]

        if im.ndim == 2:
            im_max = float(np.max(im)) if im.size else 1.0
            vmax = 1.0 if im.dtype.kind in "fc" or im_max <= 1.0 else 255.0
            ax.imshow(im, cmap="gray", vmin=0.0, vmax=vmax)
        else:
            ax.imshow(im)

        ax.set_title(name.replace("_", " "))
        ax.axis("off")

        if save:
            save_png(im, outdir, stem, task, name)

    # Turn off any remaining axes if grid > number of images
    for j in range(used, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    plt.show()
