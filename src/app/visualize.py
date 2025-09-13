# -*- coding: utf-8 -*-

from __future__ import annotations

import io
import math
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from ..utils.prep_img import to_uint8_image


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

def show_images(
    images: List[np.ndarray],
    names: List[str],
    task: str,
    *,
    save: bool,
    outdir: Path,
    stem: str
):
    """Display a list of images in a grid and optionally save them."""
    n = len(images)
    cols = min(4, n)
    rows = math.ceil(n / cols)

    _, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    axes = np.atleast_1d(axes).ravel().tolist()
    plt.suptitle(f"Task: {task.replace('_', ' ')}", fontsize=16)

    for i, (im, name) in enumerate(zip(images, names)):
        ax = axes[i]
        if im.ndim == 2:
            im_max = float(np.max(im)) if im.size else 1.0
            vmax = 1.0 if im.dtype.kind in "fc" or im_max <= 1.0 else 255.0
            ax.imshow(im, cmap="gray", vmin=0.0, vmax=vmax)
        else:
            ax.imshow(im)
        ax.set_title(name.replace("_", " "))
        ax.axis("off")
        if save:
            png_data = png_bytes(im)
            outpath = outdir / f"{stem}_{task}_{name}.png"
            with open(outpath, "wb") as f:
                f.write(png_data)
            print(f"Saved: {outpath}")

    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    plt.show()
