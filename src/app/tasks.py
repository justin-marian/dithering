# -*- coding: utf-8 -*-

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import numpy as np

from ..adaptive_diffusion.adaptive_dif import adaptive_bw
from ..error_diffusion.err_diff import error_diff_bw
from ..multi_level.palette import palette_bw
from ..naive.threshold import threshold_bw
from ..ordered.ordered import ordered_bw
from ..utils.grayscale import grayscale
from .visualize import show_images

THIS_FILE = Path(__file__).resolve()
ROOT = THIS_FILE.parents[2]
OUT  = ROOT / "output"

def task_adaptive_diffusion(
    img: np.ndarray,
    img_name: str,
    *,
    serpentine: bool = True,
    save: bool = False,
    outdir: Path = ROOT / "output",
) -> Tuple[List[np.ndarray], List[str]]:
    """
    Adaptive diffusion dithering using Ostromoukhov and Zhou-Fang methods.
    Given a color image, applies both dithering methods and returns the results.
    """
    outs, names = [], []

    d_ostro = adaptive_bw(
        img,
        method="ostromoukhov",
        dtype="u8",
        threshold=128,
        serpentine=serpentine,
    )
    outs.append(d_ostro)
    names.append("ostromoukhov")

    d_zf = adaptive_bw(
        img,
        method="zhou_fang",
        dtype="u8",
        serpentine=serpentine,
        noise_scale=1.0,
        seed=42,
    )
    outs.append(d_zf)
    names.append("zhou_fang")

    show_images(outs, names, save=save, outdir=outdir, stem=img_name, task="adaptive_diffusion")
    return outs, names

def task_error_diffusion(
    img: np.ndarray,
    img_name: str,
    *,
    kernels: List[str],
    threshold=128,
    serpentine=True,
    save=False,
    outdir: Path = ROOT / "output"
) -> Tuple[List[np.ndarray], List[str]]:
    """
    Error diffusion dithering using specified kernels.
    Given a color image and a list of kernel names, applies error diffusion
    dithering with each kernel and returns the results.
    """
    outs, names = [], []
    for kname in kernels:
        d_img = error_diff_bw(
            img, dtype="u8", kernel_type=kname,
            threshold=threshold, serpentine=serpentine
        )
        outs.append(d_img) 
        names.append(kname)
    show_images(outs, names, save=save, outdir=outdir, stem=img_name, task="error_diffusion")
    return outs, names

def task_naive(
    img: np.ndarray,
    img_name: str,
    *,
    threshold: int | float = 128,
    save: bool = False,
    outdir: Path = OUT,
) -> Tuple[List[np.ndarray], List[str]]:
    """
    Naive dithering using various thresholding methods.
    Applies global, mean, percentile, and Otsu thresholding methods to a color image
    and returns the resulting binary images.
    """
    outs: List[np.ndarray] = []
    names: List[str] = []

    cfgs = [
        ("global",     dict(threshold=threshold)),
        ("mean",       dict()),
        ("percentile", dict(percentile=50.0)),
        ("otsu",       dict()),
    ]

    for method, kwargs in cfgs:
        d = threshold_bw(img, dtype="u8", method=method, **kwargs)
        outs.append(d)
        names.append(f"naive_{method}")

    show_images(outs, names, save=save, outdir=outdir, stem=img_name, task="naive")
    return outs, names

def task_random(
    img: np.ndarray,
    img_name: str,
    *,
    save=False,
    outdir: Path = ROOT / "output"
) -> Tuple[List[np.ndarray], List[str]]:
    """
    Noise dithering: per-pixel random threshold in [0..255].
    Given a color image, applies random threshold dithering and returns the result.
    Returns a single binary image.
    """
    g = grayscale(img, 'u8')
    rng = np.random.default_rng()
    noise = rng.integers(0, 256, size=g.shape, dtype=np.uint16)

    d = np.where(g >= noise, 255, 0).astype(np.uint8)

    outs = [d]
    names = ["random_threshold"]
    show_images(outs, names, save=save, outdir=outdir, stem=img_name, task="random")
    return outs, names

def task_ordered(
    img: np.ndarray,
    img_name: str,
    *,
    n: int = 8,
    save: bool = False,
    outdir: Path = OUT,
) -> Tuple[List[np.ndarray], List[str]]:
    """
    Ordered dithering showcase:
      - Bayer matrix of size n (power of two)
      - Halftone (spot function) with tile size n @ 45 degrees
    Given a color image, applies ordered dithering using both Bayer and halftone
    methods and returns the results.
    """
    gray = grayscale(img, 'u8')

    d_bayer = ordered_bw(gray, kind="bayer", n=n, dtype="u8")
    d_half  = ordered_bw(gray, kind="halftone", n=n, angle_deg=45.0,
                         spot="cos+cos", dtype="u8")

    outs  = [d_bayer, d_half]
    names = [f"bayer_{n}", f"halftone_{n}px_45deg"]

    show_images(outs, names, save=save, outdir=outdir, stem=img_name, task="ordered")
    return outs, names

def task_multi_level(
    img: np.ndarray,
    img_name: str,
    *,
    levels: int = 4,
    type: str = "floyd_steinberg",
    save: bool = False,
    outdir: Path = OUT,
) -> Tuple[List[np.ndarray], List[str]]:
    """
    Multi-level grayscale dithering using the palette-based error diffusion.
    Builds an evenly spaced gray palette of `levels` entries and applies
    a palette-based error diffusion dithering with the specified kernel.
    """
    if levels < 2:
        raise ValueError("levels must be >= 2")

    if levels == 2:
        palette = [(0, 0, 0), (255, 255, 255)]
    else:
        palette = [
            (v, v, v)
            for v in (np.rint(np.linspace(0, 255, levels)).astype(np.uint8).tolist())
        ]

    d_img = palette_bw(
        img,
        palette=palette,
        kernel_type=type,
        serpentine=True,
    )

    outs = [d_img]
    names = [f"multi_level_{levels}"]
    show_images(outs, names, save=save, outdir=outdir, stem=img_name, task="multi_level")
    return outs, names
