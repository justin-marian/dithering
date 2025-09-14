# -*- coding: utf-8 -*-
"""Task runners for the dithering demo.

Each task takes an input image and returns a list of output images plus their names.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import numpy as np

from ..adaptive_diffusion import adaptive_diff_bw
from ..error_diffusion import error_diff_bw
from ..multi_level import palette_bw
from ..naive import threshold_bw
from ..ordered import ordered_bw
from ..utils import grayscale
from .visualize import show_images

THIS_FILE = Path(__file__).resolve()
ROOT = THIS_FILE.parents[2]
OUT = ROOT / "output"


def task_adaptive_diffusion(
    img: np.ndarray,
    img_name: str,
    *,
    serpentine: bool = True,
    save: bool = False,
    outdir: Path = ROOT / "output",
) -> Tuple[List[np.ndarray], List[str]]:
    """Adaptive diffusion dithering: Ostromoukhov and Zhou-Fang."""
    outs, names = [], []

    d_ostro = adaptive_diff_bw(
        img,
        method="ostromoukhov",
        dtype="u8",
        threshold=128,
        serpentine=serpentine,
    )
    outs.append(d_ostro)
    names.append("ostromoukhov")

    d_zf = adaptive_diff_bw(
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
    outdir: Path = ROOT / "output",
) -> Tuple[List[np.ndarray], List[str]]:
    """Error diffusion dithering with the specified kernels."""
    outs, names = [], []
    for kname in kernels:
        d_img = error_diff_bw(
            img,
            dtype="u8",
            kernel_type=kname,
            threshold=threshold,
            serpentine=serpentine,
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
    """Naive thresholding variants: global, mean, percentile, Otsu."""
    outs: List[np.ndarray] = []
    names: List[str] = []

    cfgs = [
        ("global", {"threshold": threshold}),
        ("mean", {}),
        ("percentile", {"percentile": 50.0}),
        ("otsu", {}),
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
    save: bool = False,
    outdir: Path = ROOT / "output",
) -> Tuple[List[np.ndarray], List[str]]:
    """Noise dithering: per-pixel random threshold in [0..255]."""
    g = grayscale(img, "u8")
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
    """Ordered dithering showcase: Bayer (n) and halftone (n @ 45°)."""
    gray = grayscale(img, "u8")

    d_bayer = ordered_bw(gray, kind="bayer", n=n, dtype="u8")
    d_half = ordered_bw(
        gray,
        kind="halftone",
        n=n,
        angle_deg=45.0,
        spot="cos+cos",
        dtype="u8",
    )

    outs = [d_bayer, d_half]
    names = [f"bayer_{n}", f"halftone_{n}px_45deg"]

    show_images(outs, names, save=save, outdir=outdir, stem=img_name, task="ordered")
    return outs, names


def task_multi_level(
    img: np.ndarray,
    img_name: str,
    *,
    levels: int = 4,
    kernel_type: str = "floyd_steinberg",
    save: bool = False,
    outdir: Path = OUT,
) -> Tuple[List[np.ndarray], List[str]]:
    """Multi-level grayscale dithering using palette-based error diffusion."""
    if levels < 2:
        raise ValueError("levels must be >= 2")

    if levels == 2:
        palette = [(0, 0, 0), (255, 255, 255)]
    else:
        values = np.rint(np.linspace(0, 255, levels)).astype(np.uint8)
        palette = [(int(v), int(v), int(v)) for v in values]

    d_img = palette_bw(
        img,
        palette=palette,
        kernel_type=kernel_type,
        serpentine=True,
    )

    outs = [d_img]
    names = [f"multi_level_{levels}"]
    show_images(outs, names, save=save, outdir=outdir, stem=img_name, task="multi_level")
    return outs, names
