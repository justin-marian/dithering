# -*- coding: utf-8 -*-
"""
Dithering CLI entry point.

Loads (or randomly selects) a demo image and dispatches to one of the dithering
tasks chosen via CLI args (naive, ordered, random, error/adaptive diffusion, multi-level).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from skimage import io
from skimage.util import img_as_ubyte
from src.app.cli import check_kernels, parse_args
from src.app.tasks import (
    task_adaptive_diffusion,
    task_error_diffusion,
    task_multi_level,
    task_naive,
    task_ordered,
    task_random,
)

ROOT   = Path(__file__).resolve().parent
ASSETS = ROOT / "assets"


def load_demo_image(seed: int | None = None) -> tuple[np.ndarray, str]:
    """Return an RGB uint8 image and a short name.

    Picks a random image from ./assets (png/jpg/bmp/tif/webp), or falls back to
    :mod:`skimage.data`'s astronaut if no local images are found. Ensures RGB with dtype=uint8.

    Parameters
    ----------
    seed:
        Optional seed to make the random choice deterministic.

    Returns
    -------
    (image, name):
        `image` is np.ndarray (H, W, 3), dtype uint8. `name` is a short identifier.
    """
    rng = np.random.default_rng(seed)
    candidates: list[Path] = []
    if ASSETS.exists():
        for pat in ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff", "*.webp"):
            candidates.extend(ASSETS.glob(pat))

    if candidates:
        path = candidates[int(rng.integers(0, len(candidates)))]
        img = io.imread(str(path))
        name = path.stem
    else:
        from skimage import data
        img = data.astronaut()
        name = "astronaut"

    # Normalize to RGB uint8
    if img.ndim == 2:
        img = np.stack([img, img, img], axis=-1)
    if img.shape[-1] == 4:
        img = img[..., :3]
    if img.dtype != np.uint8:
        img = img_as_ubyte(img)
    return img, name


def main() -> None:
    """CLI entry: parse args, load image, and run the requested dithering task."""
    args = parse_args()
    img, img_name = load_demo_image()

    outdir = ROOT / "output"
    if args.save:
        outdir.mkdir(parents=True, exist_ok=True)

    serp = not args.no_serpentine

    def preparse_kernels() -> list[str]:
        """Split and validate the `--kernels` CSV argument using `check_kernels`."""
        items = [s.strip() for s in args.kernels.split(",") if s.strip()]
        return check_kernels(items)

    dispatch = {
        "adaptive_diffusion": lambda: task_adaptive_diffusion(
            img, serpentine=serp, save=args.save, outdir=outdir, img_name=img_name
        ),
        "error_diffusion": lambda: task_error_diffusion(
            img,
            kernels=preparse_kernels(),
            threshold=args.threshold,
            serpentine=serp,
            save=args.save,
            outdir=outdir,
            img_name=img_name,
        ),
        "naive": lambda: task_naive(
            img, threshold=args.threshold, save=args.save, outdir=outdir, img_name=img_name
        ),
        "ordered": lambda: task_ordered(
            img, n=args.bayer_n, save=args.save, outdir=outdir, img_name=img_name
        ),
        "random": lambda: task_random(img, save=args.save, outdir=outdir, img_name=img_name),
        "multi_level": lambda: task_multi_level(
            img, levels=args.levels, save=args.save, outdir=outdir, img_name=img_name
        ),
    }

    run = dispatch.get(args.task)
    if run is None:
        raise SystemExit(f"Unknown task: {args.task}")
    run()

if __name__ == "__main__":
    main()
