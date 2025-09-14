# -*- coding: utf-8 -*-
"""Packaging configuration for the dithering project."""

from __future__ import annotations

from pathlib import Path

from setuptools import find_packages, setup

ROOT = Path(__file__).parent
README = (ROOT / "README.md").read_text(encoding="utf-8") if (ROOT / "README.md").exists() else ""

setup(
    name="dithering",
    version="0.1.0",
    description=(
        "Playground for ordered/adaptive/error-diffusion/"
        "multi-level/naive/ordered/random dithering algorithms"
    ),
    long_description=README,
    long_description_content_type="text/markdown",
    author="justin-mp",
    author_email="pmarianjustin@gmail.com",
    license="UNLICENSE",
    python_requires=">=3.10",

    # Discover packages under src/
    package_dir={"": "src"},
    packages=find_packages(
        where="src",
        include=[
            "app*",
            "adaptive_diffusion",
            "error_diffusion",
            "multi_level",
            "naive",
            "ordered",
            "random",
            "utils",
        ],
    ),
    include_package_data=True,

    # Root-level CLI module /dither.py
    py_modules=["dither"],

    install_requires=[
        "numpy>=2.0",
        "scipy>=1.10",
        "scikit-image>=0.25",
        "matplotlib>=3.8",
        "pillow>=11.3.0",
        "imageio>=2.31",
        "networkx>=3.0",
        "python-dateutil>=2.8.2",
        "packaging>=25.0",
    ],

    extras_require={
        "dev": [
            "ruff>=0.5",
            "black>=25.1.0",
            "mypy>=1.7",
            "build>=1.2.1",
        ]
    },

    # Entry point for `dither` CLI command
    entry_points={
        "console_scripts": [
            "dither=dither:main",
        ]
    },

    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: Public Domain",
        "Operating System :: OS Independent",
    ],
    project_urls={
        "Source": "https://github.com/justin-marian/dithering",
    },
)

if __name__ == "__main__":
    setup()
