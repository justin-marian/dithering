# dithering

Collection of classic and modern dithering algorithms that convert grayscale/color images to black & white using various error distribution techniques.

**Algorithms:** 

- naïve threshold
- ordered (Bayer & halftone)
- random
- error diffusion (Floyd-Steinberg, JJN, Stucki, Sierra, Burkes, Atkinson…)
- adaptive diffusion (Ostromoukhov, Zhou–Fang) 
- multi-level 
- palette diffusion

Uses random image from `./assets/` or falls back to `skimage.data.astronaut()`. 
Results preview and/or save to `./outputs/`.

## Install

```bash
python3 -m pip install -e .  # Python 3.10+
```

## Usage

```bash
# Show options and available kernels
dither --help

# Error diffusion comparison
dither --task error_diffusion --kernels FS,JJN,stucki --save

# Ordered dithering
dither --task ordered --bayer-n 8 --save

# Adaptive methods  
dither --task adaptive_diffusion --save

# Simple thresholding
dither --task naive --threshold 128 --save

# Multi-level grayscale
dither --task multi_level --levels 6 --save
```

## Implemented Methods

• **Naïve:** global, mean, percentile, Otsu thresholding
• **Ordered:** Bayer matrices (2×2 to 16×16), halftone spot functions  
• **Random:** per-pixel random threshold
• **Error diffusion:** Floyd–Steinberg, Jarvis–Judice–Ninke, Stucki, Sierra family, Burkes, Atkinson, Stevenson–Arce
• **Adaptive:** Ostromoukhov (content-aware weights), Zhou–Fang (threshold jitter)
• **Multi-level:** custom palette diffusion with any number of gray levels

## Key Options

• `--task {naive,ordered,random,error_diffusion,adaptive_diffusion,multi_level}`
• `--kernels <list>` - error diffusion kernels (e.g., FS,JJN,stucki)
• `--threshold <0-255>` - threshold value
• `--bayer-n {2,4,8,16}` - Bayer matrix size
• `--levels <int>` - number of gray levels for multi-level
• `--save` - write results to ./outputs/
• `--no-serpentine` - disable alternating scan

## Library API

```python
from naive.threshold import threshold_bw
from error_diffusion.err_diff import error_diff_bw
from ordered.ordered import ordered_bw

# Apply different algorithms
result = threshold_bw(img, method="otsu")
result = error_diff_bw(img, kernel_type="floyd_steinberg") 
result = ordered_bw(img, kind="bayer", n=8)
```

All functions return grayscale uint8 images (0/255), except multi-level which returns RGB.

### Data Files
```
src/adaptive_diffusion/kernels/data/
├── weights_ostromoukhov.txt    # 256 × 3 matrix [wR, wDL, wD]
└── strengths_zhou_fang.txt     # Grid lookup for jitter strength
```

### Generated Outputs

| Method | Algorithm | File |
|--------|-----------|------|
| **Adaptive Diffusion** | Ostromoukhov | `lenna_adaptive_diffusion_ostromoukhov.png` |
| **Adaptive Diffusion** | Zhou-Fang | `lenna_adaptive_diffusion_zhou_fang.png` |
| **Error Diffusion** | Atkinson | `lenna_error_diffusion_atkinson.png` |
| **Error Diffusion** | Burkes | `lenna_error_diffusion_burkes.png` |
| **Error Diffusion** | Floyd-Steinberg | `lenna_error_diffusion_floyd_steinberg.png` |
| **Error Diffusion** | Jarvis-Judice-Ninke | `lenna_error_diffusion_jarvis_judice_ninke.png` |
| **Error Diffusion** | Sierra | `lenna_error_diffusion_sierra.png` |
| **Error Diffusion** | Stevenson-Arce | `lenna_error_diffusion_stevenson_arce.png` |
| **Error Diffusion** | Stucki | `lenna_error_diffusion_stucki.png` |
| **Error Diffusion** | Two-row Sierra | `lenna_error_diffusion_two_row_sierra.png` |
| **Multi-level** | 4 levels | `lenna_multi_level_multi_level_4.png` |
| **Naïve** | Global | `lenna_naive_naive_global.png` |
| **Naïve** | Mean | `lenna_naive_naive_mean.png` |
| **Naïve** | Otsu | `lenna_naive_naive_otsu.png` |
| **Naïve** | Percentile | `lenna_naive_naive_percentile.png` |
| **Ordered** | Bayer 8×8 | `lenna_ordered_bayer_8.png` |
| **Ordered** | Halftone 8px 45deg | `lenna_ordered_halftone_8px_45deg.png` |
| **Random** | Random threshold | `lenna_random_random_threshold.png` |

**Naming pattern:** `{image}_{method}_{algorithm}[_parameters].png`

**License:** UNLICENSE (public domain)
