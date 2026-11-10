# Homework 1 - Image Enhancement

Classical image enhancement pipeline (no deep learning) for the Computer Vision course, Fall 2026.

## Requirements

- Python 3.10 or newer
- Dependencies: numpy, opencv-python, matplotlib, scikit-image (only for SSIM)

```
pip install -r requirements.txt
```

## How to run

Put the original images (.jpg or .png) in `images/original/`, then run:

```
python main.py
```

The script creates the degraded images, applies all the methods, computes the metrics and saves everything.

## Degradation

Every original image is degraded in the same way:

- contrast reduced around the mean intensity: `I' = mean + 0.5 * (I - mean)`
- gaussian noise added with sigma = 10 (0-255 scale)
- fixed random seed = 42 (images are processed in alphabetical order)

The original images are used as references for the metrics. In a real scenario a clean reference is usually not available.

## Methods

1. **Baseline**: degraded image, no processing
2. **Contrast stretching**: linear, per channel (B, G, R), between the 1st and 99th percentile
3. **Histogram equalization**: on the L channel of CIE-Lab
4. **NLM + stretching**: non-local means denoising (h = 10) followed by contrast stretching

## Metrics

- PSNR (dB) with respect to the original
- SSIM (scikit-image)
- MAE (mean absolute error)

## File structure

```
.
├── main.py                 # full pipeline
├── requirements.txt
├── README.md
├── images/
│   ├── original/           # input images
│   └── degraded/           # generated degraded images
└── results/
    ├── <image>_<method>.png    # processed images
    ├── metrics.csv             # metrics for every image and method
    └── figures/                # side-by-side comparisons
```

## Image sources

| File | Source | License |
|------|--------|---------|
| (fill in) | (fill in) | (fill in) |
