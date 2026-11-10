# Homework 1 - Image Enhancement

Classical image enhancement pipeline (no deep learning) for the Computer Vision course, Fall 2026.

## Requirements

- Python 3.12
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

## Images

Three real color photos from Wikimedia Commons are used (see the table at the end). Before the experiment they were resized to 1600 px on the long side (area interpolation), to keep the files small and NL-means fast. The images in `images/original/` are already resized: these are the exact inputs used for all results.

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

| File | Condition | Source | Author | License |
|------|-----------|--------|--------|---------|
| 20190617_Nebbia_(3) | fog, low contrast | [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:20190617_Nebbia_(3).jpg) | Gb89.2 | CC0 1.0 |
| 500px_photo_(65567999) | backlight, non-uniform illumination | [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:500px_photo_(65567999).jpeg) (imported from 500px) | Carlota Grau | CC BY 3.0 |
| At_Cueva_del_Viento_2023_069 | cave, flashlight, dark areas | [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:At_Cueva_del_Viento_2023_069.jpg) | Mike Peel (www.mikepeel.net) | CC BY-SA 4.0 |

Changes made to the images: resized to 1600 px on the long side, then degraded and processed as described above.

## Report

The PDF report is in the course submission. Results in `results/metrics.csv` are the ones reported there.