"""
Homework 1 - Image Enhancement
Computer Vision, Fall 2026

What this script does:
  1. loads the original images from images/original
  2. creates a degraded version of each one (lower contrast + gaussian noise)
  3. applies 4 methods: baseline, contrast stretching, histogram equalization,
     NL-means denoising + contrast stretching
  4. computes PSNR, SSIM and MAE with respect to the original image
  5. saves processed images, comparison figures and a csv with the metrics

Run with:  python main.py
"""

import os
import csv

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")  # save figures without opening windows
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity



# Parameters 

SEED = 42               # random seed for the noise
CONTRAST_FACTOR = 0.5   # 1 = same contrast, smaller = lower contrast
NOISE_SIGMA = 10        # std of the gaussian noise (0-255 scale)

STRETCH_LOW = 1         # percentiles used in contrast stretching
STRETCH_HIGH = 99

NLM_H = 10              # filter strength for non-local means

INPUT_DIR = "images/original"
DEGRADED_DIR = "images/degraded"
RESULTS_DIR = "results"
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")



# Degradation

def degrade(img, rng):
    """Reduce contrast around the mean and add gaussian noise."""
    img = img.astype(np.float32)
    mean = img.mean()
    low_contrast = mean + CONTRAST_FACTOR * (img - mean)
    noise = rng.normal(0, NOISE_SIGMA, img.shape)
    out = low_contrast + noise
    return np.clip(out, 0, 255).astype(np.uint8)



# Enhancement methods

def baseline(img):
    """Method 1: no processing."""
    return img.copy()


def contrast_stretching(img):
    """
    Method 2: linear contrast stretching, done on each channel (B, G, R)
    separately. We use percentiles instead of min/max so that a few noisy
    pixels do not decide the range. Since every channel gets a different
    mapping, the colors can shift.
    """
    out = np.zeros_like(img)
    for c in range(3):
        ch = img[:, :, c].astype(np.float32)
        lo = np.percentile(ch, STRETCH_LOW)
        hi = np.percentile(ch, STRETCH_HIGH)
        if hi - lo < 1:  # avoid division by zero on flat channels
            hi = lo + 1
        ch = (ch - lo) / (hi - lo) * 255
        out[:, :, c] = np.clip(ch, 0, 255).astype(np.uint8)
    return out


def histogram_equalization(img):
    """
    Method 3: histogram equalization on the L channel of CIE-Lab.
    a and b are not touched, so the hue should stay more or less the same.
    """
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    L, a, b = cv2.split(lab)
    L = cv2.equalizeHist(L)
    lab = cv2.merge([L, a, b])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def denoise_and_stretch(img):
    """
    Method 4 (combination): first remove noise with non-local means, then
    apply contrast stretching. The idea is that stretching after denoising
    does not amplify the noise as much. Downside: NLM can also smooth
    fine details and textures.
    """
    denoised = cv2.fastNlMeansDenoisingColored(img, None, NLM_H, NLM_H, 7, 21)
    return contrast_stretching(denoised)



METHODS = [
    ("baseline", "Baseline (degraded)", baseline),
    ("stretch", "Contrast stretching", contrast_stretching),
    ("histeq", "Hist. equalization (L)", histogram_equalization),
    ("nlm_stretch", "NLM + stretching", denoise_and_stretch),
]



# Metrics

def mse(ref, img):
    diff = ref.astype(np.float64) - img.astype(np.float64)
    return np.mean(diff ** 2)


def psnr(ref, img):
    m = mse(ref, img)
    if m == 0:
        return float("inf")
    return 10 * np.log10(255.0 ** 2 / m)


def ssim(ref, img):
    return structural_similarity(ref, img, channel_axis=2, data_range=255)


def mae(ref, img):
    return np.mean(np.abs(ref.astype(np.float64) - img.astype(np.float64)))



# Figures

def save_figure(name, original, results, scores):
    """One row: original + result of every method, with PSNR/SSIM in the title."""
    n = len(results) + 1
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))

    axes[0].imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    axes[0].set_title("Original (reference)")
    axes[0].axis("off")

    for i, (key, label, _) in enumerate(METHODS):
        ax = axes[i + 1]
        ax.imshow(cv2.cvtColor(results[key], cv2.COLOR_BGR2RGB))
        p, s = scores[key]["psnr"], scores[key]["ssim"]
        ax.set_title(f"{label}\nPSNR {p:.2f} dB | SSIM {s:.3f}")
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, f"{name}_comparison.png"), dpi=150)
    plt.close(fig)



# Main

def main():
    os.makedirs(DEGRADED_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    files = sorted(f for f in os.listdir(INPUT_DIR)
                   if f.lower().endswith((".jpg", ".jpeg", ".png")))
    if len(files) == 0:
        print(f"No images found in {INPUT_DIR}")
        return

    # one generator for everything, files are sorted so the noise is
    # always the same for the same image
    rng = np.random.default_rng(SEED)
    rows = []

    for f in files:
        name = os.path.splitext(f)[0]
        original = cv2.imread(os.path.join(INPUT_DIR, f))
        if original is None:
            print(f"Could not read {f}, skipping")
            continue

        degraded = degrade(original, rng)
        cv2.imwrite(os.path.join(DEGRADED_DIR, f"{name}_degraded.png"), degraded)

        results = {}
        scores = {}
        for key, label, func in METHODS:
            out = func(degraded)
            results[key] = out
            cv2.imwrite(os.path.join(RESULTS_DIR, f"{name}_{key}.png"), out)

            scores[key] = {
                "psnr": psnr(original, out),
                "ssim": ssim(original, out),
                "mae": mae(original, out),
            }
            rows.append([name, key,
                         round(scores[key]["psnr"], 3),
                         round(scores[key]["ssim"], 4),
                         round(scores[key]["mae"], 3)])

        save_figure(name, original, results, scores)
        print(f"done: {f}")

    # save metrics
    csv_path = os.path.join(RESULTS_DIR, "metrics.csv")
    with open(csv_path, "w", newline="") as fp:
        writer = csv.writer(fp)
        writer.writerow(["image", "method", "psnr_db", "ssim", "mae"])
        writer.writerows(rows)

    print()
    print(f"{'image':<15}{'method':<14}{'PSNR':>8}{'SSIM':>8}{'MAE':>8}")
    for r in rows:
        print(f"{r[0]:<15}{r[1]:<14}{r[2]:>8.2f}{r[3]:>8.3f}{r[4]:>8.2f}")
    print(f"\nMetrics saved in {csv_path}")


if __name__ == "__main__":
    main()