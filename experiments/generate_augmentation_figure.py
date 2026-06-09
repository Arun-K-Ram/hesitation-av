"""
experiments/generate_augmentation_figure.py

Generates augmentation visualization figure for paper.

Shows how each augmentation technique transforms
actual HesitAV-1564 frames across all three
scenario classes.

Layout:
  Rows:    pedestrian_curb | merge_hesitation | occluded_intersection
  Columns: Original | H-Flip | Rotation | Color Jitter |
           Gaussian Blur | Sobel Edge | Grayscale | Erasing

Run:
  python experiments/generate_augmentation_figure.py
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
import random

RECORDINGS   = Path("experiments/recordings")
PAPER_FIGURES = Path("paper_figures")
PAPER_FIGURES.mkdir(exist_ok=True)

#  Pick one representative frame per scenario 

SCENARIO_FOLDERS = {
    "pedestrian\_curb":       "pedestrian_curb_1780190788",
    "occluded\_intersection": "occluded_intersection_1780197183",
}

def pick_frame(folder_name, frame_index=300):
    """Load a representative frame from a recording folder."""
    frames_dir = RECORDINGS / folder_name / "frames"
    if not frames_dir.exists():
        print(f"  [WARN] Not found: {frames_dir}")
        return None
    jpgs = sorted(frames_dir.glob("*.jpg"))
    if not jpgs:
        return None
    idx   = min(frame_index, len(jpgs) - 1)
    frame = cv2.imread(str(jpgs[idx]))
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return frame


#  Augmentation functions 

def aug_original(img):
    return img.copy()

def aug_hflip(img):
    return np.fliplr(img)

def aug_rotation(img, angle=15):
    h, w  = img.shape[:2]
    M     = cv2.getRotationMatrix2D((w//2, h//2), angle, 1.0)
    return cv2.warpAffine(img, M, (w, h),
                          borderMode=cv2.BORDER_REFLECT)

def aug_colorjitter(img):
    out = img.astype(np.float32)
    # Brightness
    out = out * random.uniform(0.6, 1.4)
    # Contrast
    mean = out.mean()
    out  = (out - mean) * random.uniform(0.7, 1.3) + mean
    # Saturation (operate in HSV)
    hsv  = cv2.cvtColor(
        np.clip(out, 0, 255).astype(np.uint8),
        cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[..., 1] *= random.uniform(0.6, 1.4)
    out = cv2.cvtColor(
        np.clip(hsv, 0, 255).astype(np.uint8),
        cv2.COLOR_HSV2RGB)
    return np.clip(out, 0, 255).astype(np.uint8)

def aug_gaussian_blur(img, ksize=11):
    return cv2.GaussianBlur(img, (ksize, ksize), 0)

def aug_sobel_edge(img):
    gray  = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    sx    = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sy    = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    mag   = np.sqrt(sx**2 + sy**2)
    mag   = (mag / mag.max() * 255).astype(np.uint8)
    return cv2.cvtColor(mag, cv2.COLOR_GRAY2RGB)

def aug_grayscale(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

def aug_erasing(img, ratio=0.25):
    out  = img.copy()
    h, w = out.shape[:2]
    rh   = int(h * ratio)
    rw   = int(w * ratio)
    y    = random.randint(0, h - rh)
    x    = random.randint(0, w - rw)
    out[y:y+rh, x:x+rw] = 128  # grey fill
    return out


AUGMENTATIONS = [
    ("Original",       aug_original),
    ("H-Flip",         aug_hflip),
    ("Rotation\n±15°", aug_rotation),
    ("Color\nJitter",  aug_colorjitter),
    ("Gaussian\nBlur", aug_gaussian_blur),
    ("Sobel\nEdge",    aug_sobel_edge),
    ("Grayscale",      aug_grayscale),
    ("Random\nErasing",aug_erasing),
]


#  Plot 

def generate_figure():
    random.seed(42)

    scenarios = list(SCENARIO_FOLDERS.items())
    n_rows    = len(scenarios)
    n_cols    = len(AUGMENTATIONS)

    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor":   "white",
        "text.color":       "#222222",
    })

    fig = plt.figure(figsize=(n_cols * 2.0,
                               n_rows * 2.4),
                      facecolor="white")
    fig.suptitle(
        "Data Augmentation Pipeline - HesitAV-1564",
        fontsize=11, fontweight="bold",
        color="#111111", y=0.98)

    gs = gridspec.GridSpec(
        n_rows, n_cols, figure=fig,
        hspace=0.08, wspace=0.04)

    for row, (label, folder) in enumerate(scenarios):
        frame = pick_frame(folder)
        if frame is None:
            print(f"  [SKIP] {folder} not found")
            continue

        # Resize to uniform size for display
        frame = cv2.resize(frame, (224, 224))

        for col, (aug_name, aug_fn) in \
                enumerate(AUGMENTATIONS):
            ax  = fig.add_subplot(gs[row, col])
            aug = aug_fn(frame)
            ax.imshow(aug)
            ax.axis("off")

            # Column headers (top row only)
            if row == 0:
                ax.set_title(aug_name,
                              fontsize=7.5,
                              color="#333333",
                              pad=4)

            # Row labels (first column only)
            if col == 0:
                ax.set_ylabel(
                    label,
                    fontsize=8,
                    color="#333333",
                    labelpad=4,
                    rotation=90,
                    va="center")

    out = PAPER_FIGURES / "augmentation_pipeline.png"
    plt.savefig(out, dpi=150,
                bbox_inches="tight",
                facecolor="white")
    print(f"\n  Saved → {out}")


if __name__ == "__main__":
    print("\nGenerating augmentation figure...")
    generate_figure()
    print("Done.")