"""
experiments/combine_carla_screenshots.py

Combines three CARLA scenario screenshots into
one side-by-side figure for the paper.

Input images (put in paper_figures/ folder):
  spawn_pedestrian_curb.png
  spawn_merge_hesitation.png
  spawn_occluded_intersection.png

Output:
  paper_figures/carla_scenarios.png

Run:
  python experiments/combine_carla_screenshots.py
"""

from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
import sys

PAPER_FIGURES = Path(__file__).parent.parent / "paper_figures"
PAPER_FIGURES.mkdir(exist_ok=True)

# Input files
INPUT_FILES = [
    ("spawn_pedestrian_curb.png",      "pedestrian\\_curb"),
    ("spawn_merge_hesitation.png",     "merge\\_hesitation"),
    ("spawn_occluded_intersection.png","occluded\\_intersection"),
]

OUTPUT_FILE = PAPER_FIGURES / "carla_scenarios.png"

# Layout settings
TARGET_HEIGHT  = 400   # px per image
PADDING        = 20    # px between images
LABEL_HEIGHT   = 40    # px for label below each image
BG_COLOR       = (15, 23, 42)    # dark background #0f172a
LABEL_COLOR    = (148, 163, 184) # #94a3b8
BORDER_COLOR   = (30, 41, 59)    # #1e293b


def load_and_resize(path: Path, target_height: int) -> Image.Image:
    """Load image and resize to target height maintaining aspect ratio."""
    img = Image.open(path).convert("RGB")
    w, h   = img.size
    ratio  = target_height / h
    new_w  = int(w * ratio)
    return img.resize((new_w, target_height),
                      Image.LANCZOS)


def add_label(img: Image.Image, label: str,
              label_height: int,
              bg_color: tuple,
              label_color: tuple) -> Image.Image:
    """Add a label strip below the image."""
    w, h    = img.size
    canvas  = Image.new("RGB",
                         (w, h + label_height),
                         bg_color)
    canvas.paste(img, (0, 0))

    draw = ImageDraw.Draw(canvas)

    # Try to load a font, fall back to default
    try:
        font = ImageFont.truetype(
            "C:/Windows/Fonts/consola.ttf", 16)
    except:
        font = ImageFont.load_default()

    # Clean label for display
    clean_label = label.replace("\\_", "_")

    # Center text
    bbox = draw.textbbox((0, 0), clean_label, font=font)
    text_w = bbox[2] - bbox[0]
    text_x = (w - text_w) // 2
    text_y = h + (label_height - (bbox[3] - bbox[1])) // 2

    draw.text((text_x, text_y), clean_label,
              fill=label_color, font=font)

    return canvas


def add_border(img: Image.Image,
               border: int,
               color: tuple) -> Image.Image:
    """Add a thin border around the image."""
    w, h   = img.size
    canvas = Image.new("RGB",
                        (w + border*2, h + border*2),
                        color)
    canvas.paste(img, (border, border))
    return canvas


def main():
    print("\nCombining CARLA scenario screenshots...")
    print(f"Looking in: {PAPER_FIGURES}\n")

    images = []
    missing = []

    for filename, label in INPUT_FILES:
        path = PAPER_FIGURES / filename
        if not path.exists():
            print(f"  [!] Missing: {filename}")
            missing.append(filename)
        else:
            print(f"  [✓] Found: {filename}")
            img = load_and_resize(path, TARGET_HEIGHT)
            img = add_label(img, label,
                            LABEL_HEIGHT,
                            BG_COLOR, LABEL_COLOR)
            img = add_border(img, 2, BORDER_COLOR)
            images.append(img)

    if missing:
        print(f"\n  Missing {len(missing)} image(s):")
        for f in missing:
            print(f"    → {f}")
        print(f"\n  Copy your screenshots to:")
        print(f"    {PAPER_FIGURES}")
        print(f"  With these exact names:")
        for filename, _ in INPUT_FILES:
            print(f"    {filename}")
        if len(images) == 0:
            sys.exit(1)
        print(f"\n  Proceeding with {len(images)} available image(s)...")

    if not images:
        print("  No images to combine.")
        sys.exit(1)

    # Combine side by side
    total_w = sum(img.width for img in images) \
              + PADDING * (len(images) + 1)
    max_h   = max(img.height for img in images) \
              + PADDING * 2

    canvas = Image.new("RGB", (total_w, max_h), BG_COLOR)

    x = PADDING
    for img in images:
        y = PADDING
        canvas.paste(img, (x, y))
        x += img.width + PADDING

    canvas.save(OUTPUT_FILE, dpi=(150, 150))
    print(f"\n  Saved → {OUTPUT_FILE}")
    print(f"  Size: {canvas.width} × {canvas.height} px")
    print(f"\n  Upload carla_scenarios.png to Overleaf paper_figures/")
    print(f"  Done.")


if __name__ == "__main__":
    # Check PIL is available
    try:
        from PIL import Image
    except ImportError:
        print("Installing Pillow...")
        import subprocess
        subprocess.run(["pip", "install", "Pillow",
                        "--break-system-packages"])
        from PIL import Image
    main()