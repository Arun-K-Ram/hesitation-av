from PIL import Image, ImageDraw, ImageFont
from pathlib import Path

PAPER_FIGURES = Path("paper_figures")

files = [
    ("pedestrian_curb_test.png",   "pedestrian\\_curb"),
    ("merge_hesitation.png",       "merge\\_hesitation"),
    ("occluded_intersection.png",  "occluded\\_intersection"),
]

TARGET_HEIGHT = 400
PADDING       = 20
LABEL_HEIGHT  = 36
BG_COLOR      = (15, 23, 42)
LABEL_COLOR   = (148, 163, 184)

images = []
for filename, label in files:
    path = PAPER_FIGURES / filename
    img  = Image.open(path).convert("RGB")
    w, h = img.size
    new_w = int(w * TARGET_HEIGHT / h)
    img   = img.resize((new_w, TARGET_HEIGHT),
                        Image.LANCZOS)

    # Add label strip
    canvas = Image.new("RGB",
                        (new_w, TARGET_HEIGHT + LABEL_HEIGHT),
                        BG_COLOR)
    canvas.paste(img, (0, 0))
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype(
            "C:/Windows/Fonts/consola.ttf", 14)
    except:
        font = ImageFont.load_default()
    clean = label.replace("\\_", "_")
    bbox  = draw.textbbox((0,0), clean, font=font)
    tx = (new_w - (bbox[2]-bbox[0])) // 2
    ty = TARGET_HEIGHT + 8
    draw.text((tx, ty), clean,
               fill=LABEL_COLOR, font=font)
    images.append(canvas)

total_w = sum(img.width for img in images) \
          + PADDING * (len(images) + 1)
max_h   = max(img.height for img in images) \
          + PADDING * 2

canvas = Image.new("RGB", (total_w, max_h), BG_COLOR)
x = PADDING
for img in images:
    canvas.paste(img, (x, PADDING))
    x += img.width + PADDING

out = PAPER_FIGURES / "hesitav_samples.png"
canvas.save(out, dpi=(150, 150))
print(f"Saved {out}")