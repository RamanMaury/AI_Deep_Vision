from pathlib import Path
from PIL import Image

# Include both datasets
PARTS = [
    Path("part_A_final/train_data"),
    Path("part_A_final/test_data"),
    Path("part_B_final/train_data"), 
    Path("part_B_final/test_data"),
]

OUT = Path("data_processed")
OUT.mkdir(parents=True, exist_ok=True)
TARGET_LONG = 768  # resize longest edge to 768 px

def resize_keep_ratio(img):
    """Resize while maintaining aspect ratio."""
    w, h = img.size
    scale = TARGET_LONG / max(w, h)
    return img.resize((int(w * scale), int(h * scale)), Image.BILINEAR)

count = 0
for root in PARTS:
    img_dir = root / "images"
    if not img_dir.exists():
        print(f"⚠️ Skipping {img_dir} (not found)")
        continue

    for p in img_dir.glob("*.jpg"):
        im = Image.open(p).convert("RGB")
        im_r = resize_keep_ratio(im)
        out_path = OUT / root.parent.name / root.name / "images" / p.name
        out_path.parent.mkdir(parents=True, exist_ok=True)
        im_r.save(out_path)
        count += 1

print(f"✅ Resized {count} images from Part A and Part B into {OUT}")
