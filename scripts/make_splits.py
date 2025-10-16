# scripts/make_splits.py
from pathlib import Path
import random

IMAGES_DIR = Path("data_processed")
DMAPS_DIR  = Path("density_maps")
SPLITS_DIR = Path("splits"); SPLITS_DIR.mkdir(parents=True, exist_ok=True)

random.seed(42)

pairs = []
for img in IMAGES_DIR.rglob("*.jpg"):
    dmap = (DMAPS_DIR / img.relative_to(IMAGES_DIR)).with_suffix(".npy")
    if dmap.exists():
        pairs.append((img.as_posix(), dmap.as_posix()))

print("Found pairs:", len(pairs))
random.shuffle(pairs)

# 85% train, 15% val
cut = int(0.85 * len(pairs))
train, val = pairs[:cut], pairs[cut:] 

def write_list(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for a,b in rows:
            f.write(a + "|" + b + "\n")

write_list(SPLITS_DIR/"train.txt", train)
write_list(SPLITS_DIR/"val.txt", val)
print("Wrote:", SPLITS_DIR/"train.txt", SPLITS_DIR/"val.txt")
 