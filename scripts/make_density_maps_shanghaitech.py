from pathlib import Path
import numpy as np
from PIL import Image
import scipy.ndimage as nd
import scipy.io as sio

PROCESSED = Path("data_processed")
OUT = Path("density_maps"); OUT.mkdir(parents=True, exist_ok=True)
SIGMA = 4

def load_points(mat_path):
    data = sio.loadmat(mat_path)
    pts = data["image_info"][0,0][0,0][0]  # (x,y)
    return np.array(pts, dtype=np.float32)

def make_map(h, w, pts):
    m = np.zeros((h, w), np.float32)
    for x, y in pts:
        xi, yi = int(x), int(y)
        if 0 <= xi < w and 0 <= yi < h:
            m[yi, xi] = 1
    return nd.gaussian_filter(m, SIGMA)

for p_img in PROCESSED.rglob("*.jpg"):
    # locate original + GT
    rel = p_img.relative_to(PROCESSED)
    part, split, _, fname = rel.parts[:4]
    stem = p_img.stem
    orig = Path(part) / split / "images" / f"{stem}.jpg"
    mat  = Path(part) / split / "ground_truth" / f"GT_{stem}.mat"
    if not mat.exists(): 
        print(f"[skip] {stem}")
        continue

    # scale points
    pw, ph = Image.open(p_img).size
    ow, oh = Image.open(orig).size
    sx, sy = pw/ow, ph/oh
    pts = load_points(mat)
    pts[:,0] *= sx; pts[:,1] *= sy

    dmap = make_map(ph, pw, pts)
    out_path = OUT / rel.with_suffix(".npy")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, dmap)
    print(f"✅ {stem}  sum={dmap.sum():.1f}")
