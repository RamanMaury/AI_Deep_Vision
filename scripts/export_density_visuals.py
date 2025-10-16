# scripts/export_density_visuals.py
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import scipy.ndimage as nd

# -------- SETTINGS --------
IMAGES_DIR = Path("data_processed")
DMAPS_DIR  = Path("density_maps")
OUTPUT_DIR = Path("density_visuals")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CMAP = "plasma"       # "plasma", "inferno", "jet", "hot", "Blues"
ALPHA = 0.6           # transparency for overlay
GAUSSIAN_SMOOTH = 2   # blur for smoother look
# --------------------------

def export_overlay(img_path, dmap_path):
    """Create and save overlay visualization for one image."""
    im = Image.open(img_path).convert("RGB")
    im_np = np.array(im)
    h, w = im_np.shape[:2]
    dm = np.load(dmap_path).astype(np.float32)

    # Resize and smooth
    if dm.shape != (h, w):
        dm = np.array(Image.fromarray(dm).resize((w, h), Image.BILINEAR))
    dm = nd.gaussian_filter(dm, sigma=GAUSSIAN_SMOOTH)
    dm_disp = dm / dm.max() if dm.max() > 0 else dm
    total_count = dm.sum()

    # --- Create overlay ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    # 1️⃣ Original
    axes[0].imshow(im_np)
    axes[0].set_title("Original Image"); axes[0].axis("off")

    # 2️⃣ Density Heatmap
    im2 = axes[1].imshow(dm_disp, cmap=CMAP)
    axes[1].set_title(f"Density Map\nSum ≈ {total_count:.1f}")
    axes[1].axis("off")
    plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)

    # 3️⃣ Overlay
    axes[2].imshow(im_np)
    axes[2].imshow(dm_disp, cmap=CMAP, alpha=ALPHA)
    axes[2].set_title("Overlay (Crowd Intensity)")
    axes[2].axis("off")

    plt.tight_layout()

    # Save overlay PNG
    out_path = OUTPUT_DIR / f"{img_path.stem}_overlay.png"
    plt.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"✅ Saved overlay: {out_path.name} | Count ≈ {total_count:.1f}")

def main():
    pairs = []
    for img in IMAGES_DIR.rglob("*.jpg"):
        dmap = DMAPS_DIR / img.relative_to(IMAGES_DIR)
        dmap = dmap.with_suffix(".npy")
        if dmap.exists():
            pairs.append((img, dmap))
    if not pairs:
        print("⚠️ No matching image + density map found.")
        return

    print(f"Found {len(pairs)} images. Exporting overlays...\n")
    for i, (img, dmap) in enumerate(pairs, 1):
        print(f"[{i}/{len(pairs)}] {img.name}")
        export_overlay(img, dmap)
    print(f"\n🎉 Done! All overlays saved in: {OUTPUT_DIR.resolve()}")

if __name__ == "__main__":
    main()
