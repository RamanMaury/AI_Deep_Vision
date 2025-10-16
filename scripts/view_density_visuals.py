# scripts/view_density_visuals.py
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt

VISUALS_DIR = Path("density_visuals")

def main():
    overlays = sorted(VISUALS_DIR.glob("*.png"))
    if not overlays:
        print("⚠️ No overlay images found. Run export_density_visuals.py first.")
        return

    print(f"🖼 Found {len(overlays)} overlay images.\n")

    for i, p in enumerate(overlays, 1):
        im = Image.open(p)
        plt.figure(figsize=(8, 6))
        plt.imshow(im)
        plt.axis("off")
        plt.title(f"{p.name}  ({i}/{len(overlays)})", fontsize=10)
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    main()
