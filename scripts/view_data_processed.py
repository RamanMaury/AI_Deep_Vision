# scripts/view_data_processed.py
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt

# -------- SETTINGS --------
FOLDER = Path("data_processed")  # folder to open images from
IMG_EXTS = [".jpg", ".jpeg", ".png"]
# ---------------------------

def main():
    # find all image files
    images = []
    for ext in IMG_EXTS:
        images.extend(FOLDER.rglob(f"*{ext}"))
    images = sorted(images)

    if not images:
        print(f"⚠️ No images found in {FOLDER.resolve()}")
        return

    print(f"🖼 Found {len(images)} image(s) inside {FOLDER.resolve()}.\n")
    print("Press ➡️ any key in the popup window to see the next image.")
    print("Press ❌ close button or CTRL+C in terminal to stop.\n")

    # loop through all images
    for i, img_path in enumerate(images, 1):
        im = Image.open(img_path).convert("RGB")
        plt.figure(figsize=(8, 6))
        plt.imshow(im)
        plt.axis("off")
        plt.title(f"{img_path.name}  ({i}/{len(images)})", fontsize=11)
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    main()
