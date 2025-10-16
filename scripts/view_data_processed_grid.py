# scripts/view_data_processed_grid.py
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import math

# ---------- SETTINGS ----------
FOLDER = Path("data_processed")   # folder containing your images
IMG_EXTS = [".jpg", ".jpeg", ".png"]
ROWS, COLS = 3, 3                 # grid size → 3x3 = 9 images per page
IMAGE_SIZE = (400, 300)           # display size for thumbnails
# --------------------------------

def get_images(folder):
    """Fetch all image files from folder recursively."""
    images = []
    for ext in IMG_EXTS:
        images.extend(folder.rglob(f"*{ext}"))
    return sorted(images)

def show_grid(images, page, total_pages):
    """Display a grid of images using matplotlib."""
    fig, axes = plt.subplots(ROWS, COLS, figsize=(COLS * 4, ROWS * 3))
    fig.suptitle(f"Page {page}/{total_pages}  |  Showing {len(images)} image(s)", fontsize=14)
    
    # Flatten axes array for easy iteration
    axes = axes.flatten()

    for ax, img_path in zip(axes, images):
        try:
            im = Image.open(img_path).convert("RGB")
            im.thumbnail(IMAGE_SIZE)
            ax.imshow(im)
            ax.set_title(img_path.name, fontsize=8)
            ax.axis("off")
        except Exception as e:
            ax.text(0.5, 0.5, f"Error\n{img_path.name}", ha='center', va='center', color='red')
            ax.axis("off")

    # Hide unused subplots
    for ax in axes[len(images):]:
        ax.axis("off")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

def main():
    images = get_images(FOLDER)
    if not images:
        print(f"⚠️ No images found in {FOLDER.resolve()}")
        return

    per_page = ROWS * COLS
    total_pages = math.ceil(len(images) / per_page)

    print(f"🖼 Found {len(images)} image(s) in {FOLDER.resolve()}")
    print(f"📄 Displaying {per_page} per page ({total_pages} pages total)")
    print("Press any key or close the popup to move to next page.\n")

    for i in range(total_pages):
        batch = images[i*per_page:(i+1)*per_page]
        show_grid(batch, i+1, total_pages)

    print("✅ Done viewing all images!")

if __name__ == "__main__":
    main()
