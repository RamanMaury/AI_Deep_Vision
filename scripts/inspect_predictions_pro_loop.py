# scripts/inspect_predictions_pro_loop.py
import sys, os
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
from models.csrnet import CSRNet

# ---------- CONFIG ---------- 
CKPT = "checkpoints/best.pt"
VAL_LIST = "splits/val.txt"
SAVE_DIR = "results/inspect_pro"
CMAP = "inferno"
ALPHA = 0.55
SMOOTH_SIGMA = 2.0
PCLIP = (1, 99.7)
BG_COLOR = "#00122B"
# ----------------------------

def load_model():
    """Safely load checkpoint."""
    model = CSRNet()
    try:
        sd = torch.load(CKPT, map_location="cpu")
        if isinstance(sd, dict) and any(k.startswith("frontend") for k in sd.keys()):
            model.load_state_dict(sd)
            print("✅ Loaded weights-only checkpoint.")
            return model.eval()
    except Exception:
        pass
    from torch.serialization import add_safe_globals
    add_safe_globals([CSRNet])
    model = torch.load(CKPT, map_location="cpu", weights_only=False)
    print("✅ Loaded full-object checkpoint.")
    return model.eval()

def read_pairs():
    """Read val list file."""
    pairs = []
    with open(VAL_LIST, "r", encoding="utf-8") as f:
        for line in f:
            if "|" in line:
                img, dmap = line.strip().split("|")
                pairs.append((img, dmap))
    print(f"✅ Found {len(pairs)} validation pairs.")
    return pairs

def normalize_map(dm, sigma=SMOOTH_SIGMA, pclip=PCLIP):
    """Normalize + optional Gaussian smoothing."""
    dm = dm.astype(np.float32)
    try:
        from scipy.ndimage import gaussian_filter
        if sigma > 0:
            dm = gaussian_filter(dm, sigma)
    except Exception:
        pass
    lo, hi = np.percentile(dm, pclip)
    dm = np.clip(dm, lo, hi)
    dm = (dm - dm.min()) / (dm.max() - dm.min() + 1e-6)
    return dm

def predict(model, img_path):
    """Predict density map."""
    im = Image.open(img_path).convert("RGB")
    arr = np.asarray(im, dtype=np.float32) / 255.0
    x = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    with torch.no_grad():
        pred = model(x).squeeze().numpy()
    return im, pred

def style_axes(ax):
    ax.axis("off")
    ax.set_facecolor(BG_COLOR)

def colorbar_white(cbar):
    cbar.outline.set_edgecolor("white")
    cbar.ax.yaxis.set_tick_params(color="white")
    for tick in cbar.ax.get_yticklabels():
        tick.set_color("white")

def show_popup(im, gt, pred, title):
    """Show popup with 5 panels (Original, GT, Pred, GT Overlay, Pred Overlay)."""
    gt_disp = normalize_map(gt)
    pr_disp = normalize_map(pred)
    gt_sum = gt.sum()
    pr_sum = pred.sum()

    fig, axes = plt.subplots(2, 3, figsize=(10, 6))
    fig.patch.set_facecolor(BG_COLOR)
    fig.suptitle(
        f"{title}\nGT≈{gt_sum:.1f} | Pred≈{pr_sum:.1f} | Error={pr_sum - gt_sum:+.1f}",
        color="white",
        fontsize=13,
    )

    # 1️⃣ Original
    axes[0, 0].imshow(im)
    axes[0, 0].set_title("Original", color="white")
    style_axes(axes[0, 0])

    # 2️⃣ GT Heatmap
    im2 = axes[0, 1].imshow(gt_disp, cmap=CMAP)
    axes[0, 1].set_title("GT Density Map", color="white")
    style_axes(axes[0, 1])
    cbar = plt.colorbar(im2, ax=axes[0, 1], fraction=0.046, pad=0.04)
    colorbar_white(cbar)

    # 3️⃣ Pred Heatmap
    im3 = axes[0, 2].imshow(pr_disp, cmap=CMAP)
    axes[0, 2].set_title("Pred Density Map", color="white")
    style_axes(axes[0, 2])
    cbar = plt.colorbar(im3, ax=axes[0, 2], fraction=0.046, pad=0.04)
    colorbar_white(cbar)

    # 4️⃣ GT Overlay
    axes[1, 0].imshow(im)
    axes[1, 0].imshow(gt_disp, cmap=CMAP, alpha=ALPHA)
    axes[1, 0].set_title("GT Overlay", color="white")
    style_axes(axes[1, 0])

    # 5️⃣ Pred Overlay
    axes[1, 1].imshow(im)
    axes[1, 1].imshow(pr_disp, cmap=CMAP, alpha=ALPHA)
    axes[1, 1].set_title("Pred Overlay", color="white")
    style_axes(axes[1, 1])

    # Empty panel
    axes[1, 2].axis("off")

    plt.tight_layout()
    plt.show(block=True)
    plt.close(fig)

def main():
    os.makedirs(SAVE_DIR, exist_ok=True)
    pairs = read_pairs()
    model = load_model()

    idx = 0
    while 0 <= idx < len(pairs):
        img_path, dmap_path = pairs[idx]
        try:
            im, pred = predict(model, img_path)
            gt = np.load(dmap_path).astype(np.float32)
            title = f"{os.path.basename(img_path)} ({idx + 1}/{len(pairs)})"
            show_popup(im, gt, pred, title)
        except Exception as e:
            print(f"⚠️ Error on {img_path}: {e}")

        # Navigation input
        choice = input("Enter=next | b=back | q=quit: ").strip().lower()
        if choice == "b":
            idx = max(0, idx - 1)
        elif choice == "q":
            print("👋 Exiting visualization.")
            break
        else:
            idx += 1

if __name__ == "__main__":
    main()
