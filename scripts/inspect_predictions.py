# scripts/inspect_predictions.py
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from models.csrnet import CSRNet

CKPT = "checkpoints/best.pt"

def show_one(img_path, dmap_path):
    im = Image.open(img_path).convert("RGB")
    im_np = np.array(im)
    h, w = im_np.shape[:2]

    dm_gt = np.load(dmap_path).astype(np.float32)

    from torch.serialization import add_safe_globals
    from models.csrnet import CSRNet

    add_safe_globals([CSRNet])
    model = torch.load(CKPT, map_location="cpu", weights_only=False)

    model.eval()

    x = torch.from_numpy(im_np.astype(np.float32)/255.0).permute(2,0,1).unsqueeze(0)
    with torch.no_grad():
        pred = model(x).squeeze(0).squeeze(0).cpu().numpy()

    # normalize for display
    dm_gt_disp   = dm_gt / (dm_gt.max()+1e-8)
    dm_pred_disp = pred  / (pred.max()+1e-8)

    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    axes[0,0].imshow(im_np); axes[0,0].set_title("Original"); axes[0,0].axis("off")
    im1 = axes[0,1].imshow(dm_gt_disp, cmap="inferno"); axes[0,1].set_title(f"GT (sum≈{dm_gt.sum():.1f})"); axes[0,1].axis("off"); plt.colorbar(im1, ax=axes[0,1])
    axes[0,2].imshow(im_np); axes[0,2].imshow(dm_gt_disp, cmap="inferno", alpha=0.5); axes[0,2].set_title("GT Overlay"); axes[0,2].axis("off")

    im2 = axes[1,1].imshow(dm_pred_disp, cmap="inferno"); axes[1,1].set_title(f"PRED (sum≈{pred.sum():.1f})"); axes[1,1].axis("off"); plt.colorbar(im2, ax=axes[1,1])
    axes[1,0].axis("off")
    axes[1,2].imshow(im_np); axes[1,2].imshow(dm_pred_disp, cmap="inferno", alpha=0.5); axes[1,2].set_title("Pred Overlay"); axes[1,2].axis("off")

    plt.tight_layout(); plt.show()

if __name__ == "__main__":
    # pick the first pair from splits/val.txt for a quick check
    with open("splits/val.txt","r",encoding="utf-8") as f:
        line = f.readline().strip()
    img, dmap = line.split("|")
    show_one(img, dmap)
 