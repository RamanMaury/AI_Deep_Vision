# models/dataset.py
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import torch 
import torch.nn.functional as F
from torchvision import transforms
from pathlib import Path

class CrowdDataset(Dataset): 
    """
    Crowd counting dataset for CSRNet.
    Automatically resizes all images and density maps
    to a fixed target size (default 576x768) so DataLoader can batch them.
    """
    def __init__(self, list_file: str | Path, target_size=(576, 768)):
        list_file = Path(list_file)
        if not list_file.exists():
            raise FileNotFoundError(f"Split file not found: {list_file}")

        self.items = []
        with list_file.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                img, dmap = line.split("|")
                self.items.append((img, dmap))

        if len(self.items) == 0:
            raise RuntimeError(f"No pairs found in {list_file}. Did make_splits.py run?")

        self.target_size = target_size
        self.img_transform = transforms.Compose([
            transforms.Resize(self.target_size),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        img_path, dmap_path = self.items[idx]

        # ---- Load & resize image ----
        im = Image.open(img_path).convert("RGB")
        im = self.img_transform(im)

        # ---- Load & resize density map ----
        dm = np.load(dmap_path).astype(np.float32)
        dm = torch.from_numpy(dm).unsqueeze(0)  # [1, H, W]
        dm = F.interpolate(dm.unsqueeze(0), size=self.target_size, mode='bilinear', align_corners=False).squeeze(0)

        return {"image": im, "density": dm, "name": img_path}


# ---------- Self-Test ----------
if __name__ == "__main__":
    try:
        split_file = Path("splits/train.txt")
        ds = CrowdDataset(split_file)
        print(f"✅ Dataset loaded: {len(ds)} samples from {split_file}")

        sample = ds[0]
        img_shape = tuple(sample["image"].shape)     # (3, H, W)
        dm_shape  = tuple(sample["density"].shape)   # (1, H, W)
        dm_sum    = float(sample["density"].sum())   # ≈ people count

        print(f"image tensor shape : {img_shape}")
        print(f"density tensor shape: {dm_shape}")
        print(f"density sum (≈count): {dm_sum:.1f}")

    except Exception as e:
        print("❌ Self-test failed:", e)
