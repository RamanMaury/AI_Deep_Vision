# scripts/train_csrnet.py
import torch, os, math
from torch.utils.data import DataLoader
from torch import nn, optim
from models.dataset import CrowdDataset
from models.csrnet import CSRNet
from tqdm import tqdm

TRAIN_LIST = "splits/train.txt"
VAL_LIST   = "splits/val.txt"
CKPT_DIR   = "checkpoints" 
os.makedirs(CKPT_DIR, exist_ok=True)

BATCH_SIZE = 1          # reduce to 2 if you get OOM
LR         = 1e-4
EPOCHS     = 2       # stop earlier if val stops improving
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
AMP        = True       # mixed precision for speed on GPU

def mae_rmse(pred_counts, gt_counts):
    diffs = [abs(p-g) for p,g in zip(pred_counts, gt_counts)]
    mae  = sum(diffs)/len(diffs)
    rmse = math.sqrt(sum((p-g)**2 for p,g in zip(pred_counts, gt_counts))/len(diffs))
    return mae, rmse

def run_epoch(model, loader, optimizer=None, scaler=None):
    is_train = optimizer is not None
    mode = "train" if is_train else "eval"
    if is_train: model.train()
    else: model.eval()

    mse = nn.MSELoss()
    total_loss = 0.0
    pred_counts, gt_counts = [], []

    for batch in tqdm(loader, desc=mode, ncols=90):
        img = batch["image"].to(DEVICE)
        gt  = batch["density"].to(DEVICE)

        with torch.cuda.amp.autocast(enabled=AMP), torch.set_grad_enabled(is_train):
            pred = model(img)                 # [B,1,H,W]
            loss = mse(pred, gt)

        if is_train:
            optimizer.zero_grad(set_to_none=True)
            if AMP and scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

        total_loss += loss.item() * img.size(0)
        # counts
        pred_counts += [float(p.sum().item()) for p in pred.detach().cpu()]
        gt_counts   += [float(y.sum().item()) for y in gt.detach().cpu()]

    avg_loss = total_loss / len(loader.dataset)
    mae, rmse = mae_rmse(pred_counts, gt_counts)
    return avg_loss, mae, rmse

def main():
    train_ds = CrowdDataset(TRAIN_LIST)
    val_ds   = CrowdDataset(VAL_LIST)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=1,         shuffle=False, num_workers=2, pin_memory=True)

    model = CSRNet().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scaler = torch.cuda.amp.GradScaler(enabled=AMP)

    best_mae = float("inf")
    for epoch in range(1, EPOCHS+1):
        tr_loss, tr_mae, tr_rmse = run_epoch(model, train_loader, optimizer, scaler)
        val_loss, val_mae, val_rmse = run_epoch(model, val_loader)

        print(f"\nEpoch {epoch:03d} | "
              f"train loss {tr_loss:.4f}  MAE {tr_mae:.1f}  RMSE {tr_rmse:.1f} || "
              f"val loss {val_loss:.4f}  MAE {val_mae:.1f}  RMSE {val_rmse:.1f}")

        # save best by val MAE
        if val_mae < best_mae:
            best_mae = val_mae
            path = os.path.join(CKPT_DIR, "best.pt")
            torch.save(model, path)
            print(f"✅ Saved best model to {path} (val MAE={best_mae:.1f})")

if __name__ == "__main__":
    main()
