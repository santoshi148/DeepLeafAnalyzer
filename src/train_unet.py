import argparse
from pathlib import Path
import torch
from torch.utils.data import DataLoader, random_split
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
from .config import Config
from .dataset import SegmentationDataset
from .models.unet import UNet, DiceBCELoss
from .utils import seed_everything, get_device, ensure_dirs


def dice_iou_from_logits(logits, masks, thr=0.5, eps=1e-6):
    pred = (torch.sigmoid(logits) > thr).float()
    inter = (pred * masks).sum(dim=(1,2,3))
    union = pred.sum(dim=(1,2,3)) + masks.sum(dim=(1,2,3))
    dice = ((2 * inter + eps) / (union + eps)).mean().item()
    iou = ((inter + eps) / (pred.sum(dim=(1,2,3)) + masks.sum(dim=(1,2,3)) - inter + eps)).mean().item()
    return dice, iou


def train_unet(cfg: Config):
    seed_everything(cfg.seed)
    device = get_device(cfg.device)
    ensure_dirs(cfg.checkpoint_dir, cfg.result_dir)
    ds = SegmentationDataset(cfg.raw_dir, cfg.mask_dir, cfg.img_size)
    val_len = max(1, int(len(ds) * 0.1)); train_len = len(ds) - val_len
    train_ds, val_ds = random_split(ds, [train_len, val_len], generator=torch.Generator().manual_seed(cfg.seed))
    train_loader = DataLoader(train_ds, cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, pin_memory=True)
    model = UNet().to(device)
    criterion = DiceBCELoss()
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scaler = GradScaler(enabled=cfg.use_amp and device.type == "cuda")
    best = -1
    for epoch in range(1, cfg.unet_epochs + 1):
        model.train(); total = 0
        for imgs, masks, _ in tqdm(train_loader, desc=f"U-Net Epoch {epoch}"):
            imgs, masks = imgs.to(device), masks.to(device)
            opt.zero_grad(set_to_none=True)
            with autocast(enabled=scaler.is_enabled()):
                logits = model(imgs); loss = criterion(logits, masks)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            total += loss.item() * imgs.size(0)
        model.eval(); dices=[]; ious=[]
        with torch.no_grad():
            for imgs, masks, _ in val_loader:
                imgs, masks = imgs.to(device), masks.to(device)
                logits = model(imgs); d, i = dice_iou_from_logits(logits, masks)
                dices.append(d); ious.append(i)
        md, mi = sum(dices)/len(dices), sum(ious)/len(ious)
        print(f"Epoch {epoch}: loss={total/len(train_ds):.4f}, val_dice={md:.4f}, val_iou={mi:.4f}")
        if md > best:
            best = md
            torch.save({"model": model.state_dict(), "dice": best, "cfg": cfg.__dict__}, Path(cfg.checkpoint_dir)/"best_unet.pth")
    return model

if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--config", default=None)
    args = p.parse_args(); cfg = Config.load(args.config) if args.config else Config()
    train_unet(cfg)
