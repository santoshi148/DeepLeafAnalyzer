import argparse, json
from pathlib import Path
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from .config import Config
from .dataset import make_classification_loaders
from .models.hybrid_leaf_disease_net import HybridLeafDiseaseNet
from .utils import seed_everything, get_device, ensure_dirs


def run_epoch(model, loader, criterion, device, optimizer=None, scaler=None, use_amp=True):
    train = optimizer is not None
    model.train(train)
    losses=[]; y_true=[]; y_pred=[]
    for imgs, labels, _ in tqdm(loader, leave=False):
        imgs, labels = imgs.to(device), labels.to(device)
        if train: optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(train):
            with autocast(enabled=use_amp and device.type == "cuda"):
                logits = model(imgs); loss = criterion(logits, labels)
            if train:
                scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
        losses.append(loss.item()*imgs.size(0))
        y_true.extend(labels.detach().cpu().tolist())
        y_pred.extend(logits.argmax(1).detach().cpu().tolist())
    acc = accuracy_score(y_true, y_pred)
    pr, rc, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)
    return sum(losses)/len(loader.dataset), acc, pr, rc, f1


def train_classifier(cfg: Config):
    seed_everything(cfg.seed)
    device = get_device(cfg.device)
    ensure_dirs(cfg.checkpoint_dir, cfg.result_dir)
    data_root = cfg.roi_dir if Path(cfg.roi_dir).exists() and any(Path(cfg.roi_dir).iterdir()) else cfg.raw_dir
    loaders, class_to_idx = make_classification_loaders(data_root, cfg.img_size, cfg.batch_size, cfg.val_split, cfg.test_split, cfg.seed, cfg.num_workers)
    cfg.num_classes = len(class_to_idx)
    model = HybridLeafDiseaseNet(num_classes=cfg.num_classes, dropout=cfg.dropout).to(device)
    criterion = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=5)
    scaler = GradScaler(enabled=cfg.use_amp and device.type == "cuda")
    best_f1=-1; patience=10; bad=0
    with open(Path(cfg.result_dir)/"class_to_idx.json", "w") as f: json.dump(class_to_idx, f, indent=2)
    for epoch in range(1, cfg.epochs+1):
        tr = run_epoch(model, loaders["train"], criterion, device, opt, scaler, cfg.use_amp)
        va = run_epoch(model, loaders["val"], criterion, device, None, scaler, cfg.use_amp)
        scheduler.step(va[0])
        print(f"Epoch {epoch}: train_loss={tr[0]:.4f}, val_loss={va[0]:.4f}, val_acc={va[1]:.4f}, val_f1={va[4]:.4f}")
        if va[4] > best_f1:
            best_f1 = va[4]; bad=0
            torch.save({"model": model.state_dict(), "class_to_idx": class_to_idx, "cfg": cfg.__dict__}, Path(cfg.checkpoint_dir)/"best_hybridleaf.pth")
        else:
            bad += 1
            if bad >= patience:
                print("Early stopping."); break
    ckpt = torch.load(Path(cfg.checkpoint_dir)/"best_hybridleaf.pth", map_location=device)
    model.load_state_dict(ckpt["model"])
    te = run_epoch(model, loaders["test"], criterion, device, None, scaler, cfg.use_amp)
    metrics = {"loss": te[0], "accuracy": te[1], "precision": te[2], "recall": te[3], "f1": te[4]}
    with open(Path(cfg.result_dir)/"test_metrics.json", "w") as f: json.dump(metrics, f, indent=2)
    print("Test:", metrics)
    return model

if __name__ == "__main__":
    p=argparse.ArgumentParser(); p.add_argument("--config", default=None)
    args=p.parse_args(); cfg=Config.load(args.config) if args.config else Config()
    train_classifier(cfg)
