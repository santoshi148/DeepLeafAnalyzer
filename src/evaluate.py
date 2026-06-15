import argparse, json, time
from pathlib import Path
import torch
import torch.nn as nn
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from .config import Config
from .dataset import make_classification_loaders
from .models.hybrid_leaf_disease_net import HybridLeafDiseaseNet
from .utils import get_device, ensure_dirs, now_ms


def evaluate_classifier(cfg: Config, weights=None):
    device = get_device(cfg.device); ensure_dirs(cfg.result_dir, cfg.figure_dir)
    data_root = cfg.roi_dir if Path(cfg.roi_dir).exists() and any(Path(cfg.roi_dir).iterdir()) else cfg.raw_dir
    loaders, class_to_idx = make_classification_loaders(data_root, cfg.img_size, cfg.batch_size, cfg.val_split, cfg.test_split, cfg.seed, cfg.num_workers)
    idx_to_class = {v:k for k,v in class_to_idx.items()}
    weights = weights or str(Path(cfg.checkpoint_dir)/"best_hybridleaf.pth")
    ckpt = torch.load(weights, map_location=device)
    model = HybridLeafDiseaseNet(num_classes=len(class_to_idx), dropout=cfg.dropout).to(device)
    model.load_state_dict(ckpt["model"]); model.eval()
    y_true=[]; y_pred=[]; rows=[]; times=[]
    with torch.no_grad():
        for imgs, labels, paths in loaders["test"]:
            imgs = imgs.to(device)
            start = now_ms(); logits = model(imgs); elapsed = (now_ms()-start)/imgs.size(0); times.append(elapsed)
            probs = torch.softmax(logits, 1).cpu(); preds = probs.argmax(1)
            for pth, lab, pr, prob in zip(paths, labels, preds, probs):
                y_true.append(int(lab)); y_pred.append(int(pr))
                rows.append({"image": pth, "true": idx_to_class[int(lab)], "predicted": idx_to_class[int(pr)], "confidence": float(prob[int(pr)])})
    report = classification_report(y_true, y_pred, target_names=[idx_to_class[i] for i in range(len(idx_to_class))], output_dict=True, zero_division=0)
    pd.DataFrame(rows).to_csv(Path(cfg.result_dir)/"classification_predictions.csv", index=False)
    pd.DataFrame(report).transpose().to_csv(Path(cfg.result_dir)/"classification_report.csv")
    pd.DataFrame(confusion_matrix(y_true, y_pred)).to_csv(Path(cfg.result_dir)/"confusion_matrix.csv", index=False)
    summary = {"accuracy": accuracy_score(y_true,y_pred), "avg_inference_ms_per_image": sum(times)/len(times)}
    with open(Path(cfg.result_dir)/"evaluation_summary.json", "w") as f: json.dump(summary, f, indent=2)
    print(summary)

if __name__ == "__main__":
    p=argparse.ArgumentParser(); p.add_argument("--config", default=None); p.add_argument("--weights", default=None)
    args=p.parse_args(); cfg=Config.load(args.config) if args.config else Config(); evaluate_classifier(cfg, args.weights)
