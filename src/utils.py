import os, random, time
from pathlib import Path
import numpy as np
import torch

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

def seed_everything(seed: int = 42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

def get_device(device="auto"):
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)

def list_images(root):
    root = Path(root)
    return sorted([p for p in root.rglob("*") if p.suffix.lower() in IMG_EXTS])

def ensure_dirs(*dirs):
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)

def now_ms():
    return time.perf_counter() * 1000.0

def severity_from_confidence(conf, severe=0.85, moderate=0.60):
    if conf >= severe:
        return "Severe"
    if conf >= moderate:
        return "Moderate"
    return "Mild"
