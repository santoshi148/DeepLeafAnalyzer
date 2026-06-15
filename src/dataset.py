from pathlib import Path
import random
from typing import Dict, List, Tuple, Optional
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import train_test_split
from .augmentation import train_transforms, eval_transforms, mask_transforms
from .utils import IMG_EXTS


def discover_classes(root: str) -> Dict[str, int]:
    root = Path(root)
    classes = sorted([p.name for p in root.iterdir() if p.is_dir()])
    return {c: i for i, c in enumerate(classes)}


def collect_classification_samples(root: str, class_to_idx: Optional[Dict[str, int]] = None):
    root = Path(root)
    if class_to_idx is None:
        class_to_idx = discover_classes(root)
    samples = []
    for cls, idx in class_to_idx.items():
        for p in (root / cls).rglob("*"):
            if p.suffix.lower() in IMG_EXTS:
                samples.append((str(p), idx))
    if not samples:
        raise RuntimeError(f"No class-folder images found in {root}. Expected raw/class_name/*.jpg")
    return samples, class_to_idx


class LeafClassificationDataset(Dataset):
    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, torch.tensor(label, dtype=torch.long), path


class SegmentationDataset(Dataset):
    def __init__(self, image_dir: str, mask_dir: str, img_size=224):
        self.image_paths = sorted([p for p in Path(image_dir).rglob("*") if p.suffix.lower() in IMG_EXTS])
        self.mask_dir = Path(mask_dir)
        self.img_tf = eval_transforms(img_size)
        self.mask_tf = mask_transforms(img_size)
        self.pairs = []
        for ip in self.image_paths:
            candidates = [self.mask_dir / f"{ip.stem}.png", self.mask_dir / ip.name, self.mask_dir / ip.relative_to(image_dir)]
            mp = next((c for c in candidates if c.exists()), None)
            if mp:
                self.pairs.append((ip, mp))
        if not self.pairs:
            raise RuntimeError("No image-mask pairs found. Masks must match image stem/name.")
    def __len__(self): return len(self.pairs)
    def __getitem__(self, idx):
        ip, mp = self.pairs[idx]
        img = Image.open(ip).convert("RGB")
        mask = Image.open(mp).convert("L")
        img = self.img_tf(img)
        mask = self.mask_tf(mask)
        mask = (mask > 0.5).float()
        return img, mask, str(ip)


def make_splits(samples, val_split=0.1, test_split=0.1, seed=42):
    labels = [s[1] for s in samples]
    train_samples, temp_samples = train_test_split(samples, test_size=val_split+test_split, stratify=labels, random_state=seed)
    temp_labels = [s[1] for s in temp_samples]
    rel_test = test_split / (val_split + test_split)
    val_samples, test_samples = train_test_split(temp_samples, test_size=rel_test, stratify=temp_labels, random_state=seed)
    return train_samples, val_samples, test_samples


def make_classification_loaders(root, img_size=224, batch_size=32, val_split=0.1, test_split=0.1, seed=42, num_workers=2):
    samples, class_to_idx = collect_classification_samples(root)
    train_s, val_s, test_s = make_splits(samples, val_split, test_split, seed)
    loaders = {
        "train": DataLoader(LeafClassificationDataset(train_s, train_transforms(img_size)), batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True),
        "val": DataLoader(LeafClassificationDataset(val_s, eval_transforms(img_size)), batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True),
        "test": DataLoader(LeafClassificationDataset(test_s, eval_transforms(img_size)), batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True),
    }
    return loaders, class_to_idx
