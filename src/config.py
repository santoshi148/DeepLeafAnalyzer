from dataclasses import dataclass, asdict
from pathlib import Path
import json

@dataclass
class Config:
    project_root: str = "."
    raw_dir: str = "data/raw"
    processed_dir: str = "data/processed"
    mask_dir: str = "data/masks"
    yolo_label_dir: str = "data/annotations/yolo"
    roi_dir: str = "data/processed/roi"
    checkpoint_dir: str = "outputs/checkpoints"
    result_dir: str = "outputs/results"
    figure_dir: str = "outputs/figures"
    img_size: int = 224
    num_classes: int = 38
    batch_size: int = 32
    epochs: int = 100
    lr: float = 1e-4
    weight_decay: float = 1e-4
    dropout: float = 0.3
    seed: int = 42
    num_workers: int = 2
    val_split: float = 0.10
    test_split: float = 0.10
    unet_epochs: int = 50
    confidence_severe: float = 0.85
    confidence_moderate: float = 0.60
    use_amp: bool = True
    device: str = "auto"

    def abs_path(self, relative: str) -> Path:
        return Path(self.project_root) / relative

    def save(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, indent=2)

    @staticmethod
    def load(path: str):
        with open(path, "r", encoding="utf-8") as f:
            return Config(**json.load(f))

CFG = Config()
