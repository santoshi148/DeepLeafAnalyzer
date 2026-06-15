import argparse, json
from pathlib import Path
import torch
from PIL import Image
from torchvision import transforms
from .config import Config
from .augmentation import eval_transforms
from .models.hybrid_leaf_disease_net import HybridLeafDiseaseNet
from .utils import get_device, severity_from_confidence


def load_model(weights, cfg):
    device = get_device(cfg.device)
    ckpt = torch.load(weights, map_location=device)
    class_to_idx = ckpt.get("class_to_idx")
    if class_to_idx is None:
        with open(Path(cfg.result_dir)/"class_to_idx.json") as f: class_to_idx=json.load(f)
    idx_to_class = {v:k for k,v in class_to_idx.items()}
    model = HybridLeafDiseaseNet(num_classes=len(class_to_idx), dropout=cfg.dropout).to(device)
    model.load_state_dict(ckpt["model"]); model.eval()
    return model, idx_to_class, device


def predict_image(image_path, weights, cfg: Config):
    model, idx_to_class, device = load_model(weights, cfg)
    img = Image.open(image_path).convert("RGB")
    x = eval_transforms(cfg.img_size)(img).unsqueeze(0).to(device)
    with torch.no_grad():
        prob = torch.softmax(model(x), 1)[0].cpu()
    pred = int(prob.argmax()); conf = float(prob[pred])
    return {"image": str(image_path), "predicted_class": idx_to_class[pred], "confidence": conf, "severity": severity_from_confidence(conf, cfg.confidence_severe, cfg.confidence_moderate)}

if __name__ == "__main__":
    p=argparse.ArgumentParser(); p.add_argument("--image", required=True); p.add_argument("--weights", default="outputs/checkpoints/best_hybridleaf.pth"); p.add_argument("--config", default=None)
    args=p.parse_args(); cfg=Config.load(args.config) if args.config else Config()
    print(json.dumps(predict_image(args.image, args.weights, cfg), indent=2))
