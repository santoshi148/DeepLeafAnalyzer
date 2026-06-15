import argparse
from pathlib import Path
import cv2
import numpy as np
import torch
from PIL import Image
from .config import Config
from .augmentation import eval_transforms
from .models.hybrid_leaf_disease_net import HybridLeafDiseaseNet
from .utils import get_device, ensure_dirs

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model; self.target_layer = target_layer; self.activations=None; self.gradients=None
        target_layer.register_forward_hook(self._forward_hook)
        target_layer.register_full_backward_hook(self._backward_hook)
    def _forward_hook(self, module, inp, out): self.activations = out.detach()
    def _backward_hook(self, module, grad_in, grad_out): self.gradients = grad_out[0].detach()
    def __call__(self, x, class_idx=None):
        logits = self.model(x)
        if class_idx is None: class_idx = logits.argmax(1).item()
        self.model.zero_grad(set_to_none=True)
        logits[:, class_idx].sum().backward()
        weights = self.gradients.mean(dim=(2,3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = torch.relu(cam)
        cam = torch.nn.functional.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)
        cam = cam[0,0].cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam, class_idx

def make_gradcam(image_path, weights, out_path, cfg: Config):
    device = get_device(cfg.device)
    ckpt = torch.load(weights, map_location=device)
    class_to_idx = ckpt.get("class_to_idx", {})
    idx_to_class = {v:k for k,v in class_to_idx.items()}
    model = HybridLeafDiseaseNet(num_classes=len(class_to_idx), dropout=cfg.dropout).to(device)
    model.load_state_dict(ckpt["model"]); model.eval()
    pil = Image.open(image_path).convert("RGB")
    x = eval_transforms(cfg.img_size)(pil).unsqueeze(0).to(device)
    cam = GradCAM(model, model.refine[-1].net[0])
    heat, cls = cam(x)
    img = np.array(pil.resize((cfg.img_size, cfg.img_size)))
    heatmap = cv2.applyColorMap(np.uint8(255*heat), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(cv2.cvtColor(img, cv2.COLOR_RGB2BGR), 0.6, heatmap, 0.4, 0)
    out_path = Path(out_path); out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), overlay)
    return idx_to_class.get(cls, str(cls))

if __name__ == "__main__":
    p=argparse.ArgumentParser(); p.add_argument("--image", required=True); p.add_argument("--weights", default="outputs/checkpoints/best_hybridleaf.pth"); p.add_argument("--out", default="outputs/figures/gradcam.png"); p.add_argument("--config", default=None)
    args=p.parse_args(); cfg=Config.load(args.config) if args.config else Config()
    label = make_gradcam(args.image, args.weights, args.out, cfg); print(f"Saved Grad-CAM for {label}: {args.out}")
