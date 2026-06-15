from pathlib import Path
import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from .models.unet import UNet
from .preprocessing import read_rgb, resize_image, apply_clahe_rgb, save_image
from .utils import list_images, ensure_dirs, get_device


def bbox_from_mask(mask, min_area=20):
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None
    x1, y1, x2, y2 = xs.min(), ys.min(), xs.max(), ys.max()
    if (x2-x1+1) * (y2-y1+1) < min_area:
        return None
    return [int(x1), int(y1), int(x2), int(y2)]


def fuse_mask_bbox(img, mask, bbox=None, output_size=224):
    h, w = mask.shape
    if bbox is not None:
        x1, y1, x2, y2 = [max(0, int(v)) for v in bbox]
        x2, y2 = min(w-1, x2), min(h-1, y2)
        box_mask = np.zeros_like(mask, dtype=np.uint8); box_mask[y1:y2+1, x1:x2+1] = 1
        fused = (mask.astype(np.uint8) & box_mask)
        if fused.sum() == 0:
            fused = box_mask
    else:
        fused = mask.astype(np.uint8)
    bb = bbox_from_mask(fused) or bbox_from_mask(mask)
    if bb is None:
        roi = img
    else:
        x1,y1,x2,y2 = bb
        roi = img[y1:y2+1, x1:x2+1]
    roi = resize_image(roi, output_size)
    return apply_clahe_rgb(roi)


def predict_unet_mask(model, img_rgb, device, img_size=224, threshold=0.5):
    tf = transforms.Compose([transforms.ToPILImage(), transforms.Resize((img_size,img_size)), transforms.ToTensor(), transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
    x = tf(img_rgb).unsqueeze(0).to(device)
    with torch.no_grad():
        prob = torch.sigmoid(model(x))[0,0].cpu().numpy()
    return (prob > threshold).astype(np.uint8)


def load_yolo_model(weights=None):
    if weights is None or not Path(weights).exists():
        return None
    try:
        from ultralytics import YOLO
        return YOLO(str(weights))
    except Exception as e:
        print(f"YOLO not loaded: {e}")
        return None


def predict_yolo_bbox(yolo_model, image_path, img_size=224, conf=0.25):
    if yolo_model is None: return None
    res = yolo_model.predict(str(image_path), imgsz=img_size, conf=conf, verbose=False)
    if not res or res[0].boxes is None or len(res[0].boxes) == 0: return None
    boxes = res[0].boxes.xyxy.cpu().numpy(); scores = res[0].boxes.conf.cpu().numpy()
    return boxes[int(np.argmax(scores))].tolist()


def generate_roi_dataset(image_root, output_root, unet_weights=None, yolo_weights=None, img_size=224, device="auto"):
    ensure_dirs(output_root)
    device = get_device(device)
    unet = None
    if unet_weights and Path(unet_weights).exists():
        unet = UNet().to(device); ckpt = torch.load(unet_weights, map_location=device)
        unet.load_state_dict(ckpt.get("model", ckpt)); unet.eval()
    yolo = load_yolo_model(yolo_weights)
    for ip in list_images(image_root):
        rel = ip.relative_to(image_root)
        out = Path(output_root) / rel
        img = read_rgb(ip); img_rs = resize_image(img, img_size)
        mask = predict_unet_mask(unet, img_rs, device, img_size) if unet else np.ones((img_size,img_size), dtype=np.uint8)
        bbox = predict_yolo_bbox(yolo, ip, img_size) if yolo else bbox_from_mask(mask)
        roi = fuse_mask_bbox(img_rs, mask, bbox, img_size)
        save_image(roi, out)
    print(f"ROI dataset written to {output_root}")

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--image_root", required=True); p.add_argument("--output_root", required=True)
    p.add_argument("--unet_weights", default=None); p.add_argument("--yolo_weights", default=None); p.add_argument("--img_size", type=int, default=224)
    args = p.parse_args()
    generate_roi_dataset(**vars(args))
