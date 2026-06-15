from pathlib import Path
import cv2
import numpy as np
from PIL import Image


def read_rgb(path):
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def resize_image(img, size=224):
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)


def median_denoise(img, ksize=3):
    return cv2.medianBlur(img, ksize)


def gaussian_denoise(img, ksize=3):
    return cv2.GaussianBlur(img, (ksize, ksize), 0)


def apply_clahe_rgb(img, clip_limit=2.0, tile_grid_size=(8, 8)):
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    l2 = clahe.apply(l)
    lab = cv2.merge((l2, a, b))
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


def normalize_01(img):
    return img.astype(np.float32) / 255.0


def preprocess_image(path_or_img, size=224, denoise=True, clahe=False):
    img = read_rgb(path_or_img) if not isinstance(path_or_img, np.ndarray) else path_or_img.copy()
    img = resize_image(img, size)
    if denoise:
        img = median_denoise(img, 3)
    if clahe:
        img = apply_clahe_rgb(img)
    return img


def save_image(img, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(img.astype(np.uint8)).save(path)
