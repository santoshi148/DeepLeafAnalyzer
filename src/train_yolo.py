import argparse
from pathlib import Path


def train_yolo(data_yaml: str, model: str = "yolov5s.pt", epochs: int = 50, imgsz: int = 224, batch: int = 16, project: str = "outputs/yolo"):
    """Train/fine-tune YOLO using Ultralytics.
    data_yaml must follow YOLO format with train/val paths and class names.
    """
    try:
        from ultralytics import YOLO
    except Exception as e:
        raise ImportError("Install ultralytics: pip install ultralytics") from e
    yolo = YOLO(model)
    results = yolo.train(data=data_yaml, epochs=epochs, imgsz=imgsz, batch=batch, project=project, name="deep_leaf_roi")
    return results

if __name__ == "__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--data_yaml", required=True)
    p.add_argument("--model", default="yolov5s.pt")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--imgsz", type=int, default=224)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--project", default="outputs/yolo")
    args=p.parse_args(); train_yolo(**vars(args))
