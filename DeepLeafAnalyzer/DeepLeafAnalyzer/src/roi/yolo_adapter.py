from pathlib import Path

class YOLODetector:
    """Optional Ultralytics adapter.

    For a manuscript-faithful YOLOv5 experiment, pass a locally trained compatible checkpoint.
    If Ultralytics is unavailable, construction raises a clear ImportError; the rest of the project remains usable.
    """
    def __init__(self, weights, conf=0.25):
        try:
            from ultralytics import YOLO
        except Exception as e:
            raise ImportError('Install ultralytics to use YOLODetector: pip install ultralytics') from e
        self.model=YOLO(str(weights)); self.conf=conf
    def predict_bbox(self, image):
        results=self.model.predict(image, conf=self.conf, verbose=False)
        if not results or results[0].boxes is None or len(results[0].boxes)==0:
            return None, 0.0
        boxes=results[0].boxes
        i=int(boxes.conf.argmax().item())
        xyxy=boxes.xyxy[i].detach().cpu().numpy().tolist()
        conf=float(boxes.conf[i].item())
        return xyxy, conf
