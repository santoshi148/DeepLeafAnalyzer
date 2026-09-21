# DeepLeafAnalyzer

Reproducible PyTorch implementation scaffold for the manuscript **DeepLeafAnalyzer: A Hybrid Deep Learning Framework for ROI-Optimized Multi-Scale Leaf Disease Detection**.

## What is implemented
- Leakage-free 80/10/10 stratified splitting
- Deterministic preprocessing + training-only augmentation
- U-Net segmentation network
- Optional YOLO-family detector adapter for local lesion-detector checkpoints
- Dual-stage ROI fusion with safe fallbacks
- Residual CNN backbone
- Feature Pyramid Network (FPN)
- Atrous Spatial Pyramid Pooling (ASPP)
- ViT-style multi-head spatial self-attention
- Channel-Spatial Attention Module (CSAM)
- LSTM feature-sequence encoder (spatial/multi-scale sequence, not temporal disease progression)
- Classification training/evaluation
- ROI metrics, calibration utility, bootstrap CI, efficiency utility
- Grad-CAM implementation
- Ablation-friendly model flags
- Unit/smoke tests

## Important scientific interpretation
Softmax probability is treated as **model confidence**, not biological disease severity. The LSTM processes ordered feature positions and does not claim longitudinal disease progression.

## Installation
```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

## Quick smoke test
```bash
python scripts/demo_smoke.py
```
Expected final line: `SMOKE TEST PASSED`.

## Dataset manifest
Create a CSV containing at minimum:
```csv
image_path,class_id,class_name
images/a.jpg,0,healthy
images/b.jpg,1,disease_a
```
Then split before any augmentation:
```bash
python -m src.data.split_dataset --manifest data/manifest.csv --outdir data/splits
```

## Train classifier
```bash
python scripts/train_classifier.py \
  --train data/splits/train.csv \
  --val data/splits/val.csv \
  --root . \
  --config configs/default.yaml \
  --outdir outputs/classifier
```

## Evaluate
```bash
python scripts/evaluate.py \
  --test data/splits/test.csv \
  --checkpoint outputs/classifier/best.pt \
  --root . \
  --config configs/default.yaml
```

## ROI annotations
For full manuscript reproduction, prepare pixel masks for U-Net and lesion bounding boxes for YOLO. The included `YOLODetector` in `src/roi/yolo_adapter.py` accepts a locally trained Ultralytics-compatible checkpoint. The manuscript's stated YOLOv5 choice should be reproduced with a YOLOv5-compatible local training setup/checkpoint; this repository intentionally avoids downloading weights at runtime.

## Suggested annotation layout
```text
data/<dataset>/
  images/
  masks/
  labels_yolo/
  manifest.csv
```

## Reproducibility defaults
- seed: 42
- image size: 224 x 224
- Adam LR: 1e-4
- batch size: 32
- maximum epochs: 100
- ReduceLROnPlateau factor: 0.5
- scheduler patience: 5
- early-stopping patience: 10
- dropout: 0.3
- weight decay: 1e-4 (implementation choice because the manuscript text omits the numerical value)

## Recommended experiment sequence
1. Prepare separate manifests for PlantVillage, FieldPlant, and AI Challenger.
2. Split each dataset before augmentation.
3. Annotate a representative ROI subset with pixel masks and boxes.
4. Train U-Net segmentation.
5. Train/fine-tune lesion detector.
6. Generate ROI crops using U-Net + detector fusion.
7. Train `HybridLeafDiseaseNet` on ROI crops.
8. Evaluate baselines, ablations, cross-validation, robustness, calibration, Grad-CAM, and efficiency.

## Notes
- The code is CPU compatible; CUDA is used automatically when available.
- No dataset or pretrained detector weights are bundled.
- The architecture is configuration-driven and can be downsized for Colab/CPU experiments.

## Functional synthetic demo
To verify the training path without downloading agricultural datasets:
```bash
python scripts/make_demo_dataset.py
python scripts/train_classifier.py --train demo_data/splits/train.csv --val demo_data/splits/val.csv --root . --config configs/demo.yaml --outdir outputs/demo --epochs 1
python scripts/evaluate.py --test demo_data/splits/test.csv --checkpoint outputs/demo/best.pt --root . --config configs/demo.yaml
```
This synthetic demo validates software execution only; it is not a scientific result.
