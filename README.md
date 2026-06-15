# 🌿 DeepLeafAnalyzer

## DeepLeafAnalyzer: A Hybrid Deep Learning Framework for ROI-Optimized Multi-Scale Leaf Disease Detection

DeepLeafAnalyzer is a hybrid deep learning framework developed for accurate and robust plant leaf disease detection under both laboratory-controlled and real-world agricultural environments.

The framework integrates dual-stage Region of Interest (ROI) optimization, multi-scale feature learning, attention mechanisms, and hybrid CNN-LSTM classification to improve disease localization, classification, and severity estimation.

Unlike conventional CNN-based disease classification systems that process the entire image, DeepLeafAnalyzer first identifies disease-relevant regions using **U-Net segmentation** and **YOLOv5 object detection**. The extracted ROI is then processed by the proposed **HybridLeafDiseaseNet** architecture consisting of:

- Feature Pyramid Networks (FPN)
- Atrous Spatial Pyramid Pooling (ASPP)
- Vision Transformer (ViT) Attention
- Channel-Spatial Attention Module (CSAM)
- CNN-LSTM Learning

The framework is designed for precision agriculture applications, enabling:

- Early disease diagnosis
- Improved crop monitoring
- Explainable AI-based decision support

---

## 🚀 Key Features

- Dual-stage ROI extraction using U-Net and YOLOv5
- ROI fusion for precise disease localization
- Multi-scale feature extraction using FPN and ASPP
- Vision Transformer self-attention mechanism
- Channel and Spatial Attention Module (CSAM)
- Hybrid CNN-LSTM disease classification
- Disease severity estimation
- Grad-CAM explainability
- Real-world deployment support
- Modular and reproducible research pipeline

---

## 📊 System Workflow

```text
Input Leaf Image
        │
        ▼
Preprocessing
        │
        ▼
ROI Optimization
 ├── U-Net Segmentation
 └── YOLOv5 Detection
        │
        ▼
ROI Fusion
        │
        ▼
Multi-Scale Feature Learning
 ├── FPN
 └── ASPP
        │
        ▼
Attention Refinement
 ├── Vision Transformer
 └── CSAM
        │
        ▼
Hybrid CNN-LSTM
        │
        ▼
Classification Layer
        │
        ▼
Disease Class + Confidence + Severity
```

---

## 📂 Supported Datasets

| Dataset | Description |
|----------|-------------|
| PlantVillage | Laboratory-controlled crop disease dataset |
| FieldPlant | Real-world field disease images |
| AI Challenger 2018 | Large-scale crop disease benchmark dataset |

---

## 📁 Project Structure

```text
DeepLeafAnalyzer/
│
├── data/
│   ├── raw/
│   ├── processed/
│   ├── masks/
│   └── annotations/
│
├── src/
│   ├── config.py
│   ├── dataset.py
│   ├── preprocessing.py
│   ├── augmentation.py
│   ├── train_unet.py
│   ├── train_yolo.py
│   ├── roi_fusion.py
│   ├── train_classifier.py
│   ├── evaluate.py
│   ├── gradcam.py
│   ├── inference.py
│   │
│   └── models/
│       ├── unet.py
│       ├── fpn_aspp.py
│       ├── attention.py
│       └── hybrid_leaf_disease_net.py
│
├── outputs/
│   ├── checkpoints/
│   ├── logs/
│   ├── figures/
│   └── results/
│
├── requirements.txt
├── README.md
└── main.py
```

---

## ⚙️ Installation

### Clone Repository

```bash
git clone https://github.com/yourusername/DeepLeafAnalyzer.git
cd DeepLeafAnalyzer
```

### Create Environment

```bash
conda create -n deepleaf python=3.10
conda activate deepleaf
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 📦 Main Dependencies

```text
torch>=2.1.0
torchvision>=0.16.0
opencv-python
numpy
pandas
matplotlib
scikit-learn
albumentations
tqdm
ultralytics
segmentation-models-pytorch
grad-cam
Pillow
```

---

## 🗂 Dataset Preparation

```text
data/raw/

├── PlantVillage/
│   ├── disease_1/
│   ├── disease_2/
│   └── ...
│
├── FieldPlant/
│   ├── disease_1/
│   ├── disease_2/
│   └── ...
│
└── AIChallenger/
    ├── disease_1/
    ├── disease_2/
    └── ...
```

### ROI Extraction Files

```text
data/masks/
    image001.png
    image002.png

data/annotations/
    image001.txt
    image002.txt
```

---

## 🏋️ Training Pipeline

### Step 1: Train U-Net Segmentation

```bash
python src/train_unet.py
```

Output:

```text
outputs/checkpoints/best_unet.pth
```

### Step 2: Train YOLOv5 Detector

```bash
python src/train_yolo.py
```

Output:

```text
outputs/checkpoints/best_yolo.pt
```

### Step 3: ROI Extraction and Fusion

```bash
python src/roi_fusion.py
```

Output:

```text
data/processed/roi_images/
```

### Step 4: Train HybridLeafDiseaseNet

```bash
python src/train_classifier.py
```

Output:

```text
outputs/checkpoints/best_hybridleaf.pth
```

---

## 📈 Evaluation

```bash
python src/evaluate.py
```

### Metrics

- Accuracy
- Precision
- Recall
- F1-Score
- IoU
- Dice Score
- Inference Time
- FLOPs

Output:

```text
outputs/results/
```

---

## 🔍 Explainability

Generate Grad-CAM visualizations:

```bash
python src/gradcam.py
```

Output:

```text
outputs/figures/gradcam/
```

---

## 🧠 Inference

```bash
python src/inference.py --image sample_leaf.jpg
```

Example Output:

```text
Disease Class : Tomato Early Blight
Confidence    : 0.94
Severity      : Severe
```

---

## 📉 Disease Severity Estimation

| Confidence Score | Severity |
|------------------|----------|
| ≥ 0.85 | Severe |
| 0.60 – 0.84 | Moderate |
| < 0.60 | Mild |

---

## 🔬 Experimental Configuration

| Parameter | Value |
|------------|--------|
| Input Resolution | 224 × 224 |
| Optimizer | Adam |
| Learning Rate | 0.0001 |
| Batch Size | 32 |
| Epochs | 100 |
| Early Stopping | 10 |
| Dropout | 0.3 |
| Weight Decay | 1e-4 |

---

## 📊 Generated Outputs

| File | Description |
|--------|-------------|
| best_unet.pth | Trained U-Net model |
| best_yolo.pt | Trained YOLO model |
| best_hybridleaf.pth | Final disease classifier |
| classification_report.csv | Classification metrics |
| roi_metrics.csv | ROI localization metrics |
| confusion_matrix.png | Confusion matrix |
| gradcam_results/ | Explainability visualizations |
| inference_results.csv | Prediction results |

---

## 🎯 Research Contributions

1. Dual-stage ROI optimization using segmentation and object detection.
2. Multi-scale disease representation using FPN and ASPP.
3. Attention-guided disease localization using ViT and CSAM.
4. Hybrid CNN-LSTM architecture for advanced feature learning.
5. Explainable disease diagnosis through Grad-CAM.
6. Real-time deployable AI framework for precision agriculture.
7. Improved robustness across laboratory and field conditions.

---

## 🔄 Reproducibility

- Fixed random seeds for NumPy, PyTorch, and Python.
- Version-controlled model checkpoints.
- Configuration-driven training.
- Modular implementation.
- Complete preprocessing pipeline included.
- Experiment logs automatically saved.

---

## 🚀 Future Extensions

- Mobile deployment using TensorFlow Lite.
- Edge AI deployment on Jetson Nano and Raspberry Pi.
- Disease progression forecasting.
- Federated learning for distributed farm intelligence.
- Integration with IoT and drone-based monitoring systems.
- Multi-modal crop health assessment using weather and soil data.

---

## 📖 Citation

```bibtex
@article{DeepLeafAnalyzer2026,
  title={DeepLeafAnalyzer: A Hybrid Deep Learning Framework for ROI-Optimized Multi-Scale Leaf Disease Detection},
  author={Author Names},
  journal={Under Review},
  year={2026}
}
```

---

## 📜 License

This project is released under the MIT License.

---

## 🙏 Acknowledgements

We acknowledge the PlantVillage, FieldPlant, and AI Challenger communities for providing publicly accessible crop disease datasets that facilitated the development and evaluation of the DeepLeafAnalyzer framework.

---

## 📬 Contact

For research collaboration, implementation support, or issue reporting, please create a GitHub Issue in this repository.
