# BoneAgeTW2

**Automated Skeletal Maturation Assessment using the Tanner-Whitehouse 2 Method and Deep Learning**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue)](https://python.org)
[![Dataset: RSNA](https://img.shields.io/badge/Dataset-RSNA%20Bone%20Age-green)](https://www.kaggle.com/datasets/kmader/rsna-bone-age)

> Master's Thesis — Artificial Intelligence · Universidad de La Salle, Bogotá, Colombia  
> Author: Juan Manuel Castillo Pinto · jmmana@gmail.com

---

## What is this?

The first open-source system that fully replicates the **Tanner-Whitehouse 2 (TW2)** protocol for bone age assessment:

- Automatically detects all **20 TW2 bones** in a hand X-ray (YOLOv8)
- Classifies each bone into its **maturation stage A→H/I** (EfficientNet-B3 × 20 heads)
- Computes **RUS + Carpal scores** using published TW2 tables
- Estimates **bone age** with 90% confidence intervals
- Shows **Gaussian bell curves** per bone (the original TW2 graphical output)
- REST API + Web interface + PDF report

Existing systems (Deeplasia, BoneXpert) are black boxes that output a single number. BoneAgeTW2 is transparent, explainable, and follows the clinical workflow.

---

## Dataset

**RSNA Pediatric Bone Age Challenge** (12,611 hand X-rays):  
https://www.kaggle.com/datasets/kmader/rsna-bone-age

---

## Quick Start

```bash
# 1. Clone
git clone https://github.com/jmmana/BoneAgeTW2.git
cd BoneAgeTW2

# 2. Install Python deps
pip install -r requirements.txt

# 3. Install frontend deps
cd frontend && npm install && cd ..

# 4. Start (backend on :8000, frontend on :5174)
./start.sh
```

Open **http://localhost:5174** — upload a hand X-ray, select sex, and get results.

---

## Architecture

```
BoneAgeTW2/
├── backend/                  # FastAPI API
│   ├── app/services/
│   │   ├── preprocessor.py   # DICOM/PNG → 512×512 tensor
│   │   ├── bone_detector.py  # YOLOv8 → 20 bone ROIs
│   │   ├── stage_classifier.py  # EfficientNet-B3 → stage A-H
│   │   ├── scorer.py         # TW2 tables → bone age
│   │   └── reporter.py       # Annotated image + PDF
│   └── ml/reference_data/
│       ├── tw2_tables.json   # TW2 published scores
│       └── gaussian_params.json  # Stage distribution params
├── frontend/                 # React + Vite + Recharts
├── training/                 # Pseudo-label generation + training scripts
└── paper/main.tex            # Academic paper (LaTeX)
```

---

## Training Pipeline

```bash
# 1. Download RSNA dataset to data/rsna/
# 2. Generate pseudo-labels
python training/02_pseudo_label_generation.py \
    --rsna_csv data/rsna/train.csv \
    --output data/annotations/pseudo_labels.csv

# 3. Train stage classifier (GPU/Colab recommended)
python training/04_train_stage_classifier.py \
    --rsna_dir data/rsna/train \
    --labels data/annotations/pseudo_labels.csv \
    --output backend/ml/weights/stage_classifier.pt
```

---

## API

```
POST /analyze
  → bone_age_months, rus_score, carpal_score,
     stages per bone, annotated image, gaussian data

POST /analyze/pdf
  → PDF clinical report

GET /reference/tw2-tables
GET /reference/gaussian-params
```

---

## Paper

Full academic paper available in [`paper/main.tex`](paper/main.tex)  
Also hosted on Overleaf: https://tesis.grimorio.dev/project/6a61765418d46ffd1fdd6619

---

## License

- Code: [MIT](LICENSE)
- Model weights (when released): CC BY-NC-SA 4.0
