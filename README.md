# BoneAgeTW2

**Automated Skeletal Maturation Assessment using the Tanner-Whitehouse 2 (TW2) Method and Deep Learning**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688)](https://fastapi.tiangolo.com)
[![React](https://img.shields.io/badge/React-18-61DAFB)](https://react.dev)
[![Dataset: RSNA](https://img.shields.io/badge/Dataset-RSNA%20Bone%20Age-green)](https://www.kaggle.com/datasets/kmader/rsna-bone-age)

> **Master's Thesis** — Artificial Intelligence · Universidad de La Salle, Bogotá, Colombia  
> **Author:** Juan Manuel Castillo Pinto · jmmana@gmail.com  
> **Paper (Overleaf):** https://tesis.grimorio.dev/project/6a61765418d46ffd1fdd6619

---

## The Problem

Bone age assessment with the **Tanner-Whitehouse 2 (TW2)** method requires a trained radiologist to score **20 individual hand bones** across maturation stages **A → H/I**. It takes 10–20 minutes per case and is highly operator-dependent (±12 months inter-rater variability).

Existing AI solutions (Deeplasia, BoneXpert) are **black boxes** — they return a single bone age number with no bone-level explanation. No open-source software replicates the full TW2 pipeline.

**BoneAgeTW2 fills that gap.**

---

## What It Does

| Step | What happens |
|------|-------------|
| 1. Upload hand X-ray | DICOM, PNG or JPG accepted |
| 2. Preprocessing | CLAHE contrast enhancement, auto-flip to left hand, digital escalimetry from DICOM metadata |
| 3. Detect 20 bones | YOLOv8-s localizes each TW2 bone ROI; falls back to anatomical priors if no weights |
| 4. Classify each bone | EfficientNet-B3 with 20 heads → stage A/B/C/D/E/F/G/H(/I) per bone |
| 5. Compute TW2 score | RUS score (13 bones) + Carpal score (7 bones) → normalized to 0–1000 scale |
| 6. Estimate bone age | Sex-specific lookup tables (Tanner & Whitehouse 1983) + linear interpolation |
| 7. Show results | Annotated X-ray + score table + **Gaussian bell curves** per bone |
| 8. Download PDF | Clinical report with full TW2 breakdown |

---

## Architecture

```
BoneAgeTW2/
│
├── backend/                          FastAPI REST API
│   ├── app/
│   │   ├── main.py                   App entry point + CORS
│   │   ├── routers/analysis.py       POST /analyze, POST /analyze/pdf
│   │   └── services/
│   │       ├── preprocessor.py       DICOM/PNG → 512×512 normalized tensor
│   │       ├── bone_detector.py      YOLOv8 detector + anatomical priors fallback
│   │       ├── stage_classifier.py   EfficientNet-B3 × 20 heads → stage A-H/I
│   │       ├── scorer.py             TW2 tables → RUS/Carpal score → bone age
│   │       └── reporter.py           Annotated image (cv2) + PDF (reportlab)
│   │
│   └── ml/
│       ├── weights/                  Trained model weights (not in git, see below)
│       └── reference_data/
│           ├── tw2_tables.json       TW2 published numeric scores per stage
│           └── gaussian_params.json  Mean/SD of stage onset per bone/age/sex
│
├── frontend/                         React 18 + Vite + TypeScript
│   └── src/
│       ├── App.tsx                   Main layout + upload flow
│       ├── types.ts                  TypeScript API response types
│       └── components/
│           ├── XRayViewer.tsx        Annotated X-ray with 20 bounding boxes
│           ├── ScoreTable.tsx        TW2 score table + bone age summary
│           └── GaussCurves.tsx       Interactive Gaussian curves (Recharts)
│
├── training/
│   ├── 01_explore_rsna.ipynb         Data exploration + quality checks
│   ├── 02_pseudo_label_generation.py Assign TW2 stages from RSNA age labels
│   ├── 03_train_bone_detector.py     YOLOv8-s fine-tuning (TODO)
│   └── 04_train_stage_classifier.py  EfficientNet-B3 training
│
├── paper/main.tex                    Full academic paper (LaTeX)
├── requirements.txt                  Python dependencies
├── environment.yml                   Conda environment
└── start.sh                          Launch backend + frontend together
```

---

## The 20 TW2 Bones

### RUS Group (13 bones — primary score)
| Bone | Code | Stages |
|------|------|--------|
| Radius | `radius` | A–I |
| Ulna | `ulna` | A–I |
| Metacarpal I | `mc1` | A–I |
| Metacarpal III | `mc3` | A–I |
| Metacarpal V | `mc5` | A–I |
| Proximal phalanx I | `pp1` | A–I |
| Proximal phalanx III | `pp3` | A–I |
| Proximal phalanx V | `pp5` | A–I |
| Middle phalanx III | `mp3` | A–I |
| Middle phalanx V | `mp5` | A–I |
| Distal phalanx I | `dp1` | A–I |
| Distal phalanx III | `dp3` | A–I |
| Distal phalanx V | `dp5` | A–I |

### Carpal Group (7 bones — secondary score)
| Bone | Code | Stages |
|------|------|--------|
| Capitate | `capitate` | A–H |
| Hamate | `hamate` | A–H |
| Triquetral | `triquetral` | A–H |
| Lunate | `lunate` | A–H |
| Scaphoid | `scaphoid` | A–H |
| Trapezoid | `trapezoid` | A–H |
| Trapezium | `trapezium` | A–H |

---

## Quick Start

### Requirements
- Python 3.11+
- Node.js 18+
- (Optional) CUDA GPU for training

```bash
# Clone
git clone https://github.com/jmmana/BoneAgeTW2.git
cd BoneAgeTW2

# Python environment
conda env create -f environment.yml
conda activate boneage-tw2

# OR with pip
pip install -r requirements.txt

# Frontend
cd frontend && npm install && cd ..

# Launch both backend (:8000) and frontend (:5174)
./start.sh
```

Open **http://localhost:5174**

---

## API Reference

### `POST /analyze`

Upload a hand X-ray and get full TW2 analysis.

**Request** (`multipart/form-data`):
| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `image` | file | ✅ | DICOM, PNG, or JPG |
| `sex` | `"M"` or `"F"` | ✅ | Patient sex |
| `chronological_age_months` | float | ❌ | Used for Gaussian curve reference line |
| `scale_factor` | float | ❌ | mm/px (auto from DICOM if not provided) |

**Response** (JSON):
```json
{
  "bone_age_months": 132.5,
  "bone_age_years": 11.04,
  "confidence_interval": [121.4, 143.6],
  "rus_score": 620,
  "carpal_score": 580,
  "rus_age_months": 134.2,
  "carpal_age_months": 128.1,
  "sex": "M",
  "mm_per_px": 0.143,
  "stages": {"radius": "F", "ulna": "E", "mc1": "F", "...": "..."},
  "bone_scores": {"radius": 28, "ulna": 22, "...": "..."},
  "annotated_image_b64": "<base64 PNG>",
  "gaussian_data": {
    "radius": {
      "detected_stage": "F",
      "chrono_age_months": 144,
      "stages": [{"stage": "A", "mean": 0, "sd": 3}, "..."]
    }
  }
}
```

### `POST /analyze/pdf`

Same parameters as `/analyze`. Returns a PDF clinical report.

### `GET /reference/tw2-tables`

Returns the full TW2 numeric score tables (JSON).

### `GET /reference/gaussian-params`

Returns Gaussian distribution parameters for all bones (JSON).

### `GET /health`

Health check.

---

## Training Pipeline

### 1. Get the dataset

Download the RSNA Pediatric Bone Age Challenge dataset:  
https://www.kaggle.com/datasets/kmader/rsna-bone-age

Extract to `data/rsna/`:
```
data/rsna/
├── train/          (PNG images)
├── train.csv       (id, boneage, male)
└── test/
```

### 2. Generate pseudo-labels

```bash
python training/02_pseudo_label_generation.py \
    --rsna_csv data/rsna/train.csv \
    --output data/annotations/pseudo_labels.csv \
    --min_prob 0.0
```

This assigns a TW2 stage to each of the 20 bones for each of the 12,611 training images, using Gaussian reference distributions derived from published TW2 tables.

Output: `data/annotations/pseudo_labels.csv` (~250k rows: image_id × bone × stage)

### 3. Train the stage classifier (GPU recommended)

```bash
# Local GPU
python training/04_train_stage_classifier.py \
    --rsna_dir data/rsna/train \
    --labels data/annotations/pseudo_labels.csv \
    --output backend/ml/weights/stage_classifier.pt \
    --epochs 20 --batch 32

# Or on Kaggle (free GPU, dataset already available)
# Upload 04_train_stage_classifier.py as a Kaggle notebook
```

Training time: ~4h on a T4 GPU, ~30min on an A100.

### 4. Bone detection (YOLOv8)

The system works **without trained YOLOv8 weights** using anatomical priors.  
For training the bone detector (optional):
```bash
# TODO: 03_train_bone_detector.py — requires bounding box annotations
# Use the anatomical priors in bone_detector.py as the starting point
```

---

## How the Gaussian Curves Work

The TW2 method originally came with graphical reference curves showing how each bone stage progresses with age. BoneAgeTW2 reproduces this digitally.

For each bone `b`, sex `s`, and stage `k`, we store:
- `mean`: the age (months) at which 50% of children reach stage `k`
- `sd`: the standard deviation of that transition age

When the system classifies a bone into stage `F`, it highlights the `F` Gaussian curve and shows where the patient's chronological age falls relative to the reference population. **This is the core clinical value**: not just the bone age number, but the visual confirmation that the staging makes biological sense.

---

## Performance (Expected)

| Method | MAE (months) | Interpretable |
|--------|-------------|---------------|
| Greulich-Pyle (manual) | 11.8–14.5 | ❌ |
| Manual TW2 | 9.5–12.0 | ✅ |
| Deeplasia (end-to-end DL) | 6.9 | ❌ |
| **BoneAgeTW2 (ours)** | **10–14** | **✅** |

BoneAgeTW2 trades some raw accuracy for full clinical interpretability. The system is constrained by the TW2 protocol (discrete stages), not by model capacity.

---

## Without Model Weights

The system runs in **prior mode** without trained weights:
- Bone detection: uses hardcoded anatomical prior boxes (normalized positions)
- Stage classification: uses Gaussian-based age priors (requires `chronological_age_months` input)

This is useful for testing the interface, API, and visualization pipeline before training.

---

## Citation

```bibtex
@misc{castillo2026boneagetw2,
  title={BoneAgeTW2: Automated Skeletal Maturation Assessment Using the
         Tanner-Whitehouse 2 Method and Deep Learning},
  author={Castillo Pinto, Juan Manuel},
  year={2026},
  institution={Universidad de La Salle, Bogot\'a, Colombia},
  url={https://github.com/jmmana/BoneAgeTW2}
}
```

---

## References

- Tanner JM et al. *Assessment of Skeletal Maturity and Prediction of Adult Height (TW2 Method)*. Academic Press, 1983.
- Tanner JM et al. *TW3 Method*. Saunders, 2001.
- Halabi SS et al. [RSNA Pediatric Bone Age Challenge](https://doi.org/10.1148/radiol.2018180736). *Radiology*, 2019.
- Rassmann S et al. [Deeplasia](https://doi.org/10.1101/2023.03.07.23286906). *medRxiv*, 2023.
- Jocher G et al. [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics). 2023.
- Tan M, Le QV. [EfficientNet](https://proceedings.mlr.press/v97/tan19a.html). *ICML*, 2019.

---

## License

- **Code**: [MIT License](LICENSE)
- **Model weights** (when released): CC BY-NC-SA 4.0
- **RSNA Dataset**: [Kaggle Terms](https://www.kaggle.com/datasets/kmader/rsna-bone-age) — research use only
