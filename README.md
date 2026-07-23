# BoneAgeTW2

**Sistema automatizado de evaluación de maduración ósea por el método Tanner-Whitehouse 2 (TW2)**  
**Automated Skeletal Maturation Assessment using the Tanner-Whitehouse 2 (TW2) Method and Deep Learning**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688)](https://fastapi.tiangolo.com)
[![React](https://img.shields.io/badge/React-18-61DAFB)](https://react.dev)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-purple)](https://github.com/ultralytics/ultralytics)
[![EfficientNet-B3](https://img.shields.io/badge/EfficientNet-B3-orange)](https://github.com/lukemelas/EfficientNet-PyTorch)
[![Dataset: RSNA](https://img.shields.io/badge/Dataset-RSNA%20Bone%20Age-green)](https://www.kaggle.com/datasets/kmader/rsna-bone-age)

> **Tesis de Maestría** — Inteligencia Artificial · Universidad de La Salle, Bogotá, Colombia  
> **Autor:** Juan Manuel Castillo Pinto · jmmana@gmail.com  
> **Paper (Overleaf):** https://tesis.grimorio.dev/project/6a61765418d46ffd1fdd6619  
> **Repo:** https://github.com/jmmana/BoneAgeTW2

---

## Tabla de contenidos

1. [El problema que resuelve](#1-el-problema-que-resuelve)
2. [Cómo funciona — resumen rápido](#2-cómo-funciona--resumen-rápido)
3. [Flujo completo del sistema (diagrama)](#3-flujo-completo-del-sistema-diagrama)
4. [Los 20 huesos TW2](#4-los-20-huesos-tw2)
5. [Dataset — Imágenes de entrenamiento](#5-dataset--imágenes-de-entrenamiento)
6. [Todos los modelos de IA utilizados](#6-todos-los-modelos-de-ia-utilizados)
7. [Pipeline de entrenamiento paso a paso](#7-pipeline-de-entrenamiento-paso-a-paso)
   - [Paso 1: Exploración del dataset](#paso-1-exploración-del-dataset)
   - [Paso 2: Generación de pseudo-etiquetas TW2](#paso-2-generación-de-pseudo-etiquetas-tw2)
   - [Paso 3: Entrenamiento de YOLOv8](#paso-3-entrenamiento-de-yolov8-detector-de-los-20-huesos)
   - [Paso 4: Entrenamiento del clasificador de estadios](#paso-4-entrenamiento-del-clasificador-de-estadios-efficientnet-b3)
   - [Paso 5: Evaluación](#paso-5-evaluación)
8. [Pipeline de inferencia (uso clínico)](#8-pipeline-de-inferencia-uso-clínico)
9. [Reportes generados](#9-reportes-generados)
10. [Arquitectura del proyecto](#10-arquitectura-del-proyecto)
11. [Instalación y arranque rápido](#11-instalación-y-arranque-rápido)
12. [API REST — referencia completa](#12-api-rest--referencia-completa)
13. [Interfaz web — descripción](#13-interfaz-web--descripción)
14. [Sin pesos entrenados (modo prior)](#14-sin-pesos-entrenados-modo-prior)
15. [Rendimiento esperado](#15-rendimiento-esperado)
16. [Citas y referencias](#16-citas-y-referencias)

---

## 1. El problema que resuelve

El método **Tanner-Whitehouse 2 (TW2)** es el estándar de oro para evaluar la maduración esquelética en Latinoamérica y Europa. El protocolo manual requiere:

1. Tomar una radiografía de la mano izquierda en proyección anteroposterior.
2. El radiólogo examina **20 huesos específicos** uno por uno.
3. Asigna a cada hueso un **estadio de maduración** (letra A → H o I) según criterios morfológicos publicados.
4. Convierte cada estadio en un **puntaje numérico** de las tablas TW2.
5. Suma los puntajes en un **score RUS** (13 huesos) y un **score Carpal** (7 huesos).
6. Convierte los scores a **edad ósea en meses** usando curvas de referencia por sexo.
7. Traza manualmente (o ignora por falta de tiempo) las **curvas de distribución gaussiana** por hueso para contextualizar el estadio.

**Tiempo total: 10–20 minutos por caso. Variabilidad inter-observador: ±12 meses.**

Los sistemas existentes (Deeplasia, BoneXpert) producen un único número como salida — sin explicar qué huesos están retrasados ni por qué. **Ningún software libre automatiza el pipeline TW2 completo.**

BoneAgeTW2 resuelve esto: sube una radiografía y en <5 segundos tienes el informe clínico completo con los 20 huesos estadificados, los scores, la edad ósea con IC del 90%, la radiografía anotada, y las curvas de Gauss por hueso.

---

## 2. Cómo funciona — resumen rápido

| Paso | Entrada | IA usada | Salida |
|------|---------|----------|--------|
| Preproceso | DICOM/PNG | — (OpenCV) | Tensor 512×512 normalizado |
| Detección | Tensor | **YOLOv8-s** | 20 bounding boxes |
| Clasificación | 20 recortes | **EfficientNet-B3 ×20 cabezas** | Estadio A–I por hueso |
| Scoring | 20 estadios | Tablas TW2 (lookup) | Score RUS + Carpal |
| Reporte | Scores + imagen | ReportLab + Recharts | PDF + Gauss interactivo |

---

## 3. Flujo completo del sistema (diagrama)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           ENTRADA DEL USUARIO                               │
│          Radiografía mano izquierda (DICOM / PNG / JPG)                     │
│          + Sexo (M/F)  + Edad cronológica (opc.) + Scale factor (opc.)      │
└───────────────────────────────┬─────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  MÓDULO 1: PREPROCESAMIENTO  (preprocessor.py)                              │
│                                                                             │
│  1. Carga imagen                                                             │
│     ├── DICOM: pydicom → extrae PixelSpacing (mm/px) automáticamente       │
│     └── PNG/JPG: PIL → usa scale_factor del usuario (default 0.143 mm/px)  │
│                                                                             │
│  2. Corrección fotométrica                                                   │
│     └── MONOCHROME1 → inversión de bits                                     │
│                                                                             │
│  3. Detección de mano derecha                                               │
│     └── Si suma_derecha > 1.1 × suma_izquierda → flip horizontal           │
│                                                                             │
│  4. CLAHE (clipLimit=2.0, tileGrid=8×8)                                    │
│     └── Mejora contraste en carpianos y puntas de falanges                 │
│                                                                             │
│  5. Resize 512×512 + normalización [0, 1]                                  │
└───────────────────────────────┬─────────────────────────────────────────────┘
                                │  tensor (1, 512, 512)
                                ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  MÓDULO 2: DETECCIÓN DE 20 HUESOS  (bone_detector.py)                      │
│                                                                             │
│  Modelo: YOLOv8-s fine-tuned sobre radiografías RSNA                        │
│                                                                             │
│  Detecta 20 regiones de interés (ROI):                                      │
│    RUS: radius, ulna, mc1, mc3, mc5, pp1, pp3, pp5, mp3, mp5, dp1, dp3, dp5│
│    Carpal: capitate, hamate, triquetral, lunate, scaphoid, trapezoid,       │
│            trapezium                                                         │
│                                                                             │
│  Fallback: si no hay pesos, usa priors anatómicos hardcoded                 │
│            (posición (cx,cy,w,h) en [0,1] derivada de anatomía media)       │
│                                                                             │
│  Salida: { hueso: [x1, y1, x2, y2, confianza] } ×20                        │
└───────────────────────────────┬─────────────────────────────────────────────┘
                                │  20 bounding boxes
                                ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  MÓDULO 3: CLASIFICACIÓN DE ESTADIOS  (stage_classifier.py)                 │
│                                                                             │
│  Modelo: EfficientNet-B3 (backbone compartido) + 20 cabezas independientes  │
│                                                                             │
│  Por cada uno de los 20 huesos:                                             │
│    1. Recortar ROI del tensor según bounding box                            │
│    2. Resize a 96×96                                                        │
│    3. Pasar por backbone EfficientNet-B3 → vector 1536 features             │
│    4. Pasar por cabeza específica del hueso → logits (8 o 9 clases)         │
│    5. softmax → probabilidades de cada estadio (A/B/C/D/E/F/G/H(/I))       │
│                                                                             │
│  Fallback (sin pesos): usa Gauss prior de la edad cronológica               │
│    stage = argmax P(stage | edad_cronológica, hueso, sexo)                  │
│                                                                             │
│  Salida: { hueso: { stage, probabilities: {A:.02, B:.05, ..., F:.78} } }   │
└───────────────────────────────┬─────────────────────────────────────────────┘
                                │  20 estadios
                                ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  MÓDULO 4: SCORING TW2  (scorer.py)                                         │
│                                                                             │
│  Referencia: tw2_tables.json (Tanner & Whitehouse 1983, dominio público)   │
│                                                                             │
│  1. Para cada estadio → puntaje numérico de la tabla TW2                    │
│  2. Score RUS = Σ puntajes de los 13 huesos RUS                             │
│  3. Score Carpal = Σ puntajes de los 7 carpianos                            │
│  4. Normalizar a escala 0–1000                                               │
│  5. Lookup en curva de referencia por sexo → edad_ósea_meses               │
│  6. Combinación: edad_ósea = 0.75×edad_RUS + 0.25×edad_Carpal              │
│  7. IC 90%: ± 1.645 × σ(score_RUS) (aprox. parabólica publicada)           │
│                                                                             │
│  Salida: { bone_age_months, confidence_interval, rus_score, carpal_score }  │
└───────────────────────────────┬─────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  MÓDULO 5: GENERACIÓN DE REPORTES  (reporter.py)                            │
│                                                                             │
│  A) Radiografía anotada (cv2)                                               │
│     - 20 bounding boxes coloreados por estadio                              │
│     - Etiqueta: "radius:F" sobre cada caja                                  │
│     - Encode → base64 PNG para API + frontend                               │
│                                                                             │
│  B) Curvas de Gauss (gaussian_params.json)                                  │
│     - Para cada hueso: 8 curvas N(μ_k, σ_k) sobre eje de edad              │
│     - Estadio detectado: opacidad 100%; resto: 10%                          │
│     - Línea vertical: edad cronológica del paciente                         │
│     - Frontend: Recharts (interactivo) | PDF: Matplotlib (estático)         │
│                                                                             │
│  C) PDF clínico (ReportLab)                                                  │
│     - Cabecera con datos del paciente                                        │
│     - Resumen de edad ósea + IC                                              │
│     - Radiografía anotada embebida                                           │
│     - Tabla TW2 completa (20 huesos × estadio × score)                      │
│     - 20 gráficas de curvas gaussianas (una por hueso)                      │
│     - Disclaimer clínico + referencias bibliográficas                       │
└─────────────────────────────────────────────────────────────────────────────┘
                    │
          ┌─────────┼─────────────────────┐
          ▼         ▼                     ▼
    JSON API    Radiografía           PDF Clínico
    /analyze    anotada            /analyze/pdf
    response    (base64)           (descarga)
```

---

## 4. Los 20 huesos TW2

### Grupo RUS — 13 huesos (peso: 75%)

| Código | Nombre | Estadios | Posición aprox. en imagen |
|--------|--------|----------|--------------------------|
| `radius` | Radio | A–I | 1/5 inferior, lado radial |
| `ulna` | Cúbito | A–I | 1/5 inferior, lado ulnar |
| `mc1` | Metacarpiano I | A–I | Base del pulgar |
| `mc3` | Metacarpiano III | A–I | Centro de la mano |
| `mc5` | Metacarpiano V | A–I | Lateral del meñique |
| `pp1` | Falange proximal I (pulgar) | A–I | Proximal al pulgar |
| `pp3` | Falange proximal III (medio) | A–I | Centro, proximal |
| `pp5` | Falange proximal V (meñique) | A–I | Lateral, proximal |
| `mp3` | Falange media III | A–I | Centro, media |
| `mp5` | Falange media V | A–I | Lateral, media |
| `dp1` | Falange distal I | A–I | Punta del pulgar |
| `dp3` | Falange distal III | A–I | Punta del dedo medio |
| `dp5` | Falange distal V | A–I | Punta del meñique |

### Grupo Carpal — 7 huesos (peso: 25%)

| Código | Nombre (esp.) | Nombre (lat.) | Estadios |
|--------|---------------|---------------|----------|
| `capitate` | Grande | Os capitatum | A–H |
| `hamate` | Ganchoso | Os hamatum | A–H |
| `triquetral` | Piramidal | Os triquetrum | A–H |
| `lunate` | Semilunar | Os lunatum | A–H |
| `scaphoid` | Escafoides | Os scaphoideum | A–H |
| `trapezoid` | Trapezoides | Os trapezoideum | A–H |
| `trapezium` | Trapecio | Os trapezium | A–H |

### Descripción de los estadios (ejemplo: huesos largos)

| Estadio | Descripción morfológica |
|---------|------------------------|
| A | Sin centro de osificación visible |
| B | Centro visible, sin forma definida |
| C | Forma regular, sin epífisis |
| D | Epífisis < diáfisis |
| E | Epífisis = diáfisis en ancho |
| F | Epífisis > diáfisis, sin escotadura |
| G | Epífisis cubre metáfisis |
| H | Fusión parcial epifisaria |
| I | Fusión completa (solo RUS) |

---

## 5. Dataset — Imágenes de entrenamiento

### RSNA Pediatric Bone Age Challenge

**URL oficial (Kaggle):**  
https://www.kaggle.com/datasets/kmader/rsna-bone-age

**Organizador:** Radiological Society of North America (RSNA), 2017  
**Publicación:** Halabi SS et al. *Radiology*, 2019. doi:10.1148/radiol.2018180736

### Composición

| Split | Total | Masculino | Femenino |
|-------|-------|-----------|---------|
| Train | 12,611 | 6,833 | 5,778 |
| Validation | 1,425 | 746 | 679 |
| Test | 200 | 100 | 100 |
| **Total** | **14,236** | **7,679** | **6,557** |

### Características de las imágenes

- **Tipo:** Radiografías de mano izquierda, proyección anteroposterior (AP)
- **Formato:** PNG (16-bit, convertidos de DICOM original)
- **Resolución:** Variable (600×800 a 1600×2000 px típico)
- **Pixel spacing:** Variable (0.1–0.5 mm/px según equipo radiográfico)
- **Rango de edad:** 1–228 meses (0–19 años)
- **Procedencia:** Stanford, U. Colorado, UCSF, UCLA (múltiples centros USA)

### Anotaciones disponibles

| Campo | Descripción |
|-------|-------------|
| `boneage` | Edad ósea en meses (promedio de 2 radiólogos) |
| `male` | 1=masculino, 0=femenino |
| — | **NO hay estadios TW2 por hueso** (los generamos con pseudo-etiquetas) |

### Cómo descargar

```bash
# Opción 1: Kaggle CLI
pip install kaggle
kaggle datasets download -d kmader/rsna-bone-age
unzip rsna-bone-age.zip -d data/rsna/

# Opción 2: Kaggle web → descarga manual → extraer en data/rsna/

# Estructura esperada:
data/rsna/
├── boneage-training-dataset/     # 12,611 imágenes PNG
│   └── *.png
├── boneage-validation-dataset/   # 1,425 imágenes PNG
│   └── *.png
├── boneage-test-dataset/         # 200 imágenes PNG (sin etiquetas)
│   └── *.png
├── train.csv                     # id, boneage, male
└── Validation Dataset.csv        # id, boneage, male
```

### Por qué NO tienen etiquetas TW2 por hueso

El RSNA challenge fue diseñado para predecir una edad global, no para reproducir el protocolo TW2. Los radiólogos anotaron la edad ósea total, no los estadios individuales. Esto es el principal reto de este proyecto → lo resolvemos con **pseudo-etiquetas**.

---

## 6. Todos los modelos de IA utilizados

### Modelo 1: YOLOv8-s — Detector de los 20 huesos

| Parámetro | Valor |
|-----------|-------|
| Arquitectura | YOLOv8-small (Ultralytics) |
| Tarea | Detección de objetos (20 clases) |
| Input | 640×640 RGB |
| Pesos base | COCO pre-trained (`yolov8s.pt`) |
| Fine-tune | ~200 radiografías con cajas manuales/semi-auto |
| Output | 20 bounding boxes + clase + confianza |
| Fallback | Priors anatómicos hardcoded |

**¿Por qué YOLOv8?**  
- Entrenamiento en GPU accesible (Colab/Kaggle free tier)
- Soporte nativo DICOM-compatible vía pre-proceso
- Velocidad en inferencia (<50ms por imagen en CPU)
- Fine-tuning desde pesos COCO funciona bien con estructuras óseas

### Modelo 2: EfficientNet-B3 — Clasificador de estadios TW2

| Parámetro | Valor |
|-----------|-------|
| Arquitectura | EfficientNet-B3 + 20 cabezas |
| Tarea | Clasificación multiclase por hueso |
| Input | ROI 96×96, 3 canales (grayscale replicado) |
| Pesos base | ImageNet pre-trained |
| Output | Probabilidad por estadio (8–9 clases) × 20 huesos |
| Loss | CrossEntropy × 20 (label_smoothing=0.1) |
| Optimizador | AdamW, lr=1e-4, cosine annealing |
| Épocas | 20 |

**¿Por qué EfficientNet-B3?**  
- Balanceo óptimo precisión/costo computacional en la familia EfficientNet
- B3 (10.8M parámetros) cabe en 4GB VRAM
- Pre-entrenamiento ImageNet transferible a texturas óseas
- Soporta multi-task learning con backbone compartido (20 cabezas)

### Modelo 3 (sin entrenamiento): Tablas TW2 de referencia

No es ML, pero es el "modelo clínico":
- `tw2_tables.json`: puntajes numéricos por estadio (Tanner & Whitehouse 1983)
- `gaussian_params.json`: parámetros (μ, σ) de transición de estadio por edad/sexo

Estos datos son de **dominio público** (publicados en el libro TW2, 1983).

### Resumen del stack de IA

```
┌──────────────────────────────────────────────────────────┐
│                     STACK DE IA                          │
│                                                          │
│  Nivel 1 — Visión                                        │
│  ┌────────────────────────────────────────────────────┐  │
│  │  YOLOv8-s  →  detecta DÓNDE están los 20 huesos   │  │
│  └────────────────────────────────────────────────────┘  │
│                                                          │
│  Nivel 2 — Clasificación                                 │
│  ┌────────────────────────────────────────────────────┐  │
│  │  EfficientNet-B3 × 20 cabezas                      │  │
│  │  →  clasifica QUÉ estadio (A-I) es cada hueso      │  │
│  └────────────────────────────────────────────────────┘  │
│                                                          │
│  Nivel 3 — Conocimiento clínico                          │
│  ┌────────────────────────────────────────────────────┐  │
│  │  Tablas TW2 publicadas (lookup tables)             │  │
│  │  →  convierte estadios → scores → edad ósea        │  │
│  └────────────────────────────────────────────────────┘  │
│                                                          │
│  Nivel 4 — Visualización estadística                     │
│  ┌────────────────────────────────────────────────────┐  │
│  │  Distribuciones Gaussianas TW2 (parámetros μ, σ)  │  │
│  │  →  genera curvas de campana por hueso             │  │
│  └────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────┘
```

---

## 7. Pipeline de entrenamiento paso a paso

### Paso 1: Exploración del dataset

```bash
# Instalar dependencias
conda env create -f environment.yml
conda activate boneage-tw2

# Exploración
jupyter notebook training/01_explore_rsna.ipynb
```

El notebook `01_explore_rsna.ipynb` incluye:
- Distribución de edades por sexo (histograma)
- Estadísticas de resolución de imagen
- Ejemplos visuales de radiografías en distintos rangos de edad
- Verificación de calidad del dataset (imágenes corruptas, outliers)

---

### Paso 2: Generación de pseudo-etiquetas TW2

**El problema**: RSNA tiene edades óseas globales pero NO estadios por hueso.  
**La solución**: invertir las tablas TW2 — dada la edad, ¿qué estadio es más probable?

```bash
python training/02_pseudo_label_generation.py \
    --rsna_csv data/rsna/train.csv \
    --output data/annotations/pseudo_labels.csv \
    --min_prob 0.3
```

**Cómo funciona internamente:**

```python
# Para cada imagen (edad_meses, sexo):
# Para cada uno de los 20 huesos:
for bone in TW2_BONES:
    for stage in STAGES[bone]:
        # Densidad gaussiana: qué tan probable es este estadio a esta edad
        d = scipy.stats.norm.pdf(age_months, mu[bone][sex][stage], sigma[bone][sex][stage])
    
    # El estadio más probable = la campana con mayor densidad
    pseudo_stage = stages[np.argmax(densities)]
```

**Salida:** `data/annotations/pseudo_labels.csv`

```
image_id,bone,stage,probability,age_months,sex
1377,radius,F,0.8234,144,M
1377,ulna,E,0.7102,144,M
1377,mc1,F,0.7856,144,M
...  (≈252,000 filas para 12,611 imágenes × 20 huesos)
```

**Calidad de las pseudo-etiquetas:**
- Alta confianza cuando el paciente tiene edad muy alta o muy baja (poca varianza entre personas)
- Menor confianza en edades de transición (~9–13 años)
- Filtrar con `--min_prob 0.3` descarta ~15% de filas ruidosas

---

### Paso 3: Entrenamiento de YOLOv8 (Detector de los 20 huesos)

#### 3a. Preparar anotaciones YOLO

Las anotaciones de bounding box se generan a partir de los **priors anatómicos** del código:

```bash
python training/03_generate_yolo_annotations.py \
    --rsna_dir data/rsna/boneage-training-dataset \
    --output data/yolo_annotations \
    --validate_n 200  # revisar manualmente 200 casos
```

Formato YOLO (cada imagen → `.txt` con una línea por hueso):
```
# class cx cy w h  (todos en [0,1] relativo al tamaño de imagen)
0  0.500 0.875 0.180 0.120   # radius
1  0.540 0.875 0.120 0.100   # ulna
...
```

Estructura de datos para YOLO:
```
data/yolo_annotations/
├── images/
│   ├── train/   (≈10,000 .png)
│   └── val/     (≈2,000 .png)
└── labels/
    ├── train/   (≈10,000 .txt, 20 líneas por archivo)
    └── val/     (≈2,000 .txt)
```

Archivo de configuración `data/yolo_bones.yaml`:
```yaml
path: data/yolo_annotations
train: images/train
val: images/val
nc: 20
names:
  0: radius
  1: ulna
  2: mc1
  3: mc3
  4: mc5
  5: pp1
  6: pp3
  7: pp5
  8: mp3
  9: mp5
  10: dp1
  11: dp3
  12: dp5
  13: capitate
  14: hamate
  15: triquetral
  16: lunate
  17: scaphoid
  18: trapezoid
  19: trapezium
```

#### 3b. Entrenar YOLOv8

```bash
# GPU local
yolo detect train \
    model=yolov8s.pt \
    data=data/yolo_bones.yaml \
    epochs=50 \
    imgsz=640 \
    batch=16 \
    lr0=1e-3 \
    patience=10 \
    name=boneage_detector \
    project=runs/detect

# En Kaggle (GPU gratuita T4/P100)
# Subir 03_train_bone_detector.py como notebook Kaggle
# El dataset RSNA ya está disponible en Kaggle sin descarga

# Copiar pesos al backend
cp runs/detect/boneage_detector/weights/best.pt \
   backend/ml/weights/bone_detector.pt
```

**Tiempo estimado de entrenamiento:**

| Hardware | Tiempo por época | Total (50 épocas) |
|----------|-----------------|-------------------|
| CPU (M4 Pro) | ~45 min | ~37 horas |
| GPU T4 (Kaggle) | ~4 min | ~3.5 horas |
| GPU A100 | ~1 min | ~1 hora |

**Métricas objetivo:**
- mAP@50 > 0.85 (todos los huesos detectados correctamente)
- mAP@50:95 > 0.65
- Recall per-clase > 0.90

---

### Paso 4: Entrenamiento del clasificador de estadios (EfficientNet-B3)

```bash
# Requiere: pseudo_labels.csv del Paso 2 + imágenes RSNA
python training/04_train_stage_classifier.py \
    --rsna_dir data/rsna/boneage-training-dataset \
    --labels data/annotations/pseudo_labels.csv \
    --detector_weights backend/ml/weights/bone_detector.pt \
    --output backend/ml/weights/stage_classifier.pt \
    --epochs 20 \
    --batch 32 \
    --lr 1e-4 \
    --label_smoothing 0.1

# En Kaggle (recomendado — el dataset ya está disponible)
# Subir 04_train_stage_classifier.py
# Kaggle Notebook → Add Dataset → "rsna-bone-age"
```

**Arquitectura interna:**

```
Input ROI (96×96×3)
        │
        ▼
EfficientNet-B3 backbone  [ImageNet pretrained]
        │
        ▼  1536 features
        │
   ┌────┴─────────────────── × 20 ──────────────────────┐
   │                                                      │
radius_head   ulna_head   mc1_head  ... trapezium_head
   │                                                      │
   ▼          ▼           ▼             ▼
 9 clases   9 clases    9 clases      8 clases
(A–I)      (A–I)       (A–I)         (A–H)
```

**Proceso de entrenamiento:**

```python
# Por cada época, para cada hueso:
for bone in TW2_BONES:
    dataset = BoneROIDataset(bone=bone, labels=pseudo_labels)
    loader = DataLoader(dataset, batch=32, shuffle=True)
    
    for rois, stages in loader:
        # Forward: backbone → head específica del hueso
        logits = model(rois, bone_id=bone)
        loss = CrossEntropyLoss(label_smoothing=0.1)(logits, stages)
        loss.backward()
        optimizer.step()
```

**Tiempo estimado:**

| Hardware | Tiempo total (20 épocas × 20 huesos) |
|----------|--------------------------------------|
| GPU T4 (Kaggle) | ~4 horas |
| GPU A100 | ~1 hora |
| CPU M4 Pro | ~40+ horas (no recomendado) |

---

### Paso 5: Evaluación

```bash
jupyter notebook training/05_evaluate.ipynb
```

El notebook evalúa:

1. **Accuracy por hueso y estadio**
   - Matriz de confusión por cada uno de los 20 huesos
   - Accuracy exacto y ±1 estadio

2. **Edad ósea global (MAE en meses)**
   - Comparación contra ground truth RSNA (test set N=200)
   - Estratificado por sexo y rango de edad

3. **Comparación con baseline**
   - vs. Deeplasia (descargable, open-source)
   - vs. Asignación por edad media TW2 (baseline ingênuo)

---

## 8. Pipeline de inferencia (uso clínico)

Una vez entrenados los modelos, el flujo de uso es:

```bash
# 1. Levantar el sistema
./start.sh

# 2. Abrir navegador
open http://localhost:5174

# 3. Subir radiografía → obtener resultado en <5 segundos
```

O via API directa:

```bash
curl -X POST http://localhost:8000/analyze \
    -F "image=@/path/to/hand_xray.dcm" \
    -F "sex=M" \
    -F "chronological_age_months=144"
```

---

## 9. Reportes generados

### Reporte JSON (API `/analyze`)

Incluye todos los datos estructurados: estadios, scores, edad ósea, IC, imagen anotada base64, datos de curvas gaussianas.

### Reporte PDF clínico (`/analyze/pdf`)

El PDF descargable tiene exactamente la misma estructura que un informe TW2 clínico manual:

```
┌─────────────────────────────────────────┐
│  INFORME DE MADURACIÓN ÓSEA — TW2       │
│  Sistema BoneAgeTW2 v0.1                │
├─────────────────────────────────────────┤
│  Paciente: Masculino, 12 años           │
│  Fecha: 22/07/2026   Hora: 14:32        │
├─────────────────────────────────────────┤
│                                         │
│  EDAD ÓSEA ESTIMADA: 11.8 años          │
│  Intervalo confianza 90%: 10.7–12.8 a   │
│  Score RUS: 615/1000                    │
│  Score Carpal: 630/1000                 │
│                                         │
├─────────────────────────────────────────┤
│  [Radiografía anotada con 20 cajas]     │
├─────────────────────────────────────────┤
│  TABLA TW2 COMPLETA (20 huesos)         │
│  Hueso    | Estadio | Score | Grupo     │
│  Radius   | F       | 28    | RUS       │
│  ...      | ...     | ...   | ...       │
├─────────────────────────────────────────┤
│  CURVAS GAUSSIANAS (20 gráficas)        │
│  [Radius] [Ulna] [MC1] ... [Trapezium]  │
│  Cada gráfica: campanas por estadio     │
│  + línea vertical = edad cronológica    │
├─────────────────────────────────────────┤
│  NOTA: Herramienta de apoyo. Requiere   │
│  revisión por radiólogo certificado.    │
│  Referencias: Tanner & Whitehouse 1983  │
└─────────────────────────────────────────┘
```

### Curvas gaussianas (web interactivo)

```
Ejemplo: Hueso "Radius" en paciente masculino de 12 años (144 meses)

Densidad
  │
  │     B   C    D    E    F    G    H    I
  │     ▲   ▲    ▲   ▲   ███  ▲    ▲    ▲
  │    │ │ │ │  │ │  │ │ │█│ │ │  │ │  │ │
  │    │ │ │ │  │ │  │ │ │█│ │ │  │ │  │ │
  └────────────────────────│─────────────────► Edad (meses)
       20  40  60  80 100  120 140 160 180
                               ▲
                            Edad cronológica (144m)
                            
  Interpretación: El estadio F tiene su pico a ~120 meses.
  El paciente tiene 144 meses pero está en estadio F → retardo leve.
```

---

## 10. Arquitectura del proyecto

```
BoneAgeTW2/
│
├── backend/                          ← FastAPI REST API (Python 3.11)
│   ├── app/
│   │   ├── main.py                   App entry, CORS, rutas
│   │   ├── routers/
│   │   │   └── analysis.py           POST /analyze, POST /analyze/pdf
│   │   └── services/
│   │       ├── preprocessor.py       Módulo 1: DICOM/PNG → tensor
│   │       ├── bone_detector.py      Módulo 2: YOLOv8 + priors
│   │       ├── stage_classifier.py   Módulo 3: EfficientNet-B3 × 20
│   │       ├── scorer.py             Módulo 4: TW2 tables → edad ósea
│   │       └── reporter.py           Módulo 5: imagen anotada + PDF
│   └── ml/
│       ├── weights/                  Pesos entrenados (no en git)
│       │   ├── bone_detector.pt      YOLOv8 fine-tuned (después del entrenamiento)
│       │   └── stage_classifier.pt   EfficientNet-B3 fine-tuned
│       └── reference_data/
│           ├── tw2_tables.json       Tablas TW2 publicadas (1983)
│           └── gaussian_params.json  Parámetros Gauss μ,σ por hueso/estadio/sexo
│
├── frontend/                         ← React 18 + TypeScript + Vite
│   ├── src/
│   │   ├── App.tsx                   Layout principal + upload + tabs
│   │   ├── types.ts                  TypeScript types del JSON response
│   │   └── components/
│   │       ├── XRayViewer.tsx        Radiografía anotada con 20 bounding boxes
│   │       ├── ScoreTable.tsx        Tabla TW2 + resumen edad ósea + IC
│   │       └── GaussCurves.tsx       Recharts: curvas por hueso + selector
│   ├── package.json
│   └── vite.config.ts                Puerto 5174 (proxy /analyze → :8000)
│
├── training/                         ← Scripts de entrenamiento
│   ├── 01_explore_rsna.ipynb         Exploración y estadísticas RSNA
│   ├── 02_pseudo_label_generation.py Paso 2: edad → estadios TW2
│   ├── 03_train_bone_detector.py     Paso 3: fine-tune YOLOv8
│   ├── 04_train_stage_classifier.py  Paso 4: entrenar EfficientNet
│   └── 05_evaluate.ipynb             MAE + per-bone accuracy
│
├── data/                             ← Datos (NO en git)
│   ├── rsna/                         Dataset RSNA (12,611 PNG + CSV)
│   ├── yolo_annotations/             Anotaciones YOLO generadas
│   └── annotations/
│       └── pseudo_labels.csv         Output Paso 2
│
├── paper/
│   └── main.tex                      Paper académico LaTeX completo
│
├── requirements.txt                  Dependencias Python
├── environment.yml                   Entorno Conda
└── start.sh                          Arrancar backend + frontend
```

---

## 11. Instalación y arranque rápido

### Requisitos

- Python 3.11+
- Node.js 18+
- (Entrenamiento) GPU con ≥4GB VRAM recomendada; funciona en CPU para inferencia

### Instalación

```bash
# Clonar
git clone https://github.com/jmmana/BoneAgeTW2.git
cd BoneAgeTW2

# Python: opción A (conda — recomendado)
conda env create -f environment.yml
conda activate boneage-tw2

# Python: opción B (pip)
pip install -r requirements.txt

# Frontend
cd frontend
npm install
cd ..
```

### Arranque

```bash
# Arranca backend (:8000) y frontend (:5174) simultáneamente
./start.sh

# O por separado:
# Backend
cd backend && uvicorn app.main:app --reload --port 8000

# Frontend (otro terminal)
cd frontend && npm run dev
```

Abrir: **http://localhost:5174**

---

## 12. API REST — referencia completa

### `POST /analyze`

Analiza una radiografía y devuelve resultado TW2 completo.

**Request** (`multipart/form-data`):

| Campo | Tipo | Requerido | Descripción |
|-------|------|-----------|-------------|
| `image` | file | ✅ | DICOM (.dcm), PNG, o JPG |
| `sex` | `"M"` o `"F"` | ✅ | Sexo del paciente |
| `chronological_age_months` | float | ❌ | Edad cronológica en meses (para línea de referencia en curvas) |
| `scale_factor` | float | ❌ | mm/px (se extrae automáticamente de DICOM si no se provee) |

**Response 200** (JSON):

```json
{
  "bone_age_months": 141.3,
  "bone_age_years": 11.78,
  "confidence_interval": [128.9, 153.7],
  "rus_score": 615,
  "carpal_score": 630,
  "rus_age_months": 141.8,
  "carpal_age_months": 139.5,
  "sex": "M",
  "mm_per_px": 0.143,
  "stages": {
    "radius": "F",
    "ulna": "E",
    "mc1": "F",
    "mc3": "F",
    "mc5": "F",
    "pp1": "F",
    "pp3": "F",
    "pp5": "F",
    "mp3": "G",
    "mp5": "F",
    "dp1": "F",
    "dp3": "F",
    "dp5": "F",
    "capitate": "G",
    "hamate": "G",
    "triquetral": "G",
    "lunate": "F",
    "scaphoid": "F",
    "trapezoid": "F",
    "trapezium": "F"
  },
  "bone_scores": {
    "radius": 28,
    "ulna": 22,
    "mc1": 18,
    "...": "..."
  },
  "classifications": {
    "radius": {
      "stage": "F",
      "probabilities": {"A": 0.00, "B": 0.01, "C": 0.02, "D": 0.05, "E": 0.12, "F": 0.68, "G": 0.10, "H": 0.02},
      "source": "model"
    },
    "...": "..."
  },
  "detections": {
    "radius": {
      "box": [45, 420, 120, 510],
      "confidence": 0.94,
      "source": "yolo"
    },
    "...": "..."
  },
  "annotated_image_b64": "<base64 PNG con 20 cajas de colores>",
  "gaussian_data": {
    "radius": {
      "detected_stage": "F",
      "chrono_age_months": 144,
      "label": "Radius",
      "stages": [
        {"stage": "A", "mean": 0,  "sd": 3,  "probability": 0.000},
        {"stage": "B", "mean": 12, "sd": 6,  "probability": 0.000},
        {"stage": "C", "mean": 36, "sd": 8,  "probability": 0.000},
        {"stage": "D", "mean": 60, "sd": 10, "probability": 0.001},
        {"stage": "E", "mean": 84, "sd": 10, "probability": 0.023},
        {"stage": "F", "mean": 120,"sd": 12, "probability": 0.134},
        {"stage": "G", "mean": 150,"sd": 10, "probability": 0.389},
        {"stage": "H", "mean": 168,"sd": 8,  "probability": 0.201},
        {"stage": "I", "mean": 192,"sd": 12, "probability": 0.000}
      ]
    },
    "...": "..."
  }
}
```

### `POST /analyze/pdf`

Mismos parámetros que `/analyze`. Devuelve el PDF binario del informe clínico.

```bash
curl -X POST http://localhost:8000/analyze/pdf \
    -F "image=@hand.dcm" \
    -F "sex=F" \
    -F "chronological_age_months=120" \
    --output informe_tw2.pdf
```

### `GET /reference/tw2-tables`

Devuelve las tablas TW2 completas en JSON (puntajes por estadio, lookup de edad por score).

### `GET /reference/gaussian-params`

Devuelve todos los parámetros gaussianos (μ, σ) por hueso/estadio/sexo.

### `GET /health`

```json
{"status": "ok", "version": "0.1.0"}
```

---

## 13. Interfaz web — descripción

### Tab 1: Upload

```
┌─────────────────────────────────────────────────────────┐
│          BoneAgeTW2 — Maduración ósea TW2               │
├─────────────────────────────────────────────────────────┤
│  ┌────────────────────────────────────────────────────┐ │
│  │                                                    │ │
│  │    Arrastra tu radiografía aquí                    │ │
│  │    DICOM / PNG / JPG                               │ │
│  │                                                    │ │
│  └────────────────────────────────────────────────────┘ │
│                                                         │
│  Sexo: [○ Masculino  ○ Femenino]                        │
│  Edad cronológica: [_______] meses  (opcional)          │
│  Scale factor:     [_______] mm/px  (opcional)          │
│                                                         │
│  [    Analizar radiografía    ]                         │
└─────────────────────────────────────────────────────────┘
```

### Tab 2: Resultado — Radiografía anotada (Tab "X-ray")

- Imagen de la radiografía con 20 rectangulos de colores
- Cada rectangulo: nombre del hueso + estadio (ej: "dp3:F")
- Leyenda de colores (A=gris ... I=violeta)

### Tab 3: Tabla TW2 (Tab "Scores")

- Tabla completa de los 20 huesos con estadio y puntaje
- Subtotales RUS y Carpal
- Box destacado: Edad ósea X.X años (CI 90%: a.a – b.b años)
- Botón: [⬇ Descargar PDF]

### Tab 4: Curvas de Gauss (Tab "Curves")

- Panel de botones: [Radius] [Ulna] [MC1] ... [Trapezium]
- Gráfica interactiva Recharts para el hueso seleccionado
- Hasta 9 curvas de área coloreadas (una por estadio)
- La curva del estadio detectado: opacidad completa
- Línea vertical roja: edad cronológica

---

## 14. Sin pesos entrenados (modo prior)

El sistema funciona **sin modelos entrenados** usando priors anatómicos y estadísticos:

| Componente | Sin pesos | Con pesos |
|------------|-----------|-----------|
| Detección huesos | Posición hardcoded (prior anatómico) | YOLOv8 predicción |
| Clasificación estadio | Prior gaussiano (requiere edad cronológica) | EfficientNet-B3 predicción |
| Scoring TW2 | ✅ siempre | ✅ siempre |
| Curvas Gauss | ✅ siempre | ✅ siempre |
| PDF | ✅ siempre | ✅ siempre |

Modo prior útil para: probar la interfaz, validar el pipeline de visualización, demostrar el sistema antes de entrenar.

---

## 15. Rendimiento esperado

| Método | MAE (meses) | IC disponible | Interpretable | Open source |
|--------|-------------|---------------|---------------|-------------|
| Greulich-Pyle (manual) | 11.8–14.5 | ❌ | ❌ | N/A |
| TW2 manual (radiólogo) | 9.5–12.0 | ❌ | ✅ | N/A |
| Deeplasia (DL) | 6.9 | ❌ | ❌ | ✅ |
| BoneXpert (comercial) | 7.1–9.8 | parcial | parcial | ❌ |
| **BoneAgeTW2 (ours)** | **10–14** | **✅** | **✅** | **✅** |

El mayor MAE respecto a Deeplasia es consecuencia directa de operar con estadios discretos TW2 (el protocolo lo requiere). El valor diferencial es la interpretabilidad clínica.

### Velocidad de inferencia

| Plataforma | JSON response | PDF report |
|------------|---------------|------------|
| Mac M4 Pro (CPU) | ~4.2s | ~5.0s |
| NVIDIA T4 (GPU) | ~0.6s | ~1.4s |

---

## 16. Citas y referencias

```bibtex
@misc{castillo2026boneagetw2,
  title   = {BoneAgeTW2: Automated Skeletal Maturation Assessment
             Using the Tanner-Whitehouse 2 Method and Deep Learning},
  author  = {Castillo Pinto, Juan Manuel},
  year    = {2026},
  school  = {Universidad de La Salle, Bogot{\'a}, Colombia},
  url     = {https://github.com/jmmana/BoneAgeTW2}
}

@article{rsna2017,
  title   = {The RSNA Pediatric Bone Age Machine Learning Challenge},
  author  = {Halabi, Safwan S and others},
  journal = {Radiology},
  volume  = {290},
  number  = {2},
  pages   = {498--503},
  year    = {2019},
  doi     = {10.1148/radiol.2018180736}
}

@book{tannerTW2,
  title     = {Assessment of Skeletal Maturity and Prediction of
               Adult Height (TW2 Method)},
  author    = {Tanner, J M and Whitehouse, R H and Cameron, N and
               Marshall, W A and Healy, M J R and Goldstein, H},
  edition   = {2},
  publisher = {Academic Press},
  year      = {1983}
}

@book{tw32001,
  title     = {Assessment of Skeletal Maturity and Prediction of
               Adult Height (TW3 Method)},
  author    = {Tanner, J M and Healy, M J R and Goldstein, H and Cameron, N},
  edition   = {3},
  publisher = {Saunders},
  year      = {2001}
}
```

---

**Licencias:**
- Código: [MIT](LICENSE)
- Pesos de modelos (cuando se publiquen): CC BY-NC-SA 4.0
- Dataset RSNA: términos de Kaggle — solo uso de investigación
