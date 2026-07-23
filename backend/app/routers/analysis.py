from __future__ import annotations
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import Response
from typing import Optional
import json
from pathlib import Path

from ..services.preprocessor import preprocess
from ..services.bone_detector import BoneDetector
from ..services.stage_classifier import StageClassifier
from ..services.scorer import estimate_bone_age
from ..services.reporter import annotate_image, build_gaussian_data, generate_pdf

router = APIRouter(prefix="/analyze", tags=["analysis"])

_ML_DIR = Path(__file__).parent.parent.parent / "ml"
_detector = BoneDetector(weights_path=_ML_DIR / "weights" / "bone_detector.pt")
_classifier = StageClassifier(weights_path=_ML_DIR / "weights" / "stage_classifier.pt")


@router.post("")
async def analyze(
    image: UploadFile = File(...),
    sex: str = Form(..., pattern="^[MF]$"),
    scale_factor: Optional[float] = Form(None),
    chronological_age_months: Optional[float] = Form(None),
):
    raw = await image.read()
    if not raw:
        raise HTTPException(400, "Empty image file")

    # 1. Preprocess
    img_float, mm_per_px = preprocess(raw, scale_factor=scale_factor)

    # 2. Detect 20 bone ROIs
    detections = _detector.detect(img_float)
    rois = _detector.extract_rois(img_float, detections)

    # 3. Classify stages
    classifications = _classifier.classify_all(rois, sex, age_months=chronological_age_months)

    # 4. TW2 score → bone age
    stages = {bone: clf["stage"] for bone, clf in classifications.items()}
    result = estimate_bone_age(stages, sex)

    # 5. Annotated image (base64)
    annotated_b64 = annotate_image(img_float, detections, classifications)

    # 6. Gaussian data for frontend charts
    gaussian_data = build_gaussian_data(classifications, sex, chronological_age_months)

    return {
        **result,
        "sex": sex,
        "mm_per_px": round(mm_per_px, 4),
        "stages": stages,
        "classifications": {
            b: {"stage": c["stage"], "source": c.get("source"), "probabilities": c.get("probabilities")}
            for b, c in classifications.items()
        },
        "detections": {
            b: {"box": d["box"], "conf": d["conf"], "source": d.get("source")}
            for b, d in detections.items()
        },
        "annotated_image_b64": annotated_b64,
        "gaussian_data": gaussian_data,
    }


@router.post("/pdf")
async def analyze_pdf(
    image: UploadFile = File(...),
    sex: str = Form(..., pattern="^[MF]$"),
    scale_factor: Optional[float] = Form(None),
    chronological_age_months: Optional[float] = Form(None),
):
    """Same as /analyze but returns a PDF report."""
    raw = await image.read()
    img_float, mm_per_px = preprocess(raw, scale_factor=scale_factor)
    detections = _detector.detect(img_float)
    rois = _detector.extract_rois(img_float, detections)
    classifications = _classifier.classify_all(rois, sex, age_months=chronological_age_months)
    stages = {bone: clf["stage"] for bone, clf in classifications.items()}
    result = estimate_bone_age(stages, sex)
    result["sex"] = sex
    result["stages"] = stages

    pdf_bytes = generate_pdf(result)
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": "attachment; filename=bone_age_report.pdf"},
    )
