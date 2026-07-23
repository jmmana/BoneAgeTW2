from __future__ import annotations
import base64
import io
import json
from pathlib import Path
import numpy as np
import cv2

from .bone_detector import STAGE_COLORS

_GAUSS = json.loads(
    (Path(__file__).parent.parent.parent / "ml" / "reference_data" / "gaussian_params.json").read_text()
)

BONE_LABELS = {
    "radius": "Radio", "ulna": "Cúbito",
    "mc1": "MC I", "mc3": "MC III", "mc5": "MC V",
    "pp1": "FF I", "pp3": "FF III", "pp5": "FF V",
    "mp3": "FM III", "mp5": "FM V",
    "dp1": "FD I", "dp3": "FD III", "dp5": "FD V",
    "capitate": "Grande", "hamate": "Ganchoso", "triquetral": "Piramidal",
    "lunate": "Semilunar", "scaphoid": "Escafoides",
    "trapezoid": "Trapezoides", "trapezium": "Trapecio",
}


def annotate_image(
    img_float: np.ndarray,
    detections: dict,
    classifications: dict,
) -> str:
    """Draw bounding boxes + stage labels on the X-ray. Returns base64 PNG."""
    img_u8 = (img_float * 255).clip(0, 255).astype(np.uint8)
    img_bgr = cv2.cvtColor(img_u8, cv2.COLOR_GRAY2BGR)

    for bone, det in detections.items():
        x1, y1, x2, y2 = det["box"]
        stage = classifications.get(bone, {}).get("stage", "?")
        hex_color = STAGE_COLORS.get(stage, "#FFFFFF")
        color = _hex_to_bgr(hex_color)
        cv2.rectangle(img_bgr, (x1, y1), (x2, y2), color, 2)
        label = f"{BONE_LABELS.get(bone, bone)} {stage}"
        cv2.putText(img_bgr, label, (x1, max(y1 - 4, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1, cv2.LINE_AA)

    _, buf = cv2.imencode(".png", img_bgr)
    return base64.b64encode(buf).decode()


def build_gaussian_data(classifications: dict, sex: str, chronological_age_months: float | None) -> dict:
    """
    Build data for the Gaussian curve plots.
    Returns {bone: {stages: [{stage, mean, sd, probability}], detected_stage, chrono_age}}
    """
    sex_key = "male" if sex == "M" else "female"
    output = {}
    for bone, clf in classifications.items():
        bone_params = _GAUSS.get(sex_key, {}).get(bone, {})
        stages_data = []
        for stage, params in bone_params.items():
            stages_data.append({
                "stage": stage,
                "mean": params["mean"],
                "sd": params["sd"],
                "probability": clf.get("probabilities", {}).get(stage, 0.0),
            })
        output[bone] = {
            "stages": stages_data,
            "detected_stage": clf.get("stage"),
            "chrono_age_months": chronological_age_months,
            "label": BONE_LABELS.get(bone, bone),
        }
    return output


def generate_pdf(result: dict) -> bytes:
    """Generate a PDF report of the analysis."""
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors
    from reportlab.lib.units import cm
    from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image as RLImage
    from reportlab.lib.styles import getSampleStyleSheet
    import tempfile, os

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, topMargin=1.5*cm, bottomMargin=1.5*cm)
    styles = getSampleStyleSheet()
    story = []

    # Title
    story.append(Paragraph("Informe de Maduración Ósea — Método TW2", styles["Title"]))
    story.append(Spacer(1, 0.4*cm))

    # Summary table
    summary_data = [
        ["Edad ósea estimada", f"{result.get('bone_age_years', '?')} años "
         f"({result.get('bone_age_months', '?')} meses)"],
        ["IC 90%", str(result.get("confidence_interval", "—"))],
        ["Score RUS", str(result.get("rus_score", "—"))],
        ["Score Carpal", str(result.get("carpal_score", "—"))],
        ["Sexo", result.get("sex", "—")],
    ]
    t = Table(summary_data, colWidths=[6*cm, 10*cm])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, -1), colors.lightgrey),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
    ]))
    story.append(t)
    story.append(Spacer(1, 0.5*cm))

    # Per-bone table
    story.append(Paragraph("Estadios por hueso", styles["Heading2"]))
    bone_data = [["Hueso", "Estadio", "Score", "Grupo"]]
    for bone, score in result.get("bone_scores", {}).items():
        from .bone_detector import BONE_PRIORS
        from ..services.scorer import _tables
        stage = result.get("stages", {}).get(bone, "?")
        group = _tables["bones"].get(bone, {}).get("group", "?")
        bone_data.append([BONE_LABELS.get(bone, bone), stage, str(score), group])

    bt = Table(bone_data, colWidths=[5*cm, 3*cm, 3*cm, 3*cm])
    bt.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1565C0")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F5F5F5")]),
    ]))
    story.append(bt)

    doc.build(story)
    return buf.getvalue()


def _hex_to_bgr(hex_color: str) -> tuple[int, int, int]:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return (b, g, r)
