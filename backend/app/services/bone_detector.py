from __future__ import annotations
import numpy as np
from pathlib import Path
from typing import Optional

# ── Anatomical priors ────────────────────────────────────────────────────────
# Relative bounding boxes for each TW2 bone in a normalized 512×512 left-hand X-ray.
# Format: (cx, cy, w, h) all in [0,1] relative to image size.
# These are anatomical priors derived from median positions in the RSNA dataset.
# Used as fallback when no trained YOLOv8 weights are available.

BONE_PRIORS: dict[str, tuple[float, float, float, float]] = {
    # Wrist/forearm
    "radius":     (0.50, 0.08, 0.30, 0.12),
    "ulna":       (0.30, 0.08, 0.18, 0.12),
    # Metacarpals
    "mc1":        (0.78, 0.28, 0.12, 0.14),
    "mc3":        (0.55, 0.26, 0.10, 0.16),
    "mc5":        (0.28, 0.26, 0.10, 0.16),
    # Proximal phalanges
    "pp1":        (0.80, 0.44, 0.10, 0.11),
    "pp3":        (0.55, 0.44, 0.08, 0.11),
    "pp5":        (0.28, 0.44, 0.08, 0.11),
    # Middle phalanges
    "mp3":        (0.55, 0.58, 0.07, 0.09),
    "mp5":        (0.28, 0.58, 0.07, 0.09),
    # Distal phalanges
    "dp1":        (0.80, 0.62, 0.08, 0.09),
    "dp3":        (0.55, 0.70, 0.06, 0.08),
    "dp5":        (0.28, 0.70, 0.06, 0.08),
    # Carpals
    "capitate":   (0.53, 0.18, 0.09, 0.07),
    "hamate":     (0.40, 0.18, 0.08, 0.07),
    "triquetral": (0.32, 0.16, 0.08, 0.06),
    "lunate":     (0.44, 0.14, 0.08, 0.06),
    "scaphoid":   (0.58, 0.14, 0.09, 0.07),
    "trapezoid":  (0.63, 0.21, 0.07, 0.06),
    "trapezium":  (0.72, 0.22, 0.08, 0.07),
}

STAGE_COLORS: dict[str, str] = {
    "A": "#9E9E9E", "B": "#2196F3", "C": "#4CAF50", "D": "#8BC34A",
    "E": "#FFC107", "F": "#FF9800", "G": "#F44336", "H": "#9C27B0", "I": "#673AB7",
}


class BoneDetector:
    def __init__(self, weights_path: Optional[Path] = None):
        self.model = None
        if weights_path and Path(weights_path).exists():
            try:
                from ultralytics import YOLO
                self.model = YOLO(str(weights_path))
            except ImportError:
                pass  # ultralytics not installed — fall back to priors

    def detect(self, img_float: np.ndarray) -> dict[str, dict]:
        """
        Detect 20 TW2 bone ROIs in a 512×512 float32 image.
        Returns {bone_name: {box: [x1,y1,x2,y2], conf: float}} in pixel coords.
        """
        h, w = img_float.shape[:2]

        if self.model is not None:
            return self._detect_yolo(img_float)

        return self._detect_priors(w, h)

    def _detect_priors(self, w: int, h: int) -> dict[str, dict]:
        """Anatomical prior fallback — boxes from known relative positions."""
        results = {}
        for bone, (cx, cy, bw, bh) in BONE_PRIORS.items():
            x1 = int((cx - bw / 2) * w)
            y1 = int((cy - bh / 2) * h)
            x2 = int((cx + bw / 2) * w)
            y2 = int((cy + bh / 2) * h)
            results[bone] = {"box": [x1, y1, x2, y2], "conf": 0.0, "source": "prior"}
        return results

    def _detect_yolo(self, img_float: np.ndarray) -> dict[str, dict]:
        """YOLOv8 inference on a float32 [0,1] image."""
        import torch
        img_uint8 = (img_float * 255).astype(np.uint8)
        if img_uint8.ndim == 2:
            import cv2
            img_uint8 = cv2.cvtColor(img_uint8, cv2.COLOR_GRAY2RGB)
        results = self.model(img_uint8, verbose=False)[0]
        h, w = img_float.shape[:2]
        detections = {}
        for box in results.boxes:
            cls_id = int(box.cls.item())
            bone_name = self.model.names[cls_id]
            conf = float(box.conf.item())
            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
            if bone_name not in detections or conf > detections[bone_name]["conf"]:
                detections[bone_name] = {"box": [x1, y1, x2, y2], "conf": conf, "source": "yolo"}
        # Fill missing bones with priors
        for bone in BONE_PRIORS:
            if bone not in detections:
                detections.update({bone: self._detect_priors(w, h)[bone]})
        return detections

    def extract_rois(
        self, img_float: np.ndarray, detections: dict[str, dict], pad: int = 4
    ) -> dict[str, np.ndarray]:
        """Crop and return each bone ROI from the full image."""
        h, w = img_float.shape[:2]
        rois = {}
        for bone, det in detections.items():
            x1, y1, x2, y2 = det["box"]
            x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
            x2, y2 = min(w, x2 + pad), min(h, y2 + pad)
            rois[bone] = img_float[y1:y2, x1:x2]
        return rois
