from __future__ import annotations
import io
import numpy as np
import cv2
from pathlib import Path
from typing import Optional


def load_image(source: bytes | str | Path) -> tuple[np.ndarray, Optional[float]]:
    """
    Load a hand X-ray from DICOM or PNG/JPG bytes/path.
    Returns (uint8 grayscale array, pixel_spacing_mm_per_px or None).
    """
    pixel_spacing: Optional[float] = None

    if isinstance(source, (str, Path)):
        source = Path(source).read_bytes()

    # Try DICOM first
    try:
        import pydicom
        ds = pydicom.dcmread(io.BytesIO(source))
        arr = ds.pixel_array.astype(np.float32)
        # Photometric: MONOCHROME1 means bone is dark → invert
        if hasattr(ds, "PhotometricInterpretation") and ds.PhotometricInterpretation == "MONOCHROME1":
            arr = arr.max() - arr
        if hasattr(ds, "PixelSpacing"):
            pixel_spacing = float(ds.PixelSpacing[0])  # mm per pixel
        img = _normalize_to_uint8(arr)
    except Exception:
        buf = np.frombuffer(source, dtype=np.uint8)
        img = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError("Could not decode image — unsupported format")

    return img, pixel_spacing


def preprocess(
    source: bytes | str | Path,
    target_size: int = 512,
    scale_factor: Optional[float] = None,
) -> tuple[np.ndarray, float]:
    """
    Full preprocessing pipeline.
    Returns (float32 [0,1] array 512×512, mm_per_px).
    """
    img, dicom_spacing = load_image(source)

    # Resolve pixel spacing (mm/px): DICOM metadata > manual input > assume 0.143 mm/px (typical DR)
    mm_per_px = dicom_spacing or scale_factor or 0.143

    img = _orient_left_hand(img)
    img = _enhance_clahe(img)
    img = cv2.resize(img, (target_size, target_size), interpolation=cv2.INTER_AREA)
    img_float = img.astype(np.float32) / 255.0
    return img_float, mm_per_px


# ── Private helpers ──────────────────────────────────────────────────────────

def _normalize_to_uint8(arr: np.ndarray) -> np.ndarray:
    lo, hi = arr.min(), arr.max()
    if hi == lo:
        return np.zeros_like(arr, dtype=np.uint8)
    return ((arr - lo) / (hi - lo) * 255).astype(np.uint8)


def _enhance_clahe(img: np.ndarray) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(img)


def _orient_left_hand(img: np.ndarray) -> np.ndarray:
    """
    Heuristic: if thumb is on the right side → it's a right hand → flip.
    We detect the thumb by finding where the largest bright region is on each side.
    """
    h, w = img.shape
    _, thresh = cv2.threshold(img, 30, 255, cv2.THRESH_BINARY)
    left_mass = thresh[:, : w // 2].sum()
    right_mass = thresh[:, w // 2 :].sum()
    # Thumb adds mass; in a left hand the thumb is on the left side of the image
    # (patient's palm facing the detector). If right side is heavier, flip.
    if right_mass > left_mass * 1.1:
        img = cv2.flip(img, 1)
    return img
