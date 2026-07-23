from __future__ import annotations
import numpy as np
import json
from pathlib import Path
from typing import Literal, Optional

STAGES = ["A", "B", "C", "D", "E", "F", "G", "H", "I"]
Sex = Literal["M", "F"]

_REF_DIR = Path(__file__).parent.parent.parent / "ml"
_WEIGHTS = _REF_DIR / "weights" / "stage_classifier.pt"
_GAUSS = json.loads((_REF_DIR / "reference_data" / "gaussian_params.json").read_text())


class StageClassifier:
    """
    Classifies each bone ROI into a TW2 maturation stage (A→H/I).

    Two modes:
    1. Model mode: EfficientNet-B3 backbone + 20 heads (loaded from weights/).
    2. Fallback mode: returns the most probable stage given patient age (prior-only).
    """

    def __init__(self, weights_path: Optional[Path] = None):
        self.model = None
        self.transform = None
        path = weights_path or _WEIGHTS
        if Path(path).exists():
            self._load_model(path)

    def _load_model(self, path: Path):
        try:
            import torch
            import timm
            from torchvision import transforms

            self.model = torch.load(path, map_location="cpu", weights_only=False)
            self.model.eval()
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((96, 96)),
                transforms.Grayscale(num_output_channels=3),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ])
        except Exception:
            self.model = None

    def classify_all(
        self,
        rois: dict[str, np.ndarray],
        sex: Sex,
        age_months: Optional[float] = None,
    ) -> dict[str, dict]:
        """
        Classify each bone ROI.
        Returns {bone: {stage: str, probabilities: {A:.., B:.., ...}, source: str}}
        """
        if self.model is not None:
            return self._classify_model(rois, sex, age_months)
        return self._classify_prior(list(rois.keys()), sex, age_months)

    def _classify_model(
        self, rois: dict[str, np.ndarray], sex: Sex, age_months: Optional[float]
    ) -> dict[str, dict]:
        import torch
        results = {}
        with torch.no_grad():
            for bone, roi in rois.items():
                if roi.size == 0:
                    results[bone] = self._prior_for_bone(bone, sex, age_months)
                    continue
                roi_uint8 = (roi * 255).clip(0, 255).astype(np.uint8)
                tensor = self.transform(roi_uint8).unsqueeze(0)
                logits = self.model(tensor, bone_name=bone)
                probs = torch.softmax(logits, dim=-1).squeeze().numpy()
                stage_idx = int(probs.argmax())
                valid_stages = _valid_stages_for_bone(bone)
                stage = valid_stages[stage_idx] if stage_idx < len(valid_stages) else valid_stages[-1]
                results[bone] = {
                    "stage": stage,
                    "probabilities": {s: float(probs[i]) for i, s in enumerate(valid_stages)},
                    "source": "model",
                }
        return results

    def _classify_prior(
        self, bones: list[str], sex: Sex, age_months: Optional[float]
    ) -> dict[str, dict]:
        """Age-based prior when no model is available."""
        return {bone: self._prior_for_bone(bone, sex, age_months) for bone in bones}

    def _prior_for_bone(self, bone: str, sex: Sex, age_months: Optional[float]) -> dict:
        sex_key = "male" if sex == "M" else "female"
        params = _GAUSS.get(sex_key, {}).get(bone, {})
        valid_stages = _valid_stages_for_bone(bone)

        if not params or age_months is None:
            # Unknown age: return stage A as default (no ossification)
            probs = {s: 1.0 / len(valid_stages) for s in valid_stages}
            return {"stage": valid_stages[0], "probabilities": probs, "source": "uniform_prior"}

        # Compute P(stage | age) from Gaussian densities
        from scipy.stats import norm
        densities = {}
        for s in valid_stages:
            if s not in params:
                densities[s] = 1e-6
                continue
            mu = params[s]["mean"]
            sd = max(params[s]["sd"], 1.0)
            densities[s] = float(norm.pdf(age_months, loc=mu, scale=sd))

        total = sum(densities.values()) or 1.0
        probs = {s: densities[s] / total for s in valid_stages}
        stage = max(probs, key=probs.get)

        return {"stage": stage, "probabilities": probs, "source": "age_prior"}


def _valid_stages_for_bone(bone: str) -> list[str]:
    """Carpals go up to H only; long/short bones go up to I."""
    carpal_bones = {"capitate","hamate","triquetral","lunate","scaphoid","trapezoid","trapezium"}
    return STAGES[:8] if bone in carpal_bones else STAGES


def build_model(num_bones: int = 20) -> "torch.nn.Module":
    """
    Build EfficientNet-B3 with per-bone classification heads.
    Used in training — not at inference time.
    """
    import torch
    import torch.nn as nn
    import timm

    backbone = timm.create_model("efficientnet_b3", pretrained=True, num_classes=0)
    feat_dim = backbone.num_features

    class BoneAgeModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = backbone
            bone_names = list(__import__(
                "bone_detector", fromlist=["BONE_PRIORS"]
            ).BONE_PRIORS.keys())
            self.heads = nn.ModuleDict({
                bone: nn.Linear(feat_dim, len(_valid_stages_for_bone(bone)))
                for bone in bone_names
            })

        def forward(self, x, bone_name: str):
            features = self.backbone(x)
            return self.heads[bone_name](features)

    return BoneAgeModel()
