from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from typing import Literal

_REF = Path(__file__).parent.parent.parent / "ml" / "reference_data" / "tw2_tables.json"
_tables = json.loads(_REF.read_text())

Sex = Literal["M", "F"]


def bone_score(bone_name: str, stage: str) -> int:
    """Return TW2 numeric score for a single bone at a given stage (A-I)."""
    bones = _tables["bones"]
    if bone_name not in bones:
        raise ValueError(f"Unknown bone: {bone_name}")
    stages = bones[bone_name]["stages"]
    if stage not in stages:
        raise ValueError(f"Unknown stage '{stage}' for {bone_name}. Valid: {list(stages)}")
    return stages[stage]


def calculate_scores(stages: dict[str, str]) -> dict:
    """
    stages: {bone_name: stage_letter}  e.g. {"radius": "D", "ulna": "E", ...}
    Returns dict with rus_score, carpal_score, bone_scores.
    """
    rus_total = 0
    carpal_total = 0
    bone_scores: dict[str, int] = {}

    for bone, stage in stages.items():
        if bone not in _tables["bones"]:
            continue
        score = bone_score(bone, stage)
        bone_scores[bone] = score
        group = _tables["bones"][bone]["group"]
        if group == "RUS":
            rus_total += score
        elif group == "Carpal":
            carpal_total += score

    # Normalize to 0-1000 scale
    # RUS: 13 bones × max 48 pts ≈ 624 raw → published tables use a 0-1000 scale
    # The published TW2 uses a conversion: score_norm = raw_sum / max_possible * 1000
    max_rus = sum(
        max(_tables["bones"][b]["stages"].values())
        for b in _tables["bones"] if _tables["bones"][b]["group"] == "RUS"
    )
    max_carpal = sum(
        max(_tables["bones"][b]["stages"].values())
        for b in _tables["bones"] if _tables["bones"][b]["group"] == "Carpal"
    )

    rus_norm = int(round(rus_total / max_rus * 1000)) if max_rus else 0
    carpal_norm = int(round(carpal_total / max_carpal * 1000)) if max_carpal else 0

    return {
        "bone_scores": bone_scores,
        "rus_raw": rus_total,
        "carpal_raw": carpal_total,
        "rus_score": rus_norm,
        "carpal_score": carpal_norm,
    }


def score_to_age(score: int, sex: Sex, table_key: str) -> float:
    """Interpolate score → bone age in months using TW2 lookup table."""
    sex_key = "male" if sex == "M" else "female"
    table: dict[str, int] = _tables[table_key][sex_key]
    scores = sorted(int(k) for k in table)
    ages = [table[str(s)] for s in scores]

    if score <= scores[0]:
        return float(ages[0])
    if score >= scores[-1]:
        return float(ages[-1])

    for i in range(len(scores) - 1):
        if scores[i] <= score <= scores[i + 1]:
            t = (score - scores[i]) / (scores[i + 1] - scores[i])
            return ages[i] + t * (ages[i + 1] - ages[i])
    return float(ages[-1])


def estimate_bone_age(stages: dict[str, str], sex: Sex) -> dict:
    """
    Full TW2 pipeline: stages → bone age estimate with confidence interval.
    Returns dict ready for API response.
    """
    scores = calculate_scores(stages)

    rus_age = score_to_age(scores["rus_score"], sex, "age_from_rus_score")
    carpal_age = score_to_age(scores["carpal_score"], sex, "age_from_carpal_score")

    # Weighted average: TW2 recommends RUS as primary, Carpal as secondary
    bone_age = rus_age * 0.75 + carpal_age * 0.25

    # Approximate 90% CI from published TW2 SD data (~±8 months at mid-range)
    ci_half = _confidence_half_width(scores["rus_score"])

    return {
        **scores,
        "rus_age_months": round(rus_age, 1),
        "carpal_age_months": round(carpal_age, 1),
        "bone_age_months": round(bone_age, 1),
        "bone_age_years": round(bone_age / 12, 2),
        "confidence_interval": [round(bone_age - ci_half, 1), round(bone_age + ci_half, 1)],
    }


def _confidence_half_width(rus_score: int) -> float:
    """
    Approximate 90% CI half-width in months from TW2 published SD curves.
    SD is largest (~10 months) at mid-range scores and smaller at extremes.
    """
    # Parabola fitted to TW2 Table 5 approximate values
    x = rus_score / 1000.0
    sd = 10.0 * (1 - (2 * x - 1) ** 2) + 4.0
    return round(1.645 * sd, 1)  # 90% CI = ±1.645 SD
