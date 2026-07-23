"""
Pseudo-label generation for TW2 stage classification.

Given RSNA dataset (bone age in months + sex per image), assigns the most probable
TW2 maturation stage for each of the 20 bones using the Gaussian reference distributions.

Output: CSV with columns [image_id, bone_name, stage, probability, age_months, sex]
Used as training labels for the EfficientNet stage classifier.

Usage:
    python training/02_pseudo_label_generation.py \
        --rsna_csv data/rsna/train.csv \
        --output data/annotations/pseudo_labels.csv
"""

import argparse
import json
from pathlib import Path
import pandas as pd
from scipy.stats import norm

_REF = Path(__file__).parent.parent / "backend" / "ml" / "reference_data" / "gaussian_params.json"
_GAUSS = json.loads(_REF.read_text())

BONE_NAMES = list(json.loads(
    (Path(__file__).parent.parent / "backend" / "ml" / "reference_data" / "tw2_tables.json").read_text()
)["bones"].keys())

CARPAL_BONES = {"capitate", "hamate", "triquetral", "lunate", "scaphoid", "trapezoid", "trapezium"}


def stages_for_bone(bone: str) -> list[str]:
    return ["A","B","C","D","E","F","G","H"] if bone in CARPAL_BONES else ["A","B","C","D","E","F","G","H","I"]


def best_stage(bone: str, age_months: float, sex: str) -> tuple[str, float]:
    """Return (most probable stage, probability) for a bone given patient age."""
    sex_key = "male" if sex == "M" else "female"
    params = _GAUSS.get(sex_key, {}).get(bone, {})
    valid = stages_for_bone(bone)

    if not params:
        return "A", 1.0 / len(valid)

    densities = {}
    for stage in valid:
        if stage not in params:
            densities[stage] = 1e-8
        else:
            mu, sd = params[stage]["mean"], max(params[stage]["sd"], 1.0)
            densities[stage] = float(norm.pdf(age_months, loc=mu, scale=sd))

    total = sum(densities.values()) or 1.0
    probs = {s: densities[s] / total for s in valid}
    best = max(probs, key=probs.get)
    return best, probs[best]


def generate(rsna_csv: Path, output_csv: Path, min_prob: float = 0.0):
    df = pd.read_csv(rsna_csv)
    # Expected columns: id (or image_id), boneage (months), male (0/1)
    if "boneage" not in df.columns:
        raise ValueError("CSV must have 'boneage' column (bone age in months)")

    # Normalize column names
    if "id" in df.columns:
        df = df.rename(columns={"id": "image_id"})
    if "male" in df.columns:
        df["sex"] = df["male"].map({1: "M", 0: "F"})
    elif "Sex" in df.columns:
        df["sex"] = df["Sex"].str.upper().str[0]

    records = []
    for _, row in df.iterrows():
        age = float(row["boneage"])
        sex = str(row["sex"])
        img_id = str(row["image_id"])
        for bone in BONE_NAMES:
            stage, prob = best_stage(bone, age, sex)
            if prob >= min_prob:
                records.append({
                    "image_id": img_id,
                    "bone": bone,
                    "stage": stage,
                    "probability": round(prob, 4),
                    "age_months": age,
                    "sex": sex,
                })

    out_df = pd.DataFrame(records)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_csv, index=False)
    print(f"Saved {len(out_df):,} pseudo-labels → {output_csv}")
    print(out_df.groupby("bone")["stage"].value_counts().unstack(fill_value=0).to_string())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rsna_csv", type=Path, default=Path("data/rsna/train.csv"))
    parser.add_argument("--output", type=Path, default=Path("data/annotations/pseudo_labels.csv"))
    parser.add_argument("--min_prob", type=float, default=0.0,
                        help="Only keep labels where P(stage|age) >= this threshold")
    args = parser.parse_args()
    generate(args.rsna_csv, args.output, args.min_prob)
