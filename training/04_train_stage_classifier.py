"""
Train EfficientNet-B3 stage classifier using pseudo-labels.

Usage (Kaggle/Colab with GPU):
    python training/04_train_stage_classifier.py \
        --rsna_dir data/rsna/train/ \
        --labels data/annotations/pseudo_labels.csv \
        --output backend/ml/weights/stage_classifier.pt \
        --epochs 20 --batch 32
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import timm
from tqdm import tqdm
from sklearn.model_selection import train_test_split

BONE_NAMES = list(json.loads(
    (Path(__file__).parent.parent / "backend" / "ml" / "reference_data" / "tw2_tables.json").read_text()
)["bones"].keys())
CARPAL_BONES = {"capitate","hamate","triquetral","lunate","scaphoid","trapezoid","trapezium"}
STAGE_MAP = {s: i for i, s in enumerate("ABCDEFGHI")}


def stages_for_bone(bone):
    return "ABCDEFGH" if bone in CARPAL_BONES else "ABCDEFGHI"


class BoneROIDataset(Dataset):
    def __init__(self, df: pd.DataFrame, img_dir: Path, transform, bone_filter: str):
        self.df = df[df["bone"] == bone_filter].reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform
        self.bone = bone_filter
        self.valid_stages = stages_for_bone(bone_filter)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = self.img_dir / f"{row['image_id']}.png"
        if not img_path.exists():
            img_path = self.img_dir / f"{row['image_id']}.jpg"
        img = Image.open(img_path).convert("L")
        img_t = self.transform(img)
        stage_idx = self.valid_stages.index(row["stage"])
        return img_t, stage_idx


def build_model(bone: str, feat_dim: int) -> nn.Module:
    n_classes = len(stages_for_bone(bone))
    return nn.Sequential(nn.Linear(feat_dim, 256), nn.ReLU(), nn.Dropout(0.3), nn.Linear(256, n_classes))


def train(rsna_dir: Path, labels_csv: Path, output: Path, epochs: int = 20, batch: int = 32):
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")

    df = pd.read_csv(labels_csv)
    train_ids, val_ids = train_test_split(df["image_id"].unique(), test_size=0.1, random_state=42)
    df_train = df[df["image_id"].isin(train_ids)]
    df_val = df[df["image_id"].isin(val_ids)]

    tfm_train = transforms.Compose([
        transforms.Resize((96, 96)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3),
    ])
    tfm_val = transforms.Compose([
        transforms.Resize((96, 96)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3),
    ])

    # Shared backbone
    backbone = timm.create_model("efficientnet_b3", pretrained=True, num_classes=0).to(device)
    feat_dim = backbone.num_features

    # Per-bone heads
    heads = {bone: build_model(bone, feat_dim).to(device) for bone in BONE_NAMES}

    optimizer = torch.optim.AdamW(
        list(backbone.parameters()) + [p for h in heads.values() for p in h.parameters()],
        lr=1e-4, weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    for epoch in range(epochs):
        backbone.train()
        for h in heads.values():
            h.train()

        total_loss = 0
        n_batches = 0
        for bone in BONE_NAMES:
            ds_train = BoneROIDataset(df_train, rsna_dir, tfm_train, bone)
            if len(ds_train) == 0:
                continue
            loader = DataLoader(ds_train, batch_size=batch, shuffle=True, num_workers=2, pin_memory=True)
            for imgs, labels in loader:
                imgs, labels = imgs.to(device), labels.to(device)
                feats = backbone(imgs)
                logits = heads[bone](feats)
                loss = criterion(logits, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                n_batches += 1

        scheduler.step()
        avg_loss = total_loss / max(n_batches, 1)
        print(f"Epoch {epoch+1}/{epochs} — loss: {avg_loss:.4f}")

    # Save combined model state
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "backbone": backbone.state_dict(),
        "heads": {bone: head.state_dict() for bone, head in heads.items()},
        "feat_dim": feat_dim,
    }, output)
    print(f"Saved → {output}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--rsna_dir", type=Path, default=Path("data/rsna/train"))
    p.add_argument("--labels", type=Path, default=Path("data/annotations/pseudo_labels.csv"))
    p.add_argument("--output", type=Path, default=Path("backend/ml/weights/stage_classifier.pt"))
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch", type=int, default=32)
    args = p.parse_args()
    train(args.rsna_dir, args.labels, args.output, args.epochs, args.batch)
