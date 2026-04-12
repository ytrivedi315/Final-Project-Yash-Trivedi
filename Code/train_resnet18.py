import argparse
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, models
from torchvision.models import ResNet18_Weights

from torch.utils.data import Subset

# ----------------------------
# Utilities
# ----------------------------

def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class CHFFrameDataset(Dataset):
    """
    Dataset for single-frame CHF proximity regression.

    Expected CSV/XLSX columns:
      - frame_filename: image filename for each aligned frame
      - CHF proximity: regression target in [0, 1]

    The aligned-frame CSV produced earlier already contains frame filenames;
    if frame_filename is missing but frame_index exists, this class will build
    names like frame_000123.jpg.
    """

    def __init__(self, annotations_path: str, image_dir: str, transform=None):
        self.annotations_path = Path(annotations_path)
        self.image_dir = Path(image_dir)
        self.transform = transform

        if not self.annotations_path.exists():
            raise FileNotFoundError(f"Annotations file not found: {self.annotations_path}")
        if not self.image_dir.exists():
            raise FileNotFoundError(f"Image directory not found: {self.image_dir}")

        if self.annotations_path.suffix.lower() == ".csv":
            self.df = pd.read_csv(self.annotations_path)
        else:
            self.df = pd.read_excel(self.annotations_path)

        required_target_col = "CHF Proximity"
        if required_target_col not in self.df.columns:
            raise ValueError(
                f"Expected target column '{required_target_col}' not found. "
                f"Available columns: {list(self.df.columns)}"
            )

        # Always use sequential filenames (matches extracted frames)
        self.df["frame_filename"] = [f"frame_{i:04d}.png" for i in range(len(self.df))]
        self.mode = "filename"

        self.df = self.df.copy()
        self.df[required_target_col] = self.df[required_target_col].clip(0.0, 1.0)

        # Keep only rows with existing images.
        def build_image_name(row):
            if self.mode == "filename":
                return str(row["frame_filename"])
            elif self.mode == "frame_index":
                return f"frame_{int(row['frame_index'])}.png"
            elif self.mode == "frame_index_round":
                return f"frame_{int(row['frame_index_round'])}.png"
            else:
                raise RuntimeError("Unknown dataset mode.")

        exists_mask = [
            (self.image_dir / f"frame_{i:04d}.png").exists()
            for i in range(len(self.df))
        ]
        exists_mask = np.array(exists_mask)

        missing_count = int((~exists_mask).sum())
        if missing_count > 0:
            print(f"Warning: dropping {missing_count} rows with missing image files.")
        self.df = self.df[exists_mask].reset_index(drop=True)

        if len(self.df) == 0:
            raise RuntimeError("No valid samples remain after checking image files.")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        image_name = str(row["frame_filename"])

        img_path = self.image_dir / image_name
        image = Image.open(img_path).convert("RGB")
        target = float(row["CHF Proximity"])
        target = max(0.0, min(1.0, target))

        if self.transform is not None:
            image = self.transform(image)

        return image, torch.tensor([target], dtype=torch.float32)


class ResNet18Regressor(nn.Module):
    def __init__(self, pretrained: bool = True):
        super().__init__()
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        self.backbone = models.resnet18(weights=weights)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.backbone(x)


@torch.no_grad()
def evaluate(model, loader, criterion, device) -> Tuple[float, float, float, float]:
    model.eval()
    total_loss = 0.0
    preds_all = []
    targets_all = []

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        preds = model(images)
        loss = criterion(preds, targets)

        total_loss += loss.item() * images.size(0)
        preds_all.append(preds.cpu().numpy())
        targets_all.append(targets.cpu().numpy())

    preds_all = np.concatenate(preds_all).reshape(-1)
    targets_all = np.concatenate(targets_all).reshape(-1)

    mse = float(np.mean((preds_all - targets_all) ** 2))
    mae = float(np.mean(np.abs(preds_all - targets_all)))
    rmse = float(np.sqrt(mse))
    if np.std(targets_all) > 1e-12:
        r2 = float(1.0 - np.sum((targets_all - preds_all) ** 2) / np.sum((targets_all - np.mean(targets_all)) ** 2))
    else:
        r2 = float("nan")

    avg_loss = total_loss / len(loader.dataset)
    return avg_loss, mae, rmse, r2


def train_one_epoch(model, loader, criterion, optimizer, device) -> float:
    model.train()
    total_loss = 0.0

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        preds = model(images)
        loss = criterion(preds, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)

    return total_loss / len(loader.dataset)


def main():
    parser = argparse.ArgumentParser(description="Train ResNet-18 on case 84 for CHF proximity regression.")
    parser.add_argument("--annotations", type=str, required=True, help="Case 84 aligned CSV/XLSX path")
    parser.add_argument("--image_dir", type=str, required=True, help="Directory of extracted case 84 frames")
    parser.add_argument("--output_dir", type=str, default="outputs_case84_resnet18")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--val_split", type=float, default=0.2)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--freeze_backbone_epochs", type=int, default=3)
    args = parser.parse_args()

    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    base_dataset = CHFFrameDataset(args.annotations, args.image_dir, transform=None)

    n_total = len(base_dataset)
    n_val = max(1, int(n_total * args.val_split))
    n_train = n_total - n_val
    if n_train < 1:
        raise ValueError("Validation split too large; no training samples remain.")

    indices = torch.randperm(n_total, generator=torch.Generator().manual_seed(args.seed)).tolist()
    train_indices = indices[:n_train]
    val_indices = indices[n_train:]

    train_dataset = CHFFrameDataset(args.annotations, args.image_dir, transform=train_transform)
    val_dataset = CHFFrameDataset(args.annotations, args.image_dir, transform=val_transform)

    train_ds = Subset(train_dataset, train_indices)
    val_ds = Subset(val_dataset, val_indices)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    model = ResNet18Regressor(pretrained=True).to(device)
    criterion = nn.HuberLoss(delta=0.1)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val_mae = float("inf")
    history = []

    for epoch in range(1, args.epochs + 1):
        # Optional warm-up: freeze backbone for a few epochs.
        if epoch <= args.freeze_backbone_epochs:
            for name, param in model.backbone.named_parameters():
                param.requires_grad = name.startswith("fc")
        else:
            for param in model.parameters():
                param.requires_grad = True

        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_mae, val_rmse, val_r2 = evaluate(model, val_loader, criterion, device)

        epoch_record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_mae": val_mae,
            "val_rmse": val_rmse,
            "val_r2": val_r2,
        }
        history.append(epoch_record)

        print(
            f"Epoch {epoch:03d} | "
            f"train_loss={train_loss:.5f} | "
            f"val_loss={val_loss:.5f} | "
            f"val_mae={val_mae:.5f} | "
            f"val_rmse={val_rmse:.5f} | "
            f"val_r2={val_r2:.5f}"
        )

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_mae": best_val_mae,
                "args": vars(args),
            }
            torch.save(checkpoint, output_dir / "best_resnet18_case84.pth")
            print(f"Saved new best model at epoch {epoch} with val_mae={val_mae:.5f}")

    with open(output_dir / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"Training complete. Best val MAE: {best_val_mae:.5f}")
    print(f"Best checkpoint: {output_dir / 'best_resnet18_case84.pth'}")


if __name__ == "__main__":
    main()
