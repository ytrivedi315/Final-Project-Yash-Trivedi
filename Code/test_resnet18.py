import argparse
import csv
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from torchvision.models import ResNet18_Weights


class CHFFrameDataset(Dataset):
    # Class for dataset
    def __init__(self, annotations_path: str, image_dir: str, transform=None):
        # Store paths and transform
        self.annotations_path = Path(annotations_path)
        self.image_dir = Path(image_dir)
        self.transform = transform

        # Read file with labels
        if self.annotations_path.suffix.lower() == ".csv":
            self.df = pd.read_csv(self.annotations_path)
        else:
            self.df = pd.read_excel(self.annotations_path)

        # Extract CHF Prox labels
        required_target_col = "CHF Proximity"
        if required_target_col not in self.df.columns:
            raise ValueError(
                f"Expected column '{required_target_col}' in annotations file. "
                f"Available columns: {list(self.df.columns)}"
            )

        self.df["frame_filename"] = [f"frame_{i:04d}.png" for i in range(len(self.df))]

        self.df[required_target_col] = self.df[required_target_col].clip(0.0, 1.0)
        exists_mask = self.df["frame_filename"].apply(lambda f: (self.image_dir / str(f)).exists())
        self.df = self.df[exists_mask].reset_index(drop=True)

        if len(self.df) == 0:
            raise RuntimeError("No valid test samples remain after checking image files.")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = self.image_dir / str(row["frame_filename"])
        image = Image.open(img_path).convert("RGB")
        target = float(row["CHF Proximity"])
        target = max(0.0, min(1.0, target))

        if self.transform is not None:
            image = self.transform(image)

        return image, torch.tensor([target], dtype=torch.float32), row["frame_filename"]


class ResNet18Regressor(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = models.resnet18(weights=None) # Does not load pretrained weights in testing
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.backbone(x)


@torch.no_grad()
def evaluate(model, loader, criterion, device, output_csv: Path):
    model.eval()
    total_loss = 0.0
    preds_all = []
    targets_all = []
    rows = []

    for images, targets, filenames in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        preds = model(images)
        loss = criterion(preds, targets)
        total_loss += loss.item() * images.size(0)

        # Save predictions back to CPU
        preds_np = preds.cpu().numpy().reshape(-1)
        targets_np = targets.cpu().numpy().reshape(-1)

        preds_all.append(preds_np)
        targets_all.append(targets_np)

        for fn, pred, tgt in zip(filenames, preds_np, targets_np):
            rows.append({
                "frame_filename": fn,
                "target_chf_proximity": float(tgt),
                "predicted_chf_proximity": float(pred),
                "abs_error": float(abs(pred - tgt)),
            })

    preds_all = np.concatenate(preds_all)
    targets_all = np.concatenate(targets_all)

    mse = float(np.mean((preds_all - targets_all) ** 2))
    mae = float(np.mean(np.abs(preds_all - targets_all)))
    rmse = float(np.sqrt(mse))
    if np.std(targets_all) > 1e-12:
        r2 = float(1.0 - np.sum((targets_all - preds_all) ** 2) / np.sum((targets_all - np.mean(targets_all)) ** 2))
        corr = float(np.corrcoef(targets_all, preds_all)[0, 1])
    else:
        r2 = float("nan")
        corr = float("nan")

    avg_loss = total_loss / len(loader.dataset)

    # Write the output results to a .csv file
    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    return avg_loss, mae, rmse, r2, corr


def main():
    parser = argparse.ArgumentParser(description="Test ResNet-18 on case 91 for CHF proximity regression.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to trained .pt file")
    parser.add_argument("--annotations", type=str, required=True, help="Case 91 aligned CSV/XLSX path")
    parser.add_argument("--image_dir", type=str, required=True, help="Directory of extracted case 91 frames")
    parser.add_argument("--output_csv", type=str, default="case91_predictions.csv")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    dataset = CHFFrameDataset(args.annotations, args.image_dir, transform=transform)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False, # No shuffling
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    model = ResNet18Regressor().to(device)
    # Load training checkpoint for ResNet-18
    state_dict = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state_dict)

    criterion = nn.HuberLoss(delta=0.1)
    avg_loss, mae, rmse, r2, corr = evaluate(model, loader, criterion, device, Path(args.output_csv))

    print("\nCase 91 test results")
    print(f"loss : {avg_loss:.6f}")
    print(f"MAE  : {mae:.6f}")
    print(f"RMSE : {rmse:.6f}")
    print(f"R^2  : {r2:.6f}")
    print(f"Corr : {corr:.6f}")
    print(f"Predictions saved to: {args.output_csv}")


if __name__ == "__main__":
    main()
