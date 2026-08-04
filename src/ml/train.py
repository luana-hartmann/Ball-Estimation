"""
Trains the CNN patch classifier on outputs/ml_dataset/patches_test2.npz +
patches_test5.npz. test_6 is NEVER touched here -- it's reserved for the
final holdout comparison against the classical (hand-tuned) detector.

Handles class imbalance with a class-weighted loss (positives are only
~2-4% of the data). Tracks precision/recall/F1 on a held-out validation
split every epoch -- NOT accuracy, since a model that always predicts
"not ball" would already score ~97%+ accuracy while being completely
useless. Same principle used throughout this project: track the metric
that reflects what we actually care about, not the one that looks good
by default.

Usage:
    python src/ml/train.py --train_npz ../outputs/ml_dataset/patches_test2.npz ../outputs/ml_dataset/patches_test5.npz --out ../outputs/ml_dataset/ball_classifier.pt --epochs 15
"""

import argparse
import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset

from model import BallPatchCNN


class PatchDataset(Dataset):
    def __init__(self, patches, labels):
        self.patches = patches
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        patch = self.patches[idx].astype(np.float32) / 255.0
        patch = torch.from_numpy(patch).permute(2, 0, 1)  # HWC -> CHW, what PyTorch conv layers expect
        label = int(self.labels[idx])
        return patch, label


def load_combined(npz_paths):
    all_patches, all_labels = [], []
    for p in npz_paths:
        data = np.load(p)
        all_patches.append(data["patches"])
        all_labels.append(data["labels"])
    return np.concatenate(all_patches), np.concatenate(all_labels)


def stratified_split(labels, val_frac, seed=42):
    """
    Manual stratified train/val split (no sklearn dependency): splits
    positives and negatives separately by the same fraction, so the rare
    positive class isn't randomly under/over-represented in validation
    just by chance -- important given how few positives there are.
    """
    rng = np.random.default_rng(seed)
    idx_pos = np.where(labels == 1)[0]
    idx_neg = np.where(labels == 0)[0]
    rng.shuffle(idx_pos)
    rng.shuffle(idx_neg)

    n_val_pos = max(1, int(len(idx_pos) * val_frac))
    n_val_neg = int(len(idx_neg) * val_frac)

    idx_val = np.concatenate([idx_pos[:n_val_pos], idx_neg[:n_val_neg]])
    idx_train = np.concatenate([idx_pos[n_val_pos:], idx_neg[n_val_neg:]])
    rng.shuffle(idx_val)
    rng.shuffle(idx_train)
    return idx_train, idx_val


def evaluate(model, loader, device):
    model.eval()
    tp = fp = fn = tn = 0
    with torch.no_grad():
        for patches, labels in loader:
            patches, labels = patches.to(device), labels.to(device)
            preds = model(patches).argmax(dim=1)
            tp += ((preds == 1) & (labels == 1)).sum().item()
            fp += ((preds == 1) & (labels == 0)).sum().item()
            fn += ((preds == 0) & (labels == 1)).sum().item()
            tn += ((preds == 0) & (labels == 0)).sum().item()
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return precision, recall, f1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_npz", nargs="+", required=True,
                         help="One or more .npz files from generate_dataset.py, combined for training")
    parser.add_argument("--out", default="../outputs/ml_dataset/ball_classifier.pt")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val_frac", type=float, default=0.15)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    patches, labels = load_combined(args.train_npz)
    print(f"Loaded {len(labels)} total patches ({labels.mean()*100:.1f}% positive)")

    idx_train, idx_val = stratified_split(labels, args.val_frac)
    print(f"Train: {len(idx_train)} ({labels[idx_train].sum()} positive)  "
          f"Val: {len(idx_val)} ({labels[idx_val].sum()} positive)")

    dataset = PatchDataset(patches, labels)
    train_loader = DataLoader(Subset(dataset, idx_train), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(Subset(dataset, idx_val), batch_size=args.batch_size, shuffle=False)

    # class weights: inverse frequency, so the rare positive class isn't
    # drowned out by the abundant negatives during training
    n_pos = labels[idx_train].sum()
    n_neg = len(idx_train) - n_pos
    weight = torch.tensor([1.0, n_neg / max(n_pos, 1)], dtype=torch.float32).to(device)
    print(f"Class weights [not-ball, ball]: {weight.tolist()}")

    model = BallPatchCNN().to(device)
    criterion = nn.CrossEntropyLoss(weight=weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    best_f1 = -1.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for patches_b, labels_b in train_loader:
            patches_b, labels_b = patches_b.to(device), labels_b.to(device)
            optimizer.zero_grad()
            loss = criterion(model(patches_b), labels_b)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(labels_b)

        train_loss = total_loss / len(idx_train)
        precision, recall, f1 = evaluate(model, val_loader, device)
        print(f"Epoch {epoch:2d}  train_loss={train_loss:.4f}  "
              f"val_precision={precision:.3f}  val_recall={recall:.3f}  val_f1={f1:.3f}")

        if f1 > best_f1:
            best_f1 = f1
            torch.save(model.state_dict(), args.out)
            print(f"  -> saved new best model (f1={f1:.3f}) to {args.out}")

    print(f"\nDone. Best validation F1: {best_f1:.3f}. Model saved to {args.out}")


if __name__ == "__main__":
    main()
