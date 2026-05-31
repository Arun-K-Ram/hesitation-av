"""
backend/ml/train_cnn.py

CNN Scene Classifier using Transfer Learning (MobileNetV2).
Classifies frames into 3 scenario classes:
  0: pedestrian_curb
  1: merge_hesitation
  2: occluded_intersection

Run:
  python backend/ml/train_cnn.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from pathlib import Path
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter

ML_DIR       = Path(__file__).parent
RECORDS_DIR  = Path(__file__).parent.parent.parent / "experiments" / "recordings"
ML_DIR.mkdir(exist_ok=True)

# Class mapping

CLASSES = {
    "pedestrian_curb":       0,
    "merge_hesitation":      1,
    "occluded_intersection": 2,
}
IDX_TO_CLASS = {v: k for k, v in CLASSES.items()}


# Dataset

class SceneDataset(Dataset):
    def __init__(self, samples, transform=None):
        self.samples   = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


def load_samples():
    """
    Scan recordings folder and collect (image_path, label) pairs.
    Matches folder prefix to scenario class.
    """
    samples = []

    for folder in RECORDS_DIR.iterdir():
        if not folder.is_dir():
            continue

        # Match folder name to class
        label = None
        for class_name, class_idx in CLASSES.items():
            if folder.name.startswith(class_name):
                label = class_idx
                break

        if label is None:
            continue

        frames_dir = folder / "frames"
        if not frames_dir.exists():
            continue

        for img_path in sorted(frames_dir.glob("*.jpg")):
            samples.append((img_path, label))

    return samples


# Augmentation

def get_transforms():
    """
    Training augmentations:
      - Random horizontal flip
      - Random rotation ±15°
      - Color jitter (brightness, contrast)
      - Gaussian blur
      - Normalize (ImageNet stats for pretrained model)
    """
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(
            brightness=0.3,
            contrast=0.3,
            saturation=0.2,
            hue=0.1
        ),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])

    return train_transform, val_transform


# Model 

def build_model(num_classes=3):
    """
    MobileNetV2 pretrained on ImageNet.
    Replace final classifier for our 3 scene classes.
    Freeze early layers, fine-tune last 3 blocks + classifier.
    """
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)

    # Freeze all layers first
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze last 3 feature blocks for fine-tuning
    for layer in list(model.features.children())[-3:]:
        for param in layer.parameters():
            param.requires_grad = True

    # Replace classifier
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2),
        nn.Linear(in_features, 128),
        nn.ReLU(),
        nn.Dropout(p=0.1),
        nn.Linear(128, num_classes),
    )

    return model


# Training 

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct    = 0
    total      = 0

    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss    = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total   += labels.size(0)

    return total_loss / len(loader), 100.0 * correct / total


def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    correct    = 0
    total      = 0
    all_preds  = []
    all_labels = []

    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            loss    = criterion(outputs, labels)

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total   += labels.size(0)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return (total_loss / len(loader),
            100.0 * correct / total,
            all_preds, all_labels)


# Main

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load samples 
    print("\nScanning recordings...")
    samples = load_samples()

    if not samples:
        print("[ERROR] No samples found in experiments/recordings/")
        print("Make sure recording folders exist with frames/ subdirectory")
        return

    # Count per class
    label_counts = Counter(label for _, label in samples)
    print(f"Total samples: {len(samples)}")
    for class_name, class_idx in CLASSES.items():
        print(f"  {class_name}: {label_counts.get(class_idx, 0)} frames")

    # Transforms─
    train_transform, val_transform = get_transforms()

    # Train/val/test split─
    np.random.seed(42)
    indices   = np.random.permutation(len(samples))
    train_end = int(0.70 * len(samples))
    val_end   = int(0.85 * len(samples))

    train_samples = [samples[i] for i in indices[:train_end]]
    val_samples   = [samples[i] for i in indices[train_end:val_end]]
    test_samples  = [samples[i] for i in indices[val_end:]]

    print(f"\nSplit: train={len(train_samples)} "
          f"val={len(val_samples)} test={len(test_samples)}")

    train_ds = SceneDataset(train_samples, train_transform)
    val_ds   = SceneDataset(val_samples,   val_transform)
    test_ds  = SceneDataset(test_samples,  val_transform)

    train_loader = DataLoader(train_ds, batch_size=32,
                               shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=32,
                               shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=32,
                               shuffle=False, num_workers=0)

    # Model─
    print("\nBuilding MobileNetV2...")
    model     = build_model(num_classes=3).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-3
    )
    scheduler = optim.lr_scheduler.StepLR(
        optimizer, step_size=10, gamma=0.5
    )

    # Training loop 
    print("\n" + "="*55)
    print("  TRAINING")
    print("="*55)

    epochs          = 30
    best_val_acc    = 0.0
    best_model_path = ML_DIR / "cnn_best.pth"
    train_accs      = []
    val_accs        = []
    train_losses    = []
    val_losses      = []

    for epoch in range(epochs):
        tr_loss, tr_acc = train_epoch(
            model, train_loader, criterion, optimizer, device
        )
        vl_loss, vl_acc, _, _ = eval_epoch(
            model, val_loader, criterion, device
        )
        scheduler.step()

        train_accs.append(tr_acc)
        val_accs.append(vl_acc)
        train_losses.append(tr_loss)
        val_losses.append(vl_loss)

        if vl_acc > best_val_acc:
            best_val_acc = vl_acc
            torch.save(model.state_dict(), best_model_path)

        if epoch % 5 == 0:
            print(f"  Epoch {epoch:3d}: "
                  f"train_loss={tr_loss:.4f} train_acc={tr_acc:.1f}%  "
                  f"val_loss={vl_loss:.4f} val_acc={vl_acc:.1f}%")

    # Test evaluation
    print(f"\nBest val accuracy: {best_val_acc:.1f}%")
    print("\nLoading best model for test evaluation...")
    model.load_state_dict(torch.load(best_model_path,
                      weights_only=True))

    _, test_acc, test_preds, test_labels = eval_epoch(
        model, test_loader, criterion, device
    )
    print(f"Test accuracy: {test_acc:.1f}%")

    # Per-class accuracy
    print("\nPer-class accuracy:")
    for class_name, class_idx in CLASSES.items():
        mask     = np.array(test_labels) == class_idx
        if mask.sum() > 0:
            cls_acc = 100.0 * np.array(test_preds)[mask].tolist().count(class_idx) / mask.sum()
            print(f"  {class_name}: {cls_acc:.1f}%")

    # Save results
    results_df = pd.DataFrame({
        "epoch":       range(epochs),
        "train_loss":  train_losses,
        "val_loss":    val_losses,
        "train_acc":   train_accs,
        "val_acc":     val_accs,
    })
    results_df.to_csv(ML_DIR / "cnn_training_results.csv", index=False)

    # Plot 
    fig, axes = plt.subplots(1, 2, figsize=(14, 5),
                              facecolor="#0f172a")

    ax1 = axes[0]
    ax1.set_facecolor("#020617")
    ax1.plot(train_losses, color="#3b82f6", label="Train", linewidth=2)
    ax1.plot(val_losses,   color="#22c55e", label="Val",   linewidth=2)
    ax1.set_title("Loss", color="#e2e8f0", fontsize=11)
    ax1.set_xlabel("Epoch", color="#64748b")
    ax1.set_ylabel("Cross Entropy Loss", color="#64748b")
    ax1.tick_params(colors="#475569")
    ax1.legend(facecolor="#0f172a", labelcolor="#94a3b8",
                edgecolor="#1e293b")
    for spine in ax1.spines.values():
        spine.set_edgecolor("#1e293b")

    ax2 = axes[1]
    ax2.set_facecolor("#020617")
    ax2.plot(train_accs, color="#3b82f6", label="Train", linewidth=2)
    ax2.plot(val_accs,   color="#22c55e", label="Val",   linewidth=2)
    ax2.axhline(y=best_val_acc, color="#ef4444",
                linestyle="--", linewidth=1,
                label=f"Best val={best_val_acc:.1f}%")
    ax2.set_title("Accuracy", color="#e2e8f0", fontsize=11)
    ax2.set_xlabel("Epoch", color="#64748b")
    ax2.set_ylabel("Accuracy %", color="#64748b")
    ax2.set_ylim(0, 105)
    ax2.tick_params(colors="#475569")
    ax2.legend(facecolor="#0f172a", labelcolor="#94a3b8",
                edgecolor="#1e293b")
    for spine in ax2.spines.values():
        spine.set_edgecolor("#1e293b")

    plt.tight_layout()
    plt.savefig(ML_DIR / "cnn_training_plot.png", dpi=150,
                bbox_inches="tight", facecolor="#0f172a")

    # Summary 
    print("\n" + "="*55)
    print("  SUMMARY")
    print("="*55)
    print(f"  Total frames:    {len(samples)}")
    print(f"  Architecture:    MobileNetV2 (pretrained ImageNet)")
    print(f"  Augmentations:   flip, rotate, color jitter, blur")
    print(f"  Best val acc:    {best_val_acc:.1f}%")
    print(f"  Test accuracy:   {test_acc:.1f}%")
    print(f"\n  Model saved  → backend/ml/cnn_best.pth")
    print(f"  Results saved → backend/ml/cnn_training_results.csv")
    print(f"  Plot saved   → backend/ml/cnn_training_plot.png")


if __name__ == "__main__":
    main()