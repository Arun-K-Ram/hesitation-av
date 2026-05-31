"""
backend/ml/train_cnn_v2.py

Multi-architecture CNN comparison with extended augmentations
and hyperparameter sweep.

Architectures: MobileNetV2, MobileNetV3, EfficientNetB0,
               EfficientNetB2, ResNet18, ConvNeXt-Tiny

Run:
  python backend/ml/train_cnn_v2.py
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
from PIL import Image, ImageFilter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from collections import Counter
import time

ML_DIR      = Path(__file__).parent
RECORDS_DIR = Path(__file__).parent.parent.parent / "experiments" / "recordings"
ML_DIR.mkdir(exist_ok=True)

CLASSES = {
    "pedestrian_curb":       0,
    "merge_hesitation":      1,
    "occluded_intersection": 2,
}
IDX_TO_CLASS = {v: k for k, v in CLASSES.items()}


#  Custom transforms

class SobelFilter:
    """Apply Sobel edge detection as augmentation."""
    def __call__(self, img):
        if np.random.random() < 0.3:
            img = img.filter(ImageFilter.FIND_EDGES)
        return img


class Sharpen:
    """Apply sharpening filter."""
    def __call__(self, img):
        if np.random.random() < 0.3:
            img = img.filter(ImageFilter.SHARPEN)
        return img


#  Dataset

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
    samples = []
    for folder in RECORDS_DIR.iterdir():
        if not folder.is_dir():
            continue
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


#  Transforms

def get_transforms(img_size=224):
    """Extended augmentation pipeline."""
    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),

        # Geometric
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.RandomPerspective(distortion_scale=0.2, p=0.3),

        # Photometric
        transforms.ColorJitter(
            brightness=0.4,
            contrast=0.4,
            saturation=0.3,
            hue=0.1
        ),
        transforms.RandomGrayscale(p=0.1),

        # Filters
        SobelFilter(),
        Sharpen(),

        # Occlusion simulation
        # Blur
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),

        transforms.ToTensor(),

        # Occlusion simulation (must be after ToTensor)
        transforms.RandomErasing(
            p=0.3,
            scale=(0.02, 0.15),
            ratio=(0.3, 3.0)
        ),

        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])

    return train_transform, val_transform


#  Model builder

def build_model(arch: str, num_classes=3, dropout=0.2):
    """
    Build pretrained model with custom classifier head.
    Supports: mobilenet_v2, mobilenet_v3, efficientnet_b0,
              efficientnet_b2, resnet18, convnext_tiny
    """
    if arch == "mobilenet_v2":
        model = models.mobilenet_v2(
            weights=models.MobileNet_V2_Weights.DEFAULT)
        for param in model.parameters():
            param.requires_grad = False
        for layer in list(model.features.children())[-3:]:
            for param in layer.parameters():
                param.requires_grad = True
        in_f = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_f, 128),
            nn.ReLU(),
            nn.Dropout(dropout/2),
            nn.Linear(128, num_classes),
        )

    elif arch == "mobilenet_v3":
        model = models.mobilenet_v3_small(
            weights=models.MobileNet_V3_Small_Weights.DEFAULT)
        for param in model.parameters():
            param.requires_grad = False
        for layer in list(model.features.children())[-3:]:
            for param in layer.parameters():
                param.requires_grad = True
        in_f = model.classifier[3].in_features
        model.classifier[3] = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_f, num_classes),
        )

    elif arch == "efficientnet_b0":
        model = models.efficientnet_b0(
            weights=models.EfficientNet_B0_Weights.DEFAULT)
        for param in model.parameters():
            param.requires_grad = False
        for layer in list(model.features.children())[-3:]:
            for param in layer.parameters():
                param.requires_grad = True
        in_f = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_f, 128),
            nn.ReLU(),
            nn.Dropout(dropout/2),
            nn.Linear(128, num_classes),
        )

    elif arch == "efficientnet_b2":
        model = models.efficientnet_b2(
            weights=models.EfficientNet_B2_Weights.DEFAULT)
        for param in model.parameters():
            param.requires_grad = False
        for layer in list(model.features.children())[-3:]:
            for param in layer.parameters():
                param.requires_grad = True
        in_f = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_f, 128),
            nn.ReLU(),
            nn.Dropout(dropout/2),
            nn.Linear(128, num_classes),
        )

    elif arch == "resnet18":
        model = models.resnet18(
            weights=models.ResNet18_Weights.DEFAULT)
        for param in model.parameters():
            param.requires_grad = False
        for param in model.layer4.parameters():
            param.requires_grad = True
        in_f = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_f, 128),
            nn.ReLU(),
            nn.Dropout(dropout/2),
            nn.Linear(128, num_classes),
        )

    elif arch == "convnext_tiny":
        model = models.convnext_tiny(
            weights=models.ConvNeXt_Tiny_Weights.DEFAULT)
        for param in model.parameters():
            param.requires_grad = False
        # Unfreeze last stage
        for param in model.features[-1].parameters():
            param.requires_grad = True
        in_f = model.classifier[2].in_features
        model.classifier[2] = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_f, 128),
            nn.ReLU(),
            nn.Dropout(dropout/2),
            nn.Linear(128, num_classes),
        )

    return model


#  Train/eval functions 

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


def train_model(arch, dropout, lr, X_train, X_val,
                device, epochs=30):
    """Train one model configuration."""
    train_transform, val_transform = get_transforms()

    train_ds = SceneDataset(X_train, train_transform)
    val_ds   = SceneDataset(X_val,   val_transform)

    train_loader = DataLoader(train_ds, batch_size=128,
                               shuffle=True,  num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=128,
                               shuffle=False, num_workers=4, pin_memory=True)

    model     = build_model(arch, dropout=dropout).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs
    )

    best_val_acc  = 0.0
    best_val_loss = float("inf")
    patience      = 7
    patience_ctr  = 0
    train_accs    = []
    val_accs      = []

    for epoch in range(epochs):
        tr_loss, tr_acc = train_epoch(
            model, train_loader, criterion, optimizer, device)
        vl_loss, vl_acc, _, _ = eval_epoch(
            model, val_loader, criterion, device)
        scheduler.step()

        train_accs.append(tr_acc)
        val_accs.append(vl_acc)

        if vl_acc > best_val_acc:
            best_val_acc  = vl_acc
            best_val_loss = vl_loss
            best_state    = {k: v.clone()
                             for k, v in model.state_dict().items()}
            patience_ctr  = 0
        else:
            patience_ctr += 1
            if patience_ctr >= patience:
                break

    model.load_state_dict(best_state)
    return model, best_val_acc, train_accs, val_accs


#  Main

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    # Load samples
    samples = load_samples()
    if not samples:
        print("[ERROR] No samples found")
        return

    label_counts = Counter(label for _, label in samples)
    print(f"Total samples: {len(samples)}")
    for name, idx in CLASSES.items():
        print(f"  {name}: {label_counts.get(idx, 0)}")

    # Split
    np.random.seed(42)
    indices   = np.random.permutation(len(samples))
    train_end = int(0.70 * len(samples))
    val_end   = int(0.85 * len(samples))

    train_samples = [samples[i] for i in indices[:train_end]]
    val_samples   = [samples[i] for i in indices[train_end:val_end]]
    test_samples  = [samples[i] for i in indices[val_end:]]

    print(f"\nSplit: train={len(train_samples)} "
          f"val={len(val_samples)} test={len(test_samples)}\n")

    #  Hyperparameter sweep 
    architectures = [
        "mobilenet_v2",
        "mobilenet_v3",
        "efficientnet_b0",
        "efficientnet_b2",
        "resnet18",
        "convnext_tiny",
    ]

    param_grid = {
        "lr":      [1e-3, 5e-4],
        "dropout": [0.1, 0.2, 0.3],
    }

    sweep_results = []
    best_overall_acc  = 0.0
    best_overall_model = None
    best_overall_cfg   = None

    total = len(architectures) * len(param_grid["lr"]) * \
            len(param_grid["dropout"])
    done  = 0

    print("=" * 60)
    print(f"  ARCHITECTURE + HYPERPARAMETER SWEEP")
    print(f"  {total} combinations")
    print("=" * 60)

    for arch in architectures:
        for lr in param_grid["lr"]:
            for dropout in param_grid["dropout"]:
                done += 1
                t0 = time.time()

                model, val_acc, tr_accs, vl_accs = train_model(
                    arch, dropout, lr,
                    train_samples, val_samples,
                    device, epochs=10
                )

                elapsed = time.time() - t0

                sweep_results.append({
                    "arch":     arch,
                    "lr":       lr,
                    "dropout":  dropout,
                    "val_acc":  round(val_acc, 2),
                    "time_s":   round(elapsed, 1),
                })

                print(f"  [{done}/{total}] {arch:<18} "
                      f"lr={lr} dropout={dropout} "
                      f"→ val_acc={val_acc:.1f}% "
                      f"({elapsed:.0f}s)")

                if val_acc > best_overall_acc:
                    best_overall_acc   = val_acc
                    best_overall_model = model
                    best_overall_cfg   = {
                        "arch": arch,
                        "lr": lr,
                        "dropout": dropout
                    }

    # Save sweep results
    sweep_df = pd.DataFrame(sweep_results).sort_values(
        "val_acc", ascending=False)
    sweep_df.to_csv(ML_DIR / "cnn_sweep_results.csv", index=False)

    print(f"\nBest config: {best_overall_cfg}")
    print(f"Best val acc: {best_overall_acc:.1f}%")

    #  Test best model ─
    _, val_transform = get_transforms()
    test_ds     = SceneDataset(test_samples, val_transform)
    test_loader = DataLoader(test_ds, batch_size=32,
                              shuffle=False, num_workers=0)
    criterion   = nn.CrossEntropyLoss()

    _, test_acc, test_preds, test_labels = eval_epoch(
        best_overall_model, test_loader, criterion, device
    )

    print(f"\nTest accuracy (best model): {test_acc:.1f}%")
    print("\nPer-class accuracy:")
    for class_name, class_idx in CLASSES.items():
        mask = np.array(test_labels) == class_idx
        if mask.sum() > 0:
            cls_acc = 100.0 * np.array(
                test_preds)[mask].tolist().count(class_idx) / mask.sum()
            print(f"  {class_name}: {cls_acc:.1f}%")

    # Save best model
    torch.save(best_overall_model.state_dict(),
               ML_DIR / "cnn_best_v2.pth")

    #  Architecture comparison plot 
    fig = plt.figure(figsize=(16, 8), facecolor="#0f172a")
    gs  = gridspec.GridSpec(1, 2, figure=fig,
                            hspace=0.3, wspace=0.35)

    # Bar chart: val_acc per architecture (best per arch)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_facecolor("#020617")

    best_per_arch = (sweep_df.groupby("arch")["val_acc"]
                              .max()
                              .reset_index()
                              .sort_values("val_acc", ascending=True))

    colors = ["#3b82f6", "#22c55e", "#eab308",
              "#f97316", "#a855f7", "#ef4444"]
    bars = ax1.barh(best_per_arch["arch"],
                    best_per_arch["val_acc"],
                    color=colors[:len(best_per_arch)],
                    edgecolor="#020617")

    ax1.set_xlabel("Best Val Accuracy %", color="#64748b")
    ax1.set_title("Architecture Comparison",
                  color="#e2e8f0", fontsize=11)
    ax1.set_xlim(80, 101)
    ax1.tick_params(colors="#475569", labelsize=8)
    for spine in ax1.spines.values():
        spine.set_edgecolor("#1e293b")
    for bar, val in zip(bars, best_per_arch["val_acc"]):
        ax1.text(val + 0.1, bar.get_y() + bar.get_height()/2,
                 f"{val:.1f}%", va="center",
                 color="#e2e8f0", fontsize=8)

    # Scatter: lr vs val_acc colored by dropout
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_facecolor("#020617")

    dropout_colors = {0.1: "#3b82f6", 0.2: "#22c55e", 0.3: "#f97316"}
    for dropout, color in dropout_colors.items():
        mask = sweep_df["dropout"] == dropout
        ax2.scatter(
            sweep_df[mask]["lr"],
            sweep_df[mask]["val_acc"],
            c=color, label=f"dropout={dropout}",
            alpha=0.8, s=80
        )

    ax2.set_xscale("log")
    ax2.set_xlabel("Learning Rate", color="#64748b")
    ax2.set_ylabel("Val Accuracy %", color="#64748b")
    ax2.set_title("LR vs Accuracy by Dropout",
                  color="#e2e8f0", fontsize=11)
    ax2.tick_params(colors="#475569")
    ax2.legend(facecolor="#0f172a", labelcolor="#94a3b8",
                edgecolor="#1e293b", fontsize=8)
    for spine in ax2.spines.values():
        spine.set_edgecolor("#1e293b")

    plt.savefig(ML_DIR / "cnn_sweep_plot.png", dpi=150,
                bbox_inches="tight", facecolor="#0f172a")

    #  Final summary
    print("\n" + "=" * 60)
    print("  FINAL SUMMARY")
    print("=" * 60)
    print(f"\n  Architectures tested: {len(architectures)}")
    print(f"  Total combinations:   {total}")
    print(f"  Best architecture:    {best_overall_cfg['arch']}")
    print(f"  Best lr:              {best_overall_cfg['lr']}")
    print(f"  Best dropout:         {best_overall_cfg['dropout']}")
    print(f"  Best val accuracy:    {best_overall_acc:.1f}%")
    print(f"  Test accuracy:        {test_acc:.1f}%")
    print(f"\n  Top 5 configurations:")
    print(sweep_df.head(5)[["arch", "lr", "dropout",
                              "val_acc"]].to_string(index=False))
    print(f"\n  Model saved  → backend/ml/cnn_best_v2.pth")
    print(f"  Sweep saved  → backend/ml/cnn_sweep_results.csv")
    print(f"  Plot saved   → backend/ml/cnn_sweep_plot.png")


if __name__ == "__main__":
    main()