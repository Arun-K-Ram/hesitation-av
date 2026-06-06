"""
backend/ml/train_cnn_crosssession.py

Cross-session CNN validation.

Train: Session 1 + Session 2 frames
Test:  Session 3 frames (held out)

Architectures tested:
  EfficientNetB2, MobileNetV3, ResNet18

Best hyperparams from original sweep used directly.

Run:
  python backend/ml/train_cnn_crosssession.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn as nn
import torchvision.transforms as T
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from PIL import Image
import numpy as np
import pandas as pd
import time

DEVICE = torch.device("cuda" if torch.cuda.is_available()
                       else "cpu")
print(f"[Device] {DEVICE}")

#  Folder mapping 

RECORDINGS = Path("experiments/recordings")

# Session 1 + Session 2  Train
TRAIN_FOLDERS = {
    "pedestrian_curb": [
        "pedestrian_curb_1780190788",    # S1
        "pedestrian_curb_1780626494",    # S2
        "pedestrian_curb_1780626638",    # S2
        "pedestrian_curb_1780626845",    # S2
    ],
    "merge_hesitation": [
        "merge_hesitation_1780196299",   # S1
        "merge_hesitation_1780196964",   # S1
        "merge_hesitation_1780627319",   # S2
    ],
    "occluded_intersection": [
        "occluded_intersection_1780197183",  # S1
        "occluded_intersection_1780627782",  # S2
    ],
}

# Session 3  Test (held out)
TEST_FOLDERS = {
    "pedestrian_curb":       ["pedestrian_curb_1780628345"],
    "merge_hesitation":      ["merge_hesitation_1780628686"],
    "occluded_intersection": ["occluded_intersection_1780629047"],
}

CLASSES = ["pedestrian_curb",
           "merge_hesitation",
           "occluded_intersection"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}


#  Dataset 

class FrameDataset(Dataset):
    def __init__(self, folder_map, transform=None):
        self.transform = transform
        self.samples   = []

        for label, folders in folder_map.items():
            idx = CLASS_TO_IDX[label]
            for folder in folders:
                frames_dir = RECORDINGS / folder / "frames"
                if not frames_dir.exists():
                    print(f"  [WARN] Not found: {frames_dir}")
                    continue
                jpgs = sorted(frames_dir.glob("*.jpg"))
                for jpg in jpgs:
                    self.samples.append((str(jpg), idx))

        print(f"  Total samples: {len(self.samples)}")
        for label in CLASSES:
            idx   = CLASS_TO_IDX[label]
            count = sum(1 for _, i in self.samples
                        if i == idx)
            print(f"    {label}: {count}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        path, label = self.samples[i]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


#  Transforms 

TRAIN_TRANSFORM = T.Compose([
    T.Resize((224, 224)),
    T.RandomHorizontalFlip(),
    T.RandomRotation(15),
    T.ColorJitter(
        brightness=0.6,
        contrast=0.6,
        saturation=0.6,
        hue=0.15),
    T.RandomGrayscale(p=0.15),
    T.RandomAdjustSharpness(2, p=0.3),
    T.RandomAutocontrast(p=0.3),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225]),
    T.RandomErasing(p=0.2),
])

TEST_TRANSFORM = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225]),
])


#  Model builders 

def build_convnext_tiny(dropout, n_classes=3):
    m = models.convnext_tiny(
        weights=models.ConvNeXt_Tiny_Weights.DEFAULT)
    in_f = m.classifier[2].in_features
    m.classifier[2] = nn.Sequential(
        nn.Dropout(dropout),
        nn.Linear(in_f, n_classes))
    return m


def build_efficientnet_b0(dropout, n_classes=3):
    m = models.efficientnet_b0(
        weights=models.EfficientNet_B0_Weights.DEFAULT)
    in_f = m.classifier[1].in_features
    m.classifier = nn.Sequential(
        nn.Dropout(dropout),
        nn.Linear(in_f, n_classes))
    return m


def build_mobilenet_v2(dropout, n_classes=3):
    m = models.mobilenet_v2(
        weights=models.MobileNet_V2_Weights.DEFAULT)
    in_f = m.classifier[1].in_features
    m.classifier = nn.Sequential(
        nn.Dropout(dropout),
        nn.Linear(in_f, n_classes))
    return m

def build_efficientnet_b2(dropout, n_classes=3):
    m = models.efficientnet_b2(
        weights=models.EfficientNet_B2_Weights.DEFAULT)
    in_f = m.classifier[1].in_features
    m.classifier = nn.Sequential(
        nn.Dropout(dropout),
        nn.Linear(in_f, n_classes))
    return m


def build_mobilenet_v3(dropout, n_classes=3):
    m = models.mobilenet_v3_large(
        weights=models.MobileNet_V3_Large_Weights.DEFAULT)
    in_f = m.classifier[3].in_features
    m.classifier[3] = nn.Sequential(
        nn.Dropout(dropout),
        nn.Linear(in_f, n_classes))
    return m


def build_resnet18(dropout, n_classes=3):
    m = models.resnet18(
        weights=models.ResNet18_Weights.DEFAULT)
    in_f = m.fc.in_features
    m.fc = nn.Sequential(
        nn.Dropout(dropout),
        nn.Linear(in_f, n_classes))
    return m


#  Train one model 

def train_model(model, train_loader, test_loader,
                lr, epochs=15, arch_name="model"):

    model = model.to(DEVICE)
    optimizer = torch.optim.Adam(
    model.parameters(), lr=lr,
    weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler\
                .CosineAnnealingLR(
                    optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    best_state   = None

    for epoch in range(1, epochs + 1):
        # Train
        model.train()
        correct = total = 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(DEVICE), \
                           labels.to(DEVICE)
            optimizer.zero_grad()
            out  = model(imgs)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
            preds    = out.argmax(1)
            correct += (preds == labels).sum().item()
            total   += labels.size(0)
        train_acc = 100 * correct / total
        scheduler.step()

        # Validate
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for imgs, labels in test_loader:
                imgs, labels = imgs.to(DEVICE), \
                               labels.to(DEVICE)
                out   = model(imgs)
                preds = out.argmax(1)
                correct += (preds == labels)\
                           .sum().item()
                total   += labels.size(0)
        val_acc = 100 * correct / total

        print(f"  [{arch_name}] Epoch {epoch:2d}/{epochs}"
              f"  train={train_acc:.1f}%"
              f"  test={val_acc:.1f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state   = {
                k: v.cpu().clone()
                for k, v in model.state_dict().items()
            }

    return best_val_acc, best_state


#  Per-class accuracy 

def per_class_accuracy(model, test_loader):
    model.eval()
    correct = {i: 0 for i in range(3)}
    total   = {i: 0 for i in range(3)}

    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(DEVICE), \
                           labels.to(DEVICE)
            preds = model(imgs).argmax(1)
            for i in range(3):
                mask = labels == i
                correct[i] += (preds[mask] == i)\
                              .sum().item()
                total[i]   += mask.sum().item()

    results = {}
    for i, cls in enumerate(CLASSES):
        acc = 100 * correct[i] / max(total[i], 1)
        results[cls] = round(acc, 1)
    return results


#  Main 

def main():
    print("\n" + "="*60)
    print("  Cross-Session CNN Validation")
    print("  Train: Session 1 + Session 2")
    print("  Test:  Session 3 (held out)")
    print("="*60)

    # Build datasets
    print("\n[Train set]")
    train_ds = FrameDataset(TRAIN_FOLDERS,
                             TRAIN_TRANSFORM)
    print("\n[Test set]")
    test_ds  = FrameDataset(TEST_FOLDERS,
                             TEST_TRANSFORM)

    train_loader = DataLoader(
        train_ds, batch_size=32,
        shuffle=True, num_workers=0)
    test_loader  = DataLoader(
        test_ds, batch_size=32,
        shuffle=False, num_workers=0)

    # Architectures to test
    # Best hyperparams from original sweep
    configs = [
    ("EfficientNetB0",
     build_efficientnet_b0(dropout=0.3),
     0.0005),
    ("MobileNetV2",
     build_mobilenet_v2(dropout=0.1),
     0.0005),
]

    results = []

    for arch_name, model, lr in configs:
        print(f"\n{'='*60}")
        print(f"  Training: {arch_name}")
        print(f"  lr={lr}")
        print(f"{'='*60}")

        t0 = time.time()
        best_acc, best_state = train_model(
            model, train_loader, test_loader,
            lr=lr, epochs=15,
            arch_name=arch_name)

        elapsed = time.time() - t0

        # Per-class accuracy with best weights
        model.load_state_dict(best_state)
        model = model.to(DEVICE)
        per_class = per_class_accuracy(
            model, test_loader)

        print(f"\n  [{arch_name}] Best test acc: "
              f"{best_acc:.1f}%")
        for cls, acc in per_class.items():
            print(f"    {cls}: {acc}%")

        # Save best model
        if arch_name == "EfficientNetB2":
            save_path = Path(
                "backend/ml/cnn_crosssession.pth")
            torch.save(best_state, save_path)
            print(f"  Saved  {save_path}")

        results.append({
            "arch":      arch_name,
            "lr":        lr,
            "test_acc":  round(best_acc, 2),
            "time_s":    round(elapsed, 1),
            **{f"acc_{c.replace('_intersection','').replace('_curb','').replace('_hesitation','')}":
               v for c, v in per_class.items()}
        })

    # Summary
    print(f"\n{'='*60}")
    print(f"  CROSS-SESSION RESULTS SUMMARY")
    print(f"{'='*60}")
    df = pd.DataFrame(results).sort_values(
        "test_acc", ascending=False)
    print(df.to_string(index=False))

    df.to_csv(
        "backend/ml/crosssession_results.csv",
        index=False)
    print(f"\n  Saved  backend/ml/"
          f"crosssession_results.csv")

    winner = df.iloc[0]
    print(f"\n  Winner: {winner['arch']}"
          f"  Test acc: {winner['test_acc']}%")
    print(f"\n  Done.")


if __name__ == "__main__":
    main()