from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms


# ============================================================
# Configuration
# ============================================================

TEST_ROOT = Path("data/processed/FLAME/Test")
MODEL_PATH = Path("models/baseline_v1_best.pth")

IMAGE_SIZE = 224
BATCH_SIZE = 32
NUM_WORKERS = 0


# ============================================================
# Device
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 60)
print("SentinelEye - FLAME Test Evaluation")
print("=" * 60)

print("Device:", device)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))


# ============================================================
# Transform
# ============================================================

test_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ============================================================
# Test Dataset
# ============================================================

test_dataset = datasets.ImageFolder(
    TEST_ROOT,
    transform=test_transform
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=torch.cuda.is_available()
)


print("\nClasses:")
print(test_dataset.class_to_idx)

print("\nTest images:", len(test_dataset))


# ============================================================
# Load Model
# ============================================================

print("\nLoading best model...")

checkpoint = torch.load(
    MODEL_PATH,
    map_location=device
)

model = models.resnet18(weights=None)

num_features = model.fc.in_features

model.fc = nn.Linear(
    num_features,
    2
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model = model.to(device)

model.eval()

print("Best epoch:", checkpoint["epoch"])
print("Validation F1:", checkpoint["best_f1"])


# ============================================================
# Evaluation
# ============================================================

tp = 0
tn = 0
fp = 0
fn = 0

total = 0
correct = 0

test_loss = 0.0

criterion = nn.CrossEntropyLoss()


with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(
            device,
            non_blocking=True
        )

        labels = labels.to(
            device,
            non_blocking=True
        )

        outputs = model(images)

        loss = criterion(
            outputs,
            labels
        )

        predictions = torch.argmax(
            outputs,
            dim=1
        )

        test_loss += (
            loss.item() * images.size(0)
        )

        total += labels.size(0)

        correct += (
            predictions == labels
        ).sum().item()

        # Fire = 0
        # No_Fire = 1

        tp += (
            ((predictions == 0) &
             (labels == 0))
            .sum()
            .item()
        )

        fn += (
            ((predictions == 1) &
             (labels == 0))
            .sum()
            .item()
        )

        fp += (
            ((predictions == 0) &
             (labels == 1))
            .sum()
            .item()
        )

        tn += (
            ((predictions == 1) &
             (labels == 1))
            .sum()
            .item()
        )


# ============================================================
# Metrics
# ============================================================

test_loss /= total

accuracy = correct / total

precision = (
    tp / (tp + fp)
    if (tp + fp) > 0
    else 0.0
)

recall = (
    tp / (tp + fn)
    if (tp + fn) > 0
    else 0.0
)

f1 = (
    2 * precision * recall
    / (precision + recall)
    if (precision + recall) > 0
    else 0.0
)


# ============================================================
# Results
# ============================================================

print("\n" + "=" * 60)
print("FINAL FLAME TEST RESULTS")
print("=" * 60)

print(f"Test Loss:       {test_loss:.4f}")
print(f"Accuracy:        {accuracy:.4f}")
print(f"Precision:       {precision:.4f}")
print(f"Fire Recall:     {recall:.4f}")
print(f"F1:              {f1:.4f}")

print("\nConfusion Matrix:")
print("                 Predicted")
print("              Fire    No_Fire")
print(f"Actual Fire   {tp:5d}   {fn:7d}")
print(f"Actual NoFire {fp:5d}   {tn:7d}")

print("\nTotal:", total)
print("Correct:", correct)
print("Incorrect:", total - correct)