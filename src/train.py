from pathlib import Path
import json
import time

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms


# ============================================================
# Configuration
# ============================================================

DATASET_ROOT = Path("data/experiments/baseline_v1")

MODEL_DIR = Path("models")
RESULTS_DIR = Path("results/baseline_v1")

MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_SIZE = 224
BATCH_SIZE = 32
NUM_WORKERS = 0

EPOCHS = 10
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4

SEED = 42


# ============================================================
# Reproducibility
# ============================================================

torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ============================================================
# Device
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 60)
print("SentinelEye - Baseline Training")
print("=" * 60)

print("Device:", device)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print(
        "VRAM:",
        round(
            torch.cuda.get_device_properties(0).total_memory
            / 1024**3,
            2
        ),
        "GB"
    )


# ============================================================
# Transforms
# ============================================================

train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),

    transforms.RandomHorizontalFlip(p=0.5),

    transforms.RandomRotation(
        degrees=10
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])


val_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])


# ============================================================
# Dataset
# ============================================================

train_dataset = datasets.ImageFolder(
    DATASET_ROOT / "train",
    transform=train_transform
)

val_dataset = datasets.ImageFolder(
    DATASET_ROOT / "val",
    transform=val_transform
)


print("\nClasses:")
print(train_dataset.class_to_idx)

print("\nTraining images:", len(train_dataset))
print("Validation images:", len(val_dataset))


# ============================================================
# DataLoaders
# ============================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=torch.cuda.is_available()
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=torch.cuda.is_available()
)


# ============================================================
# Calculate class weights
# ============================================================

class_counts = []

for class_name in train_dataset.classes:

    class_dir = DATASET_ROOT / "train" / class_name

    count = len([
        p for p in class_dir.iterdir()
        if p.is_file()
    ])

    class_counts.append(count)


print("\nClass counts:")

for class_name, count in zip(
    train_dataset.classes,
    class_counts
):
    print(f"{class_name}: {count}")


total_samples = sum(class_counts)

class_weights = [
    total_samples / (len(class_counts) * count)
    for count in class_counts
]

class_weights = torch.tensor(
    class_weights,
    dtype=torch.float32
).to(device)

print("\nClass weights:")

for class_name, weight in zip(
    train_dataset.classes,
    class_weights
):
    print(
        f"{class_name}: {weight.item():.4f}"
    )


# ============================================================
# Model
# ============================================================

print("\nLoading pretrained ResNet-18...")

model = models.resnet18(
    weights=models.ResNet18_Weights.DEFAULT
)

num_features = model.fc.in_features

model.fc = nn.Linear(
    num_features,
    len(train_dataset.classes)
)

model = model.to(device)


# ============================================================
# Loss
# ============================================================

criterion = nn.CrossEntropyLoss(
    weight=class_weights
)


# ============================================================
# Optimizer
# ============================================================

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)


# ============================================================
# Mixed precision
# ============================================================

use_amp = torch.cuda.is_available()

if use_amp:
    scaler = torch.amp.GradScaler("cuda")


# ============================================================
# Metrics
# ============================================================

def calculate_metrics(
    true_positive,
    false_positive,
    false_negative,
    true_negative
):

    accuracy = (
        (true_positive + true_negative)
        /
        (
            true_positive
            + true_negative
            + false_positive
            + false_negative
        )
    )

    precision = (
        true_positive
        /
        (true_positive + false_positive)
        if (true_positive + false_positive) > 0
        else 0.0
    )

    recall = (
        true_positive
        /
        (true_positive + false_negative)
        if (true_positive + false_negative) > 0
        else 0.0
    )

    f1 = (
        2 * precision * recall
        /
        (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return accuracy, precision, recall, f1


# ============================================================
# Training
# ============================================================

history = []

best_f1 = -1.0
best_epoch = 0

for epoch in range(EPOCHS):

    epoch_start = time.time()

    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    model.train()

    train_loss = 0.0
    train_correct = 0
    train_total = 0

    for images, labels in train_loader:

        images = images.to(
            device,
            non_blocking=True
        )

        labels = labels.to(
            device,
            non_blocking=True
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        if use_amp:

            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16
            ):

                outputs = model(images)

                loss = criterion(
                    outputs,
                    labels
                )

            scaler.scale(loss).backward()

            scaler.step(optimizer)

            scaler.update()

        else:

            outputs = model(images)

            loss = criterion(
                outputs,
                labels
            )

            loss.backward()

            optimizer.step()


        train_loss += (
            loss.item() * images.size(0)
        )

        predictions = torch.argmax(
            outputs,
            dim=1
        )

        train_correct += (
            (predictions == labels)
            .sum()
            .item()
        )

        train_total += labels.size(0)


    train_loss /= train_total

    train_accuracy = (
        train_correct / train_total
    )


    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    model.eval()

    val_loss = 0.0

    tp = 0
    tn = 0
    fp = 0
    fn = 0

    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(
                device,
                non_blocking=True
            )

            labels = labels.to(
                device,
                non_blocking=True
            )

            if use_amp:

                with torch.autocast(
                    device_type="cuda",
                    dtype=torch.float16
                ):

                    outputs = model(images)

                    loss = criterion(
                        outputs,
                        labels
                    )

            else:

                outputs = model(images)

                loss = criterion(
                    outputs,
                    labels
                )


            val_loss += (
                loss.item() * images.size(0)
            )

            predictions = torch.argmax(
                outputs,
                dim=1
            )


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


    val_loss /= len(val_dataset)

    accuracy, precision, recall, f1 = calculate_metrics(
        tp,
        fp,
        fn,
        tn
    )


    # --------------------------------------------------------
    # Save best model
    # --------------------------------------------------------

    if f1 > best_f1:

        best_f1 = f1
        best_epoch = epoch + 1

        checkpoint = {
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "class_to_idx": train_dataset.class_to_idx,
            "image_size": IMAGE_SIZE,
            "best_f1": best_f1,
        }

        torch.save(
            checkpoint,
            MODEL_DIR / "baseline_v1_best.pth"
        )


    # --------------------------------------------------------
    # Record history
    # --------------------------------------------------------

    epoch_time = time.time() - epoch_start

    result = {
        "epoch": epoch + 1,
        "train_loss": train_loss,
        "train_accuracy": train_accuracy,
        "val_loss": val_loss,
        "val_accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "epoch_time_seconds": epoch_time,
    }

    history.append(result)


    # --------------------------------------------------------
    # Print results
    # --------------------------------------------------------

    print(
        f"\nEpoch {epoch + 1}/{EPOCHS}"
    )

    print(
        f"Train Loss: {train_loss:.4f}"
    )

    print(
        f"Train Accuracy: "
        f"{train_accuracy:.4f}"
    )

    print(
        f"Val Loss: {val_loss:.4f}"
    )

    print(
        f"Val Accuracy: "
        f"{accuracy:.4f}"
    )

    print(
        f"Precision: {precision:.4f}"
    )

    print(
        f"Fire Recall: {recall:.4f}"
    )

    print(
        f"F1: {f1:.4f}"
    )

    print(
        f"Confusion Matrix:"
    )

    print(
        f"  TP={tp}  FN={fn}"
    )

    print(
        f"  FP={fp}  TN={tn}"
    )

    print(
        f"Time: {epoch_time:.1f}s"
    )


# ============================================================
# Save training history
# ============================================================

with open(
    RESULTS_DIR / "history.json",
    "w"
) as f:

    json.dump(
        history,
        f,
        indent=4
    )


# ============================================================
# Final information
# ============================================================

print("\n" + "=" * 60)
print("Training completed")
print("=" * 60)

print(
    f"Best validation F1: "
    f"{best_f1:.4f}"
)

print(
    f"Best epoch: "
    f"{best_epoch}"
)

print(
    "Best model:"
)

print(
    MODEL_DIR / "baseline_v1_best.pth"
)

print(
    "History:"
)

print(
    RESULTS_DIR / "history.json"
)