from pathlib import Path
import shutil

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms


# ============================================================
# Configuration
# ============================================================

TEST_ROOT = Path("data/processed/FLAME/Test")
MODEL_PATH = Path("models/baseline_v1_best.pth")

OUTPUT_ROOT = Path("results/baseline_v1/errors")

FN_DIR = OUTPUT_ROOT / "false_negatives"
FP_DIR = OUTPUT_ROOT / "false_positives"

FN_DIR.mkdir(parents=True, exist_ok=True)
FP_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_SIZE = 224
BATCH_SIZE = 32
NUM_WORKERS = 0

# Number of representative errors to save
MAX_ERRORS = 100


# ============================================================
# Device
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 60)
print("SentinelEye - Error Analysis")
print("=" * 60)

print("Device:", device)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))


# ============================================================
# Transform
# ============================================================

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ============================================================
# Dataset
# ============================================================

dataset = datasets.ImageFolder(
    TEST_ROOT,
    transform=transform
)

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=torch.cuda.is_available()
)

print("\nClasses:")
print(dataset.class_to_idx)

print("Test images:", len(dataset))


# ============================================================
# Model
# ============================================================

print("\nLoading best model...")

checkpoint = torch.load(
    MODEL_PATH,
    map_location=device
)

model = models.resnet18(
    weights=None
)

model.fc = nn.Linear(
    model.fc.in_features,
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
# Error counters
# ============================================================

false_negatives = []
false_positives = []

total = 0
correct = 0


# ============================================================
# Prediction
# ============================================================

with torch.no_grad():

    dataset_index = 0

    for images, labels in loader:

        images = images.to(
            device,
            non_blocking=True
        )

        labels = labels.to(
            device,
            non_blocking=True
        )

        outputs = model(images)

        probabilities = torch.softmax(
            outputs,
            dim=1
        )

        predictions = torch.argmax(
            outputs,
            dim=1
        )

        for i in range(len(labels)):

            actual = labels[i].item()
            predicted = predictions[i].item()

            confidence = probabilities[
                i,
                predicted
            ].item()

            image_path, _ = dataset.samples[
                dataset_index
            ]

            total += 1

            if actual == predicted:

                correct += 1

            # ------------------------------------------------
            # False Negative
            #
            # Actual Fire = 0
            # Predicted No_Fire = 1
            # ------------------------------------------------

            elif actual == 0 and predicted == 1:

                false_negatives.append({
                    "path": image_path,
                    "confidence": confidence,
                    "actual": actual,
                    "predicted": predicted
                })

            # ------------------------------------------------
            # False Positive
            #
            # Actual No_Fire = 1
            # Predicted Fire = 0
            # ------------------------------------------------

            elif actual == 1 and predicted == 0:

                false_positives.append({
                    "path": image_path,
                    "confidence": confidence,
                    "actual": actual,
                    "predicted": predicted
                })

            dataset_index += 1


# ============================================================
# Sort by confidence
# ============================================================

# For false negatives:
# highest confidence in the WRONG No_Fire prediction first

false_negatives.sort(
    key=lambda x: x["confidence"],
    reverse=True
)

# For false positives:
# highest confidence in the WRONG Fire prediction first

false_positives.sort(
    key=lambda x: x["confidence"],
    reverse=True
)


# ============================================================
# Copy representative errors
# ============================================================

print("\nCopying representative errors...")


for index, item in enumerate(
    false_negatives[:MAX_ERRORS],
    start=1
):

    source = Path(item["path"])

    destination = (
        FN_DIR
        /
        f"{index:03d}_"
        f"conf_{item['confidence']:.4f}_"
        f"{source.name}"
    )

    shutil.copy2(
        source,
        destination
    )


for index, item in enumerate(
    false_positives[:MAX_ERRORS],
    start=1
):

    source = Path(item["path"])

    destination = (
        FP_DIR
        /
        f"{index:03d}_"
        f"conf_{item['confidence']:.4f}_"
        f"{source.name}"
    )

    shutil.copy2(
        source,
        destination
    )


# ============================================================
# Print summary
# ============================================================

print("\n" + "=" * 60)
print("ERROR ANALYSIS SUMMARY")
print("=" * 60)

print("Total images:", total)
print("Correct:", correct)
print("Incorrect:", total - correct)

print(
    "\nFalse Negatives "
    "(Fire -> No_Fire):",
    len(false_negatives)
)

print(
    "False Positives "
    "(No_Fire -> Fire):",
    len(false_positives)
)

print("\nSaved:")
print(
    "False Negatives:",
    FN_DIR
)

print(
    "False Positives:",
    FP_DIR
)

print(
    f"\nSaved first {MAX_ERRORS} errors from each category."
)