from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# --------------------------------------------------
# Configuration
# --------------------------------------------------

DATASET_ROOT = Path("data/experiments/baseline_v1")

IMAGE_SIZE = 224
BATCH_SIZE = 32
NUM_WORKERS = 0


# --------------------------------------------------
# Device
# --------------------------------------------------

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))


# --------------------------------------------------
# Transform
# --------------------------------------------------

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
])


# --------------------------------------------------
# Dataset
# --------------------------------------------------

train_dataset = datasets.ImageFolder(
    DATASET_ROOT / "train",
    transform=transform
)

val_dataset = datasets.ImageFolder(
    DATASET_ROOT / "val",
    transform=transform
)


# --------------------------------------------------
# DataLoader
# --------------------------------------------------

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=torch.cuda.is_available(),
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=torch.cuda.is_available(),
)


# --------------------------------------------------
# Dataset information
# --------------------------------------------------

print("\nClasses:")
print(train_dataset.class_to_idx)

print("\nTraining images:", len(train_dataset))
print("Validation images:", len(val_dataset))


# --------------------------------------------------
# Get one batch
# --------------------------------------------------

images, labels = next(iter(train_loader))

print("\nBatch information:")
print("Images shape:", images.shape)
print("Labels shape:", labels.shape)

print("Image dtype:", images.dtype)
print("Label dtype:", labels.dtype)

print("Image min:", images.min().item())
print("Image max:", images.max().item())


# --------------------------------------------------
# Move batch to GPU
# --------------------------------------------------

images = images.to(device)
labels = labels.to(device)

print("\nAfter moving to device:")
print("Images device:", images.device)
print("Labels device:", labels.device)