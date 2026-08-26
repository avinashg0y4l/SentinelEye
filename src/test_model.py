from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms


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

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=torch.cuda.is_available(),
)


# --------------------------------------------------
# Load pretrained ResNet-18
# --------------------------------------------------

print("\nLoading ResNet-18...")

model = models.resnet18(
    weights=models.ResNet18_Weights.DEFAULT
)


# --------------------------------------------------
# Replace classification layer
# --------------------------------------------------

num_features = model.fc.in_features

model.fc = nn.Linear(
    num_features,
    2
)


# --------------------------------------------------
# Move model to GPU
# --------------------------------------------------

model = model.to(device)


print("Model loaded.")
print("Classification classes:", 2)


# --------------------------------------------------
# Get one batch
# --------------------------------------------------

images, labels = next(iter(train_loader))

images = images.to(device)
labels = labels.to(device)


# --------------------------------------------------
# Forward pass
# --------------------------------------------------

print("\nRunning forward pass...")

with torch.no_grad():
    outputs = model(images)


# --------------------------------------------------
# Output information
# --------------------------------------------------

print("\nForward pass successful.")

print("Input shape :", images.shape)
print("Output shape:", outputs.shape)
print("Labels shape:", labels.shape)

print("\nSample output:")
print(outputs[:3])

print("\nPredicted classes:")
print(torch.argmax(outputs, dim=1)[:10])

print("\nActual classes:")
print(labels[:10])