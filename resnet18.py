
import os
import copy
import random
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import datasets, transforms, models
from torchvision.models import ResNet18_Weights

# ---------------------------
# 1. Configuration
# ---------------------------
SEED = 42
BATCH_SIZE = 32
EPOCHS = 15
LEARNING_RATE = 1e-3
NUM_CLASSES = 37
DATA_DIR = "./data"
MODEL_PATH = "best_resnet18_oxford_pet.pth"

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ---------------------------
# 2. Image transformations
# ---------------------------
weights = ResNet18_Weights.DEFAULT

train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(
        brightness=0.15, contrast=0.15, saturation=0.15
    ),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])

eval_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])

# Apply different transforms to the same underlying dataset.
class TransformSubset(Dataset):
    def __init__(self, base_dataset, indices, transform):
        self.base_dataset = base_dataset
        self.indices = indices
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        image, label = self.base_dataset[self.indices[index]]
        image = self.transform(image)
        return image, label

# ---------------------------
# 3. Download dataset
# ---------------------------
base_train = datasets.OxfordIIITPet(
    root=DATA_DIR,
    split="trainval",
    target_types="category",
    download=True,
    transform=None,
)

test_base = datasets.OxfordIIITPet(
    root=DATA_DIR,
    split="test",
    target_types="category",
    download=True,
    transform=None,
)

# Reproducible 80/20 train-validation split
num_samples = len(base_train)
indices = torch.randperm(
    num_samples, generator=torch.Generator().manual_seed(SEED)
).tolist()

split = int(0.8 * num_samples)
train_indices = indices[:split]
val_indices = indices[split:]

train_dataset = TransformSubset(
    base_train, train_indices, train_transform
)
val_dataset = TransformSubset(
    base_train, val_indices, eval_transform
)

class TransformedDataset(Dataset):
    def __init__(self, base_dataset, transform):
        self.base_dataset = base_dataset
        self.transform = transform

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, index):
        image, label = self.base_dataset[index]
        return self.transform(image), label

test_dataset = TransformedDataset(test_base, eval_transform)

train_loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE,
    shuffle=True, num_workers=0,
    pin_memory=(device.type == "cuda")
)
val_loader = DataLoader(
    val_dataset, batch_size=BATCH_SIZE,
    shuffle=False, num_workers=0,
    pin_memory=(device.type == "cuda")
)
test_loader = DataLoader(
    test_dataset, batch_size=BATCH_SIZE,
    shuffle=False, num_workers=0,
    pin_memory=(device.type == "cuda")
)

print(f"Training images:   {len(train_dataset)}")
print(f"Validation images: {len(val_dataset)}")
print(f"Test images:       {len(test_dataset)}")

# ---------------------------
# 4. Load pretrained ResNet18
# ---------------------------
model = models.resnet18(weights=weights)

# Replace the original 1000-class ImageNet output layer.
model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(
    model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4
)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="max", factor=0.5, patience=2
)

# ---------------------------
# 5. Train and evaluate one epoch
# ---------------------------
def run_epoch(model, loader, criterion, optimizer=None):
    training = optimizer is not None
    model.train() if training else model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.set_grad_enabled(training):
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            if training:
                optimizer.zero_grad(set_to_none=True)

            outputs = model(images)
            loss = criterion(outputs, labels)

            if training:
                loss.backward()
                optimizer.step()

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            total_correct += (
                outputs.argmax(dim=1) == labels
            ).sum().item()
            total_samples += batch_size

    return (
        total_loss / total_samples,
        100.0 * total_correct / total_samples,
    )

# ---------------------------
# 6. Main training loop
# ---------------------------
best_val_acc = 0.0
best_state = copy.deepcopy(model.state_dict())

history = {
    "train_loss": [],
    "train_acc": [],
    "val_loss": [],
    "val_acc": [],
}

for epoch in range(EPOCHS):
    train_loss, train_acc = run_epoch(
        model, train_loader, criterion, optimizer
    )
    val_loss, val_acc = run_epoch(
        model, val_loader, criterion
    )

    scheduler.step(val_acc)

    history["train_loss"].append(train_loss)
    history["train_acc"].append(train_acc)
    history["val_loss"].append(val_loss)
    history["val_acc"].append(val_acc)

    print(
        f"Epoch [{epoch + 1:02d}/{EPOCHS}] "
        f"Train Loss: {train_loss:.4f} | "
        f"Train Acc: {train_acc:.2f}% | "
        f"Val Loss: {val_loss:.4f} | "
        f"Val Acc: {val_acc:.2f}%"
    )

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_state = copy.deepcopy(model.state_dict())

        torch.save({
            "model_state_dict": best_state,
            "num_classes": NUM_CLASSES,
            "class_names": base_train.classes,
            "best_val_acc": best_val_acc,
        }, MODEL_PATH)

        print(f"  Saved best model: {MODEL_PATH}")

# Restore the best validation checkpoint.
model.load_state_dict(best_state)
print(f"\nBest validation accuracy: {best_val_acc:.2f}%")

# ---------------------------
# 7. Final test evaluation
# ---------------------------
test_loss, test_acc = run_epoch(
    model, test_loader, criterion
)

print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_acc:.2f}%")

# ---------------------------
# 8. Plot learning curves
# ---------------------------
epochs_range = range(1, EPOCHS + 1)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].plot(epochs_range, history["train_loss"], label="Train")
axes[0].plot(epochs_range, history["val_loss"], label="Validation")
axes[0].set_title("Loss")
axes[0].set_xlabel("Epoch")
axes[0].legend()

axes[1].plot(epochs_range, history["train_acc"], label="Train")
axes[1].plot(epochs_range, history["val_acc"], label="Validation")
axes[1].set_title("Accuracy")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Accuracy (%)")
axes[1].legend()

plt.tight_layout()
plt.show()
