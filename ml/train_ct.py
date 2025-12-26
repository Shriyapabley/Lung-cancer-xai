# ml/train_ct.py

import os
import time
from typing import Dict

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms

from datasets.ct_dataset import CTImageDataset
from models.ct_cnn import CTCNNModel


def get_dataloaders(
    data_root: str,
    batch_size: int = 16,
    img_size: int = 224,
) -> Dict[str, DataLoader]:

    train_dir = os.path.join(data_root, "train")
    val_dir = os.path.join(data_root, "val")

    train_transform = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ]
    )

    val_transform = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ]
    )

    # Infer classes from train_dir
    train_dataset = CTImageDataset(root_dir=train_dir, transform=train_transform)
    val_dataset = CTImageDataset(
        root_dir=val_dir,
        transform=val_transform,
        class_names=train_dataset.class_names,
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=2
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=2
    )

    return {"train": train_loader, "val": val_loader}


def train_ct_model(
    data_root: str = "../data/ct",
    num_classes: int = 4,
    num_epochs: int = 20,
    batch_size: int = 16,
    lr: float = 1e-4,
    device: str = None,
    output_dir: str = "checkpoints",
):

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    os.makedirs(output_dir, exist_ok=True)

    dataloaders = get_dataloaders(data_root, batch_size=batch_size)
    train_loader = dataloaders["train"]
    val_loader = dataloaders["val"]

    model = CTCNNModel(num_classes=num_classes, pretrained=True).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_val_acc = 0.0
    best_model_path = os.path.join(output_dir, "ct_resnet18_best.pth")

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        print("-" * 30)

        # Each epoch has a training and validation phase
        for phase in ["train", "val"]:
            if phase == "train":
                model.train()
                loader = train_loader
            else:
                model.eval()
                loader = val_loader

            running_loss = 0.0
            running_corrects = 0
            total_samples = 0

            start_time = time.time()

            for inputs, labels in loader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data).item()
                total_samples += inputs.size(0)

            epoch_loss = running_loss / total_samples
            epoch_acc = running_corrects / total_samples

            elapsed = time.time() - start_time
            print(
                f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} "
                f"({elapsed:.1f}s)"
            )

            # deep copy the model
            if phase == "val" and epoch_acc > best_val_acc:
                best_val_acc = epoch_acc
                torch.save(model.state_dict(), best_model_path)
                print(f"New best model saved to {best_model_path}")

        print()

    print(f"Training complete. Best val Acc: {best_val_acc:.4f}")
    return best_model_path


if __name__ == "__main__":
    train_ct_model()
