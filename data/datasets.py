from typing import Tuple, Dict, Any
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as T
from torchvision.datasets import CIFAR10, ImageFolder

def build_transforms(img_size: int = 32) -> Tuple[Any, Any]:
    train_tfms = T.Compose([
        T.Resize((img_size, img_size)),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    val_tfms = T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    return train_tfms, val_tfms

def create_dataloaders(cfg: Dict[str, Any]) -> Tuple[DataLoader, DataLoader, int]:
    name = cfg["dataset"]["name"].lower()
    root = cfg["dataset"]["root"]
    img_size = cfg["data"]["img_size"]
    batch_size = cfg["data"]["batch_size"]
    num_workers = cfg["data"]["num_workers"]

    train_tfms, val_tfms = build_transforms(img_size)

    if name == "cifar10":
        train_ds = CIFAR10(root=root, train=True, download=True, transform=train_tfms)
        val_ds   = CIFAR10(root=root, train=False, download=True, transform=val_tfms)
        num_classes = 10
    elif name == "imagefolder":
        train_ds = ImageFolder(root=f"{root}/train", transform=train_tfms)
        val_ds   = ImageFolder(root=f"{root}/val",   transform=val_tfms)
        num_classes = cfg["dataset"]["num_classes"]
    else:
        raise ValueError(f"Unknown dataset: {name}")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader, num_classes
