import os, importlib
import torch
from utils.config import load_config, parse_args_and_merge
from data.datasets import create_dataloaders
from engine.trainer import Trainer

DEFAULT_CFG = {
    "seed": 42,
    "dataset": {"name": "cifar10", "root": "./data", "num_classes": 10},
    "data": {"img_size": 32, "batch_size": 128, "num_workers": 4},
    "model": {"name": "simple_cnn", "kwargs": {}},
    "optimizer": {"lr": 3e-3, "weight_decay": 1e-4},
    "scheduler": {"one_cycle": True, "pct_start": 0.1, "div_factor": 25.0, "final_div_factor": 1e4},
    "trainer": {
        "epochs": 20,
        "device": "cuda",
        "amp": True,
        "early_stopping_patience": 5,
        "max_grad_norm": 1.0,
        "checkpoint_dir": "checkpoints",
        "log_dir": "logs",
        "resume": None
    }
}

def set_seed(seed: int):
    import random, numpy as np
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_model(name: str, num_classes: int, kwargs: dict):
    name = name.lower()
    if name == "simple_cnn":
        mod = importlib.import_module("models.simple_cnn")
        return mod.SimpleCNN(num_classes=num_classes, **kwargs)
    else:
        raise ValueError(f"Unknown model: {name}")

def main():
    cfg = parse_args_and_merge(DEFAULT_CFG)
    os.makedirs(cfg["trainer"]["checkpoint_dir"], exist_ok=True)
    os.makedirs(cfg["trainer"]["log_dir"], exist_ok=True)
    set_seed(cfg["seed"])

    train_loader, val_loader, inferred = create_dataloaders(cfg)
    if cfg["dataset"]["name"].lower() == "cifar10":
        cfg["dataset"]["num_classes"] = inferred

    model = get_model(cfg["model"]["name"], cfg["dataset"]["num_classes"], cfg["model"]["kwargs"])
    trainer = Trainer(model, cfg)
    best = trainer.fit(train_loader, val_loader)
    print(f"Best val accuracy: {best:.4f}")

if __name__ == "__main__":
    main()
