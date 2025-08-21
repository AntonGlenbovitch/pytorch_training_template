import torch, os, importlib, argparse, json
from utils.config import load_config, parse_args_and_merge
from data.datasets import create_dataloaders

def load_model_from_ckpt(ckpt_path: str):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    cfg = ckpt["cfg"]
    name = cfg["model"]["name"].lower()
    num_classes = cfg["dataset"]["num_classes"]
    if name == "simple_cnn":
        mod = importlib.import_module("models.simple_cnn")
        model = mod.SimpleCNN(num_classes=num_classes)
    else:
        raise ValueError(f"Unknown model: {name}")
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, cfg

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True, type=str)
    p.add_argument("--config", type=str, help="Optional override config")
    args = p.parse_args()

    model, cfg = load_model_from_ckpt(args.checkpoint)
    if args.config:
        from utils.config import load_config, deep_update
        cfg = deep_update(cfg, load_config(args.config))

    _, val_loader, _ = create_dataloaders(cfg)

    device = torch.device(cfg["trainer"]["device"] if torch.cuda.is_available() else "cpu")
    model.to(device)

    correct = 0; total = 0
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs).argmax(dim=1)
            correct += (outputs == targets).sum().item()
            total += targets.size(0)
    acc = correct / max(1, total)
    print(json.dumps({"val_accuracy": acc, "total": total}, indent=2))

if __name__ == "__main__":
    main()
