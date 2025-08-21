import argparse, yaml, copy
from typing import Any, Dict

def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg

def deep_update(d: dict, u: dict) -> dict:
    out = copy.deepcopy(d)
    for k, v in (u or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_update(out.get(k, {}), v)
        else:
            out[k] = v
    return out

def parse_args_and_merge(cfg: dict) -> dict:
    p = argparse.ArgumentParser(description="Training Configuration")
    p.add_argument("--config", type=str, required=False, help="Path to YAML config")
    p.add_argument("--device", type=str, help="cpu/cuda/mps")
    p.add_argument("--epochs", type=int)
    p.add_argument("--batch_size", type=int)
    p.add_argument("--lr", type=float)
    p.add_argument("--model", type=str, help="Model name override")
    p.add_argument("--data_root", type=str, help="Dataset root override")
    p.add_argument("--resume", type=str, help="Path to checkpoint to resume from")
    cli = vars(p.parse_args())
    # filter None
    cli = {k:v for k,v in cli.items() if v is not None}
    if cli.get("config"):
        cfg = deep_update(cfg, load_config(cli["config"]))
    # map common flat overrides
    if "device" in cli: cfg["trainer"]["device"] = cli["device"]
    if "epochs" in cli: cfg["trainer"]["epochs"] = cli["epochs"]
    if "batch_size" in cli: cfg["data"]["batch_size"] = cli["batch_size"]
    if "lr" in cli: cfg["optimizer"]["lr"] = cli["lr"]
    if "model" in cli: cfg["model"]["name"] = cli["model"]
    if "data_root" in cli: cfg["dataset"]["root"] = cli["data_root"]
    if "resume" in cli: cfg["trainer"]["resume"] = cli["resume"]
    return cfg
