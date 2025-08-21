from typing import Dict, Any, Tuple
import os, math, json
import torch
from torch import nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import OneCycleLR
from tqdm import tqdm

class Trainer:
    def __init__(self, model: nn.Module, cfg: Dict[str, Any]):
        self.cfg = cfg
        self.device = torch.device(cfg["trainer"]["device"] if torch.cuda.is_available() or "cpu" in cfg["trainer"]["device"] else "cpu")
        self.model = model.to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=cfg["optimizer"]["lr"], weight_decay=cfg["optimizer"]["weight_decay"])
        self.scaler = GradScaler(enabled=cfg["trainer"]["amp"])
        self.scheduler = None
        self.best_acc = 0.0
        self.ckpt_dir = cfg["trainer"]["checkpoint_dir"]
        os.makedirs(self.ckpt_dir, exist_ok=True)

    def _maybe_build_scheduler(self, train_loader_len: int):
        if self.cfg["scheduler"]["one_cycle"]:
            steps_per_epoch = max(1, train_loader_len)
            total_steps = steps_per_epoch * self.cfg["trainer"]["epochs"]
            self.scheduler = OneCycleLR(
                self.optimizer,
                max_lr=self.cfg["optimizer"]["lr"],
                total_steps=total_steps,
                pct_start=self.cfg["scheduler"]["pct_start"],
                div_factor=self.cfg["scheduler"]["div_factor"],
                final_div_factor=self.cfg["scheduler"]["final_div_factor"],
                anneal_strategy="cos"
            )

    def save(self, path: str, epoch: int, acc: float):
        torch.save({
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "epoch": epoch,
            "best_acc": self.best_acc,
            "cfg": self.cfg,
        }, path)

    def load(self, path: str) -> int:
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state"])
        self.optimizer.load_state_dict(ckpt["optimizer_state"])
        self.best_acc = ckpt.get("best_acc", 0.0)
        return ckpt.get("epoch", 0)

    def step(self, batch, train: bool):
        inputs, targets = batch
        inputs, targets = inputs.to(self.device), targets.to(self.device)
        with autocast(enabled=self.cfg["trainer"]["amp"]):
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
        if train:
            self.optimizer.zero_grad(set_to_none=True)
            self.scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg["trainer"]["max_grad_norm"])
            self.scaler.step(self.optimizer)
            self.scaler.update()
            if self.scheduler:
                self.scheduler.step()
        preds = outputs.argmax(dim=1)
        correct = (preds == targets).sum().item()
        total = targets.size(0)
        return loss.item(), correct, total

    def fit(self, train_loader, val_loader):
        start_epoch = 0
        resume_path = self.cfg["trainer"].get("resume")
        if resume_path:
            try:
                start_epoch = self.load(resume_path) + 1
                print(f"Resumed from {resume_path} at epoch {start_epoch}")
            except Exception as e:
                print(f"Resume failed: {e}")

        self._maybe_build_scheduler(len(train_loader))

        patience = self.cfg["trainer"]["early_stopping_patience"]
        wait = 0

        history = {"train_loss": [], "val_acc": []}

        for epoch in range(start_epoch, self.cfg["trainer"]["epochs"]):
            self.model.train()
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.cfg['trainer']['epochs']} (train)")
            running_loss = 0.0
            for batch in pbar:
                loss, correct, total = self.step(batch, train=True)
                running_loss += loss
                pbar.set_postfix(loss=loss)

            avg_loss = running_loss / max(1, len(train_loader))
            history["train_loss"].append(avg_loss)

            # Validation
            self.model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for batch in tqdm(val_loader, desc=f"Epoch {epoch+1} (val)"):
                    loss, c, t = self.step(batch, train=False)
                    correct += c; total += t
            acc = correct / max(1, total)
            history["val_acc"].append(acc)

            # Save last
            last_path = os.path.join(self.ckpt_dir, "last.pt")
            self.save(last_path, epoch, acc)

            # Save best
            if acc > self.best_acc:
                self.best_acc = acc
                best_path = os.path.join(self.ckpt_dir, "best.pt")
                self.save(best_path, epoch, acc)
                wait = 0
                improved = True
            else:
                wait += 1
                improved = False

            print(f"Epoch {epoch+1}: train_loss={avg_loss:.4f} val_acc={acc:.4f} {'*' if improved else ''} (best={self.best_acc:.4f})")

            # Early stopping
            if wait >= patience:
                print(f"Early stopping: no improvement for {patience} epochs.")
                break

        # Write logs
        with open(os.path.join(self.cfg['trainer']['log_dir'], 'history.json'), 'w') as f:
            json.dump(history, f, indent=2)
        return self.best_acc
