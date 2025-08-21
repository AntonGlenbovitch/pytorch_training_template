# PyTorch Training Template (Image Classification)

A clean, modular starter to train and evaluate classification models on CIFAR-10 (default) or your own image dataset.

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python train.py --config configs/cifar10.yaml
# Evaluate best checkpoint
python evaluate.py --checkpoint checkpoints/best.pt --config configs/cifar10.yaml
```

## Project Layout

```
train.py                # Entry point to train a model
evaluate.py             # Evaluate a saved checkpoint
engine/trainer.py       # Training loop, metrics, early stopping, AMP, checkpoints
models/simple_cnn.py    # Example model
data/datasets.py        # CIFAR-10 + custom ImageFolder dataset loaders
utils/config.py         # YAML + CLI config loader/merger
configs/cifar10.yaml    # Default configuration
checkpoints/            # Saved models (created at runtime)
logs/                   # Training logs (loss/accuracy per epoch)
```

## Use with Your Own Data

1. Put your data in an `ImageFolder`-style directory:
   ```
   data_root/
     train/
       class_a/xxx.png
       class_b/yyy.png
     val/
       class_a/zzz.png
       class_b/www.png
   ```
2. Copy `configs/cifar10.yaml` to `configs/custom.yaml` and change:
   - `dataset.name: imagefolder`
   - `dataset.root: /path/to/data_root`
   - `dataset.num_classes` as appropriate

## Tips
- Turn on GPU: set `trainer.device: cuda` if available.
- Mixed precision (faster on GPUs) is enabled via `trainer.amp: true`.
- Resume training by pointing `trainer.resume` to a `.pt` checkpoint.
- Swap models by editing `model.name` and `model.kwargs`.
- Add your own models in `models/` and import in `train.py`.
