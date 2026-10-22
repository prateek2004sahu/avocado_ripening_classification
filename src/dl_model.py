"""Approach 2: MobileNetV2 transfer learning (PyTorch)."""
import copy
import json
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models

from src import config
from src.dataset import get_transforms, load_splits, make_torch_dataset


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_model(num_classes, pretrained=True):
    """MobileNetV2 with frozen backbone and a new 2-layer classification head."""
    weights = models.MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.mobilenet_v2(weights=weights)
    for p in model.parameters():          # freeze everything first
        p.requires_grad = False
    in_f = model.classifier[1].in_features
    model.classifier[1] = nn.Sequential(  # new head (trainable)
        nn.Linear(in_f, 128), nn.ReLU(), nn.Dropout(0.2), nn.Linear(128, num_classes))
    return model


def _run_epoch(model, loader, criterion, device, optimizer=None):
    train = optimizer is not None
    model.train(train)
    total_loss, correct, n = 0.0, 0, 0
    with torch.set_grad_enabled(train):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = criterion(out, y)
            if train:
                optimizer.zero_grad(); loss.backward(); optimizer.step()
            total_loss += loss.item() * len(y)
            correct += (out.argmax(1) == y).sum().item()
            n += len(y)
    return total_loss / n, correct / n


def _fit(model, train_dl, val_dl, criterion, optimizer, device, max_epochs, patience, tag, history):
    """Train with early stopping on validation loss; restores the best weights."""
    best_loss, best_w, bad = float("inf"), copy.deepcopy(model.state_dict()), 0
    for ep in range(1, max_epochs + 1):
        tl, ta = _run_epoch(model, train_dl, criterion, device, optimizer)
        vl, va = _run_epoch(model, val_dl, criterion, device)
        history.append(dict(phase=tag, epoch=ep, train_loss=tl, train_acc=ta, val_loss=vl, val_acc=va))
        print(f"[{tag}] epoch {ep:02d} | train loss {tl:.4f} acc {ta:.3f} | val loss {vl:.4f} acc {va:.3f}")
        if vl < best_loss - 1e-4:
            best_loss, best_w, bad = vl, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= patience:
                print(f"[{tag}] early stopping at epoch {ep}")
                break
    model.load_state_dict(best_w)


def train_dl(head_epochs=15, finetune_epochs=10, batch_size=32, num_workers=2, pretrained=True):
    torch.manual_seed(config.SEED); np.random.seed(config.SEED)
    df, class_names = load_splits()
    device = get_device()
    print("Device:", device)

    tr, va = df[df.split == "train"], df[df.split == "val"]
    train_dl_ = DataLoader(make_torch_dataset(tr, get_transforms(True)), batch_size=batch_size,
                           shuffle=True, num_workers=num_workers)
    val_dl_ = DataLoader(make_torch_dataset(va, get_transforms(False)), batch_size=batch_size,
                         num_workers=num_workers)

    # class-weighted loss handles imbalance
    counts = np.bincount(tr["y"], minlength=len(class_names)).astype(float)
    weights = torch.tensor(counts.sum() / (len(class_names) * np.maximum(counts, 1)), dtype=torch.float32)
    criterion = nn.CrossEntropyLoss(weight=weights.to(device))

    model = build_model(len(class_names), pretrained).to(device)
    history = []
    t0 = time.time()

    # Phase 1: train only the new head (backbone frozen)
    opt = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)
    _fit(model, train_dl_, val_dl_, criterion, opt, device, head_epochs, 4, "head", history)

    # Phase 2: unfreeze the last blocks and fine-tune with a small learning rate
    if finetune_epochs > 0:
        for p in model.features[14:].parameters():
            p.requires_grad = True
        opt = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
        _fit(model, train_dl_, val_dl_, criterion, opt, device, finetune_epochs, 3, "finetune", history)

    train_time = time.time() - t0
    torch.save(model.state_dict(), config.DL_MODEL_PATH)
    hist = pd.DataFrame(history)
    hist.to_csv(config.RESULTS_DIR / "dl_history.csv", index=False)

    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    x = np.arange(len(hist))
    ax[0].plot(x, hist.train_loss, label="train"); ax[0].plot(x, hist.val_loss, label="val"); ax[0].set_title("Loss")
    ax[1].plot(x, hist.train_acc, label="train"); ax[1].plot(x, hist.val_acc, label="val"); ax[1].set_title("Accuracy")
    for a in ax:
        a.set_xlabel("epoch (head then fine-tune)"); a.legend()
    plt.tight_layout(); plt.savefig(config.RESULTS_DIR / "dl_training_curves.png", dpi=150); plt.close()
    (config.RESULTS_DIR / "dl_training.json").write_text(json.dumps(
        {"train_time_sec": round(train_time, 1), "device": str(device), "epochs_run": len(hist)}, indent=2))
    print("Saved", config.DL_MODEL_PATH)


def load_dl(device=None):
    device = device or get_device()
    class_names = json.loads(config.CLASS_NAMES_JSON.read_text())
    model = build_model(len(class_names), pretrained=False)
    model.load_state_dict(torch.load(config.DL_MODEL_PATH, map_location=device))
    return model.to(device).eval(), class_names


@torch.no_grad()
def predict_paths(model, paths, device=None, batch_size=64):
    """Return (N, num_classes) softmax probabilities for a list of image paths."""
    from PIL import Image
    device = device or next(model.parameters()).device
    tf = get_transforms(False)
    out = []
    for i in range(0, len(paths), batch_size):
        batch = torch.stack([tf(Image.open(p).convert("RGB")) for p in paths[i:i + batch_size]]).to(device)
        out.append(torch.softmax(model(batch), 1).cpu().numpy())
    return np.vstack(out)


def predict_image(model, image_path, class_names):
    proba = predict_paths(model, [image_path])[0]
    idx = int(np.argmax(proba))
    return class_names[idx], float(proba[idx]), proba


if __name__ == "__main__":
    train_dl()
