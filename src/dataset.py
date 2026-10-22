"""Dataset discovery, cleaning, stratified splitting (70/15/15) and PyTorch Dataset."""
import hashlib
import json
from pathlib import Path

import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split

from src import config

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
SPLIT_NAMES = {"train", "val", "valid", "validation", "test", "testing"}


def list_images(raw_dir=config.RAW_DIR) -> pd.DataFrame:
    """Walk raw_dir and return a DataFrame(path, label).
    Label = name of the parent folder (if the parent is 'train'/'val'/'test'
    the grandparent folder is used instead). Works for any folder layout."""
    rows = []
    for p in Path(raw_dir).rglob("*"):
        if p.suffix.lower() in IMG_EXT and "__MACOSX" not in p.parts:
            label = p.parent.name
            if label.lower() in SPLIT_NAMES:
                label = p.parent.parent.name
            rows.append((str(p), label))
    if not rows:
        raise FileNotFoundError(f"No images found in {raw_dir}. Run the download step first.")
    return pd.DataFrame(rows, columns=["path", "label"])


def _md5(path):
    return hashlib.md5(Path(path).read_bytes()).hexdigest()


def clean(df: pd.DataFrame):
    """Drop unreadable images and exact duplicates (prevents train/test leakage)."""
    report = {"total_found": int(len(df))}
    good = []
    for p in df["path"]:
        try:
            with Image.open(p) as im:
                im.verify()
            good.append(True)
        except Exception:
            good.append(False)
    df = df[good].copy()
    report["corrupted_removed"] = int(len(good) - sum(good))
    df["md5"] = df["path"].map(_md5)
    before = len(df)
    df = df.drop_duplicates(subset="md5").drop(columns="md5")
    report["duplicates_removed"] = int(before - len(df))
    report["after_cleaning"] = int(len(df))
    return df.reset_index(drop=True), report


def prepare_splits(raw_dir=config.RAW_DIR, seed=config.SEED):
    """Clean data, make a stratified 70/15/15 split, save data/processed/splits.csv."""
    df, report = clean(list_images(raw_dir))
    class_names = sorted(df["label"].unique())
    df["y"] = df["label"].map({c: i for i, c in enumerate(class_names)})

    train, rest = train_test_split(
        df, test_size=config.VAL_FRAC + config.TEST_FRAC, stratify=df["y"], random_state=seed
    )
    val, test = train_test_split(
        rest, test_size=config.TEST_FRAC / (config.VAL_FRAC + config.TEST_FRAC),
        stratify=rest["y"], random_state=seed,
    )
    out = pd.concat([train.assign(split="train"), val.assign(split="val"),
                     test.assign(split="test")]).reset_index(drop=True)
    out.to_csv(config.SPLITS_CSV, index=False)
    config.CLASS_NAMES_JSON.write_text(json.dumps(class_names))
    (config.RESULTS_DIR / "cleaning_report.json").write_text(json.dumps(report, indent=2))
    print("Cleaning:", report)
    print(out.groupby(["split", "label"]).size().unstack(fill_value=0))
    return out, class_names


def load_splits():
    df = pd.read_csv(config.SPLITS_CSV)
    class_names = json.loads(config.CLASS_NAMES_JSON.read_text())
    return df, class_names


# ----------------------------- PyTorch part -----------------------------
def get_transforms(train: bool):
    from torchvision import transforms as T
    norm = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # ImageNet stats
    if train:
        return T.Compose([
            T.RandomResizedCrop(config.IMG_SIZE, scale=(0.7, 1.0)),   # spatial shifts / zoom
            T.RandomHorizontalFlip(),
            T.RandomRotation(20),
            # brightness/contrast only. Hue jitter is deliberately OFF because hue IS the ripeness signal.
            T.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.1),
            T.ToTensor(), norm,
        ])
    return T.Compose([T.Resize((config.IMG_SIZE, config.IMG_SIZE)), T.ToTensor(), norm])


def make_torch_dataset(df, transform):
    """Return a torch Dataset yielding (image_tensor, label)."""
    from torch.utils.data import Dataset

    class AvocadoDataset(Dataset):
        def __init__(self, df, transform):
            self.paths, self.y, self.transform = df["path"].tolist(), df["y"].tolist(), transform

        def __len__(self):
            return len(self.paths)

        def __getitem__(self, i):
            img = Image.open(self.paths[i]).convert("RGB")
            return self.transform(img), int(self.y[i])

    return AvocadoDataset(df, transform)
