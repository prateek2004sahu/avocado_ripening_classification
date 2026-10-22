"""Central configuration: paths and constants used by every script."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROC_DIR = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"
REPORT_DIR = ROOT / "report"

SPLITS_CSV = PROC_DIR / "splits.csv"
CLASSICAL_MODEL_PATH = MODELS_DIR / "classical_model.joblib"
DL_MODEL_PATH = MODELS_DIR / "mobilenetv2_best.pth"
CLASS_NAMES_JSON = MODELS_DIR / "class_names.json"

DRIVE_FILE_ID = "1NMJ0FOxcWH1mwFFeERuY6fgEY-MVgeET"   # dataset on Google Drive

SEED = 42
IMG_SIZE = 224          # deep learning input size
FEAT_SIZE = 128         # classical feature extraction size
TRAIN_FRAC, VAL_FRAC, TEST_FRAC = 0.70, 0.15, 0.15

for _d in (RAW_DIR, PROC_DIR, MODELS_DIR, RESULTS_DIR, REPORT_DIR):
    _d.mkdir(parents=True, exist_ok=True)
