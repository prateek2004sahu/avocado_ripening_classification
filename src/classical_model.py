"""Approach 1: hand-crafted colour (HSV + CIELAB) and texture (LBP) features + SVM / Random Forest."""
import json
import time

import cv2
import joblib
import numpy as np
from joblib import Parallel, delayed
from skimage.feature import local_binary_pattern
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src import config
from src.dataset import load_splits


def extract_features(image_path):
    """One image -> 1-D feature vector (colour histograms + colour stats + LBP texture)."""
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Cannot read {image_path}")
    img = cv2.resize(img, (config.FEAT_SIZE, config.FEAT_SIZE))

    feats = []
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)    # a* channel: green <-> red/purple axis
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # normalised histograms (so image brightness/size does not scale the vector)
    for arr, ch, bins, rng in [(lab, 0, 16, 256), (lab, 1, 16, 256), (lab, 2, 16, 256),
                               (hsv, 0, 18, 180), (hsv, 1, 8, 256), (hsv, 2, 8, 256)]:
        h = cv2.calcHist([arr], [ch], None, [bins], [0, rng]).flatten()
        feats.append(h / (h.sum() + 1e-8))
    # mean / std per channel for both colour spaces
    for arr in (lab, hsv):
        feats.append(arr.reshape(-1, 3).mean(0) / 255.0)
        feats.append(arr.reshape(-1, 3).std(0) / 255.0)

    # texture: uniform LBP (smooth underripe skin vs pebbled/wrinkled ripe skin)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lbp = local_binary_pattern(gray, P=8, R=1, method="uniform")
    h, _ = np.histogram(lbp.ravel(), bins=np.arange(0, 11), range=(0, 10))
    feats.append(h / (h.sum() + 1e-8))
    return np.concatenate(feats).astype(np.float32)


def build_feature_matrix(paths, n_jobs=-1):
    return np.vstack(Parallel(n_jobs=n_jobs)(delayed(extract_features)(p) for p in paths))


def train_classical():
    df, class_names = load_splits()
    data = {}
    for s in ("train", "val", "test"):
        part = df[df["split"] == s]
        print(f"Extracting features for {s} ({len(part)} images) ...")
        data[s] = (build_feature_matrix(part["path"]), part["y"].values)
    np.savez_compressed(config.PROC_DIR / "classical_features.npz",
                        **{f"X_{k}": v[0] for k, v in data.items()}, **{f"y_{k}": v[1] for k, v in data.items()})
    Xtr, ytr = data["train"]; Xva, yva = data["val"]

    # --- SVM: tune C and gamma with 3-fold CV on the TRAIN set only ---
    svm_grid = GridSearchCV(
        make_pipeline(StandardScaler(), SVC(kernel="rbf", class_weight="balanced")),
        {"svc__C": [1, 10, 100], "svc__gamma": ["scale", 0.01, 0.001]},
        scoring="f1_macro", cv=3, n_jobs=-1)
    svm_grid.fit(Xtr, ytr)
    svm_val = f1_score(yva, svm_grid.predict(Xva), average="macro")
    print("SVM best params:", svm_grid.best_params_, "| val macro-F1 = %.4f" % svm_val)

    # --- Random Forest baseline ---
    rf = RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=config.SEED, n_jobs=-1)
    rf.fit(Xtr, ytr)
    rf_val = f1_score(yva, rf.predict(Xva), average="macro")
    print("Random Forest val macro-F1 = %.4f" % rf_val)

    # pick winner on VALIDATION macro-F1 (test set stays untouched until final evaluation)
    if svm_val >= rf_val:
        p = svm_grid.best_params_
        final = make_pipeline(StandardScaler(),
                              SVC(kernel="rbf", C=p["svc__C"], gamma=p["svc__gamma"],
                                  class_weight="balanced", probability=True, random_state=config.SEED))
        name = "SVM (RBF)"
    else:
        final, name = rf, "Random Forest"
    final.fit(Xtr, ytr)
    joblib.dump({"model": final, "name": name, "class_names": class_names}, config.CLASSICAL_MODEL_PATH)
    info = {"selected": name, "svm_val_f1": svm_val, "rf_val_f1": rf_val,
            "svm_best_params": {k: str(v) for k, v in svm_grid.best_params_.items()}}
    (config.RESULTS_DIR / "classical_training.json").write_text(json.dumps(info, indent=2))
    print("Saved", config.CLASSICAL_MODEL_PATH, "->", name)
    return info


def load_classical():
    bundle = joblib.load(config.CLASSICAL_MODEL_PATH)
    return bundle["model"], bundle["name"], bundle["class_names"]


def predict_image(model, image_path, class_names):
    x = extract_features(image_path).reshape(1, -1)
    proba = model.predict_proba(x)[0]
    idx = int(np.argmax(proba))
    return class_names[idx], float(proba[idx]), proba


if __name__ == "__main__":
    train_classical()
