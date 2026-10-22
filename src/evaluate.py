"""Evaluate both approaches on the held-out TEST set and write comparison tables/plots."""
import json
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             precision_recall_fscore_support)

from src import config
from src.classical_model import extract_features, load_classical
from src.dataset import load_splits


def metrics_dict(y_true, y_pred, class_names):
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    _, _, fw, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)
    rep = classification_report(y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=range(len(class_names))).tolist()
    return dict(accuracy=accuracy_score(y_true, y_pred), precision_macro=p, recall_macro=r,
                f1_macro=f, f1_weighted=fw, per_class=rep, cm=cm)


def plot_cm(y_true, y_pred, class_names, title, path):
    cm = confusion_matrix(y_true, y_pred, labels=range(len(class_names)))
    cmn = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    plt.figure(figsize=(5.5, 4.5))
    sns.heatmap(cmn, annot=cm, fmt="d", cmap="Greens", xticklabels=class_names, yticklabels=class_names)
    plt.title(title); plt.xlabel("Predicted"); plt.ylabel("True")
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()


def evaluate_all(with_dl=True):
    df, class_names = load_splits()
    test = df[df.split == "test"].reset_index(drop=True)
    y = test["y"].values
    results, preds = {}, {}

    # ---- Approach 1: classical ----
    model, cname, _ = load_classical()
    X = np.load(config.PROC_DIR / "classical_features.npz")["X_test"]
    y1 = model.predict(X)
    t = time.time()
    for p in test["path"][:50]:                       # end-to-end latency: features + predict
        model.predict(extract_features(p).reshape(1, -1))
    lat1 = (time.time() - t) / min(50, len(test)) * 1000
    results["Approach 1: " + cname + " (HSV+LAB+LBP)"] = dict(
        **metrics_dict(y, y1, class_names), latency_ms_cpu=lat1,
        size_mb=os.path.getsize(config.CLASSICAL_MODEL_PATH) / 1e6)
    preds["classical"] = y1
    plot_cm(y, y1, class_names, "Approach 1 - " + cname, config.RESULTS_DIR / "cm_classical.png")

    # ---- Approach 2: MobileNetV2 ----
    if with_dl:
        import torch
        from src.dl_model import load_dl, predict_paths
        dl, _ = load_dl(torch.device("cpu"))
        proba = predict_paths(dl, test["path"].tolist(), device=torch.device("cpu"))
        y2 = proba.argmax(1)
        t = time.time()
        for p in test["path"][:50]:                   # batch-size-1 CPU latency
            predict_paths(dl, [p], device=torch.device("cpu"))
        lat2 = (time.time() - t) / min(50, len(test)) * 1000
        results["Approach 2: MobileNetV2 transfer learning"] = dict(
            **metrics_dict(y, y2, class_names), latency_ms_cpu=lat2,
            size_mb=os.path.getsize(config.DL_MODEL_PATH) / 1e6)
        preds["dl"] = y2
        plot_cm(y, y2, class_names, "Approach 2 - MobileNetV2", config.RESULTS_DIR / "cm_dl.png")
        _plot_errors(test, y, y2, class_names)

    rows = [{"Model": k, "Accuracy": v["accuracy"], "Precision (macro)": v["precision_macro"],
             "Recall (macro)": v["recall_macro"], "F1 (macro)": v["f1_macro"],
             "F1 (weighted)": v["f1_weighted"], "CPU latency (ms/img)": v["latency_ms_cpu"],
             "Model size (MB)": v["size_mb"]} for k, v in results.items()]
    comp = pd.DataFrame(rows)
    comp.to_csv(config.RESULTS_DIR / "comparison.csv", index=False)
    (config.RESULTS_DIR / "metrics.json").write_text(json.dumps(
        {"class_names": class_names, "n_test": int(len(test)), "results": results}, indent=2, default=float))
    print(comp.round(4).to_string(index=False))
    return comp, results


def _plot_errors(test, y, y_pred, class_names, n=8):
    wrong = np.where(y != y_pred)[0][:n]
    if len(wrong) == 0:
        return
    cols = min(4, len(wrong)); rows = int(np.ceil(len(wrong) / cols))
    fig, ax = plt.subplots(rows, cols, figsize=(3.2 * cols, 3.2 * rows))
    ax = np.atleast_1d(ax).ravel()
    for a in ax:
        a.axis("off")
    for a, i in zip(ax, wrong):
        a.imshow(Image.open(test["path"][i]).convert("RGB"))
        a.set_title(f"true: {class_names[y[i]]}\npred: {class_names[y_pred[i]]}", fontsize=9)
    plt.tight_layout(); plt.savefig(config.RESULTS_DIR / "dl_misclassified.png", dpi=130); plt.close()


if __name__ == "__main__":
    evaluate_all()
