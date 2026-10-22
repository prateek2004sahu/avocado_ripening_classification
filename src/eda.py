"""Exploratory data analysis: class balance, image sizes, lighting, colour (a*) per class."""
import json

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image

from src import config
from src.dataset import list_images, clean


def run_eda(show=False):
    df, clean_report = clean(list_images())
    classes = sorted(df["label"].unique())

    # 1) class distribution + imbalance ratio
    counts = df["label"].value_counts().reindex(classes)
    imbalance = float(counts.max() / counts.min())
    plt.figure(figsize=(7, 3.5))
    sns.barplot(x=counts.index, y=counts.values, color="forestgreen")
    plt.title("Class distribution"); plt.ylabel("Images"); plt.xlabel("Stage")
    for i, v in enumerate(counts.values):
        plt.text(i, v, str(v), ha="center", va="bottom")
    plt.tight_layout(); plt.savefig(config.RESULTS_DIR / "eda_class_distribution.png", dpi=150)
    plt.show() if show else plt.close()

    # 2) per-image statistics (sampled per class for speed)
    parts = [g.sample(min(len(g), 300), random_state=0) for _, g in df.groupby("label")]
    sample = pd.concat(parts)
    rows = []
    for p, lab in zip(sample["path"], sample["label"]):
        img = cv2.imread(p)
        if img is None:
            continue
        h, w = img.shape[:2]
        small = cv2.resize(img, (64, 64))
        lab_img = cv2.cvtColor(small, cv2.COLOR_BGR2LAB)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        border = np.concatenate([gray[:4].ravel(), gray[-4:].ravel(), gray[:, :4].ravel(), gray[:, -4:].ravel()])
        rows.append(dict(label=lab, width=w, height=h, brightness=gray.mean(),
                         contrast=gray.std(), a_mean=lab_img[..., 1].mean(),
                         L_mean=lab_img[..., 0].mean(), bg_brightness=border.mean(),
                         glare_frac=(gray > 245).mean()))
    stats = pd.DataFrame(rows)
    stats.to_csv(config.RESULTS_DIR / "eda_image_stats.csv", index=False)

    fig, ax = plt.subplots(1, 4, figsize=(18, 4))
    for a, col, t in zip(ax, ["a_mean", "brightness", "bg_brightness", "glare_frac"],
                         ["CIELAB a* mean (green->red)", "Mean brightness",
                          "Background brightness", "Glare fraction (pixels>245)"]):
        sns.boxplot(data=stats, x="label", y=col, order=classes, ax=a, color="lightgreen")
        a.set_title(t); a.set_xlabel("")
        a.tick_params(axis="x", rotation=30)
    plt.tight_layout(); plt.savefig(config.RESULTS_DIR / "eda_lighting_color.png", dpi=150)
    plt.show() if show else plt.close()

    # 3) outliers (IQR rule on brightness) and size variation
    q1, q3 = stats["brightness"].quantile([.25, .75])
    iqr = q3 - q1
    n_bright_out = int(((stats["brightness"] < q1 - 1.5 * iqr) | (stats["brightness"] > q3 + 1.5 * iqr)).sum())
    size_var = stats.groupby(["width", "height"]).size().sort_values(ascending=False)

    # 4) sample grid
    fig, ax = plt.subplots(len(classes), 5, figsize=(12, 2.4 * len(classes)))
    ax = np.atleast_2d(ax)
    for r, c in enumerate(classes):
        paths = df[df["label"] == c]["path"].sample(min(5, int(counts[c])), random_state=1).tolist()
        for k in range(5):
            ax[r, k].axis("off")
            if k < len(paths):
                ax[r, k].imshow(Image.open(paths[k]).convert("RGB"))
            if k == 0:
                ax[r, k].set_title(c, loc="left", fontsize=11)
    plt.tight_layout(); plt.savefig(config.RESULTS_DIR / "eda_samples.png", dpi=120)
    plt.show() if show else plt.close()

    g = stats.groupby("label")
    summary = dict(
        classes=classes, counts={k: int(v) for k, v in counts.items()},
        imbalance_ratio=round(imbalance, 2), cleaning=clean_report,
        n_unique_image_sizes=int(len(size_var)),
        most_common_size=[int(size_var.index[0][0]), int(size_var.index[0][1])],
        median_brightness=round(float(stats["brightness"].median()), 1),
        brightness_outliers_in_sample=n_bright_out, sampled_images=int(len(stats)),
        mean_glare_fraction=round(float(stats["glare_frac"].mean()), 4),
        a_star_mean_per_class={k: round(float(v), 2) for k, v in g["a_mean"].mean().items()},
        brightness_mean_per_class={k: round(float(v), 1) for k, v in g["brightness"].mean().items()},
        background_brightness_mean_per_class={k: round(float(v), 1) for k, v in g["bg_brightness"].mean().items()},
    )
    (config.RESULTS_DIR / "eda_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    run_eda()
