"""Single-image demo: classify one avocado image with BOTH approaches.

Usage:
    python main_demo.py path/to/avocado.jpg
    python main_demo.py            # picks a random test image automatically
"""
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from src import config
from src.classical_model import load_classical, predict_image as predict_classical
from src.dataset import load_splits


def run_avocado_demo(image_path, true_label=None):
    clf, cname, class_names = load_classical()
    lab1, conf1, _ = predict_classical(clf, image_path, class_names)
    lines = [f"Approach 1 ({cname}): {lab1}  (confidence {conf1:.2%})"]
    lab2 = None
    try:
        from src.dl_model import load_dl, predict_image as predict_dl
        dl, _ = load_dl()
        lab2, conf2, _ = predict_dl(dl, image_path, class_names)
        lines.append(f"Approach 2 (MobileNetV2): {lab2}  (confidence {conf2:.2%})")
    except Exception as e:  # keep the demo useful even if torch/model is missing
        lines.append(f"Approach 2 unavailable: {e}")

    print(f"--- Prediction results for {image_path} ---")
    if true_label:
        print("True label:", true_label)
    print("\n".join(lines))

    plt.figure(figsize=(5, 5.4))
    plt.imshow(Image.open(image_path).convert("RGB")); plt.axis("off")
    plt.title(("True: %s\n" % true_label if true_label else "") + "\n".join(lines), fontsize=8)
    plt.tight_layout()
    out = config.RESULTS_DIR / "demo_output.png"
    plt.savefig(out, dpi=150)
    print("Saved visual to", out)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        run_avocado_demo(sys.argv[1])
    else:
        df, _ = load_splits()
        row = df[df.split == "test"].sample(1, random_state=7).iloc[0]
        run_avocado_demo(row["path"], row["label"])
