# Avocado Ripening Stage Detection & Classification (Greenpod Labs - Assignment 2)

Two approaches are implemented and compared on the same stratified 70/15/15 split:

| | Approach 1 - Classical ML | Approach 2 - Deep learning |
|---|---|---|
| Features | HSV + CIELAB colour histograms/stats + LBP texture | learned (MobileNetV2, ImageNet weights) |
| Classifier | RBF-SVM (CV-tuned) vs Random Forest | new 2-layer head, last blocks fine-tuned |
| Imbalance | `class_weight="balanced"` | class-weighted cross-entropy |

Metrics: accuracy, macro precision, macro recall, macro F1 (main selection metric), confusion matrices, CPU latency, model size.

## Folder layout
```
avocado_ripening_classification/
├── run_pipeline.py          # one command runs everything
├── main_demo.py             # classify one image with BOTH models
├── Avocado_Colab_Full_Run.ipynb
├── requirements.txt
├── notebooks/               # 01 EDA, 02 classical, 03 deep learning, 04 evaluation + demo
├── src/                     # config, dataset, eda, classical_model, dl_model, evaluate, make_report
├── data/raw/                # put / download the dataset here
├── data/processed/          # splits.csv + cached features (auto)
├── models/                  # saved models (auto)
├── results/                 # plots, metrics.json, comparison.csv (auto)
└── report/                  # Assignment2_Avocado_Report.docx / .md (auto)
```

## A-to-Z: run on Google Colab (recommended, free GPU)
1. create a **public** GitHub repo named `avocado_ripening_classification`, upload all files (see "GitHub" below).
2. Open https://colab.research.google.com -> File -> Upload notebook -> `Avocado_Colab_Full_Run.ipynb`.
3. Runtime -> Change runtime type -> **T4 GPU**.
4. In the first cell make sure `REPO_URL` is your repo URL. Run all cells (about 10-25 min depending on dataset size).
5. When it finishes, `outputs.zip` downloads: it contains `results/`, `report/` and `models/`. Copy the report into your repo.
6. Share -> **Anyone with the link can view** -> copy the Colab link for submission.

## A-to-Z: run on your own PC
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate      Mac/Linux: source .venv/bin/activate
pip install -r requirements.txt
python run_pipeline.py --workers 0          # downloads data, trains both, evaluates, demo, report
```
Useful options: `--skip-download` (images already in `data/raw/`), `--no-dl` (classical only), `--head-epochs 5 --finetune-epochs 3` (quicker CPU run).
If the Google Drive download is blocked, download the file in your browser, unzip it into `data/raw/` and add `--skip-download`.

Step by step instead: open notebooks 01 -> 04 in order, or run
`python -m src.eda`, `python -c "from src.dataset import prepare_splits; prepare_splits()"`, `python -m src.classical_model`, `python -m src.dl_model`, `python -m src.evaluate`, `python main_demo.py <image>`, `python -m src.make_report`.

The folder names inside the dataset are used as class names automatically (any layout: `class/img.jpg` or `train/class/img.jpg`).

## Single image demo
```bash
python main_demo.py path/to/avocado.jpg
```

## GitHub
```bash
git init
git add .
git commit -m "Avocado ripening classification"
git branch -M main
git remote add origin https://github.com/prateek2004sahu/avocado_ripening_classification.git
git push -u origin main
```

## Submission checklist
- [ ] Public GitHub repo link (this project, with README + requirements.txt)
- [ ] Colab link, "Anyone with the link can view"
- [ ] `report/Assignment2_Avocado_Report.docx` (export to PDF if required) - open it, check the text matches what you observed, add your name
- [ ] Example run (`results/demo_output.png`)
- [ ] 30-minute screen recording: code structure -> EDA -> training -> demo -> results
- [ ] Answers to the 4 mandatory questions (already in the report, section 5 - edit Q4 to reflect your real time split)
