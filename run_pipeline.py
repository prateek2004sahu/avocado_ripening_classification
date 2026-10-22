"""Run the whole project end to end.

    python run_pipeline.py              # everything (download -> EDA -> train both -> evaluate -> demo -> report)
    python run_pipeline.py --skip-download   # if data/raw already has the images
    python run_pipeline.py --no-dl      # classical approach only (quick test without PyTorch)
"""
import argparse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-download", action="store_true")
    ap.add_argument("--no-dl", action="store_true", help="skip the deep learning approach")
    ap.add_argument("--head-epochs", type=int, default=15)
    ap.add_argument("--finetune-epochs", type=int, default=10)
    ap.add_argument("--workers", type=int, default=2, help="DataLoader workers (use 0 on Windows)")
    a = ap.parse_args()

    from src import dataset, eda, classical_model, evaluate, make_report
    if not a.skip_download:
        from src.download_data import download_and_extract
        download_and_extract()
    print("\n=== 1. EDA ===");                 eda.run_eda()
    print("\n=== 2. Cleaning + split ===");    dataset.prepare_splits()
    print("\n=== 3. Approach 1 (classical) ==="); classical_model.train_classical()
    if not a.no_dl:
        from src import dl_model
        print("\n=== 4. Approach 2 (MobileNetV2) ===")
        dl_model.train_dl(a.head_epochs, a.finetune_epochs, num_workers=a.workers)
    print("\n=== 5. Evaluation ===");          evaluate.evaluate_all(with_dl=not a.no_dl)
    print("\n=== 6. Demo ===")
    import main_demo
    from src.dataset import load_splits
    df, _ = load_splits()
    row = df[df.split == "test"].sample(1, random_state=7).iloc[0]
    main_demo.run_avocado_demo(row["path"], row["label"])
    print("\n=== 7. Report ===");              make_report.main()


if __name__ == "__main__":
    main()
