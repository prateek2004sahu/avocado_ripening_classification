"""Download and unzip the dataset from Google Drive into data/raw/."""
import tarfile
import zipfile
from pathlib import Path

from src.config import DRIVE_FILE_ID, RAW_DIR


def download_and_extract():
    import gdown  # imported here so other modules work without it

    archive = RAW_DIR / "dataset_archive"
    url = f"https://drive.google.com/uc?id={DRIVE_FILE_ID}"
    print("Downloading dataset ...")
    out = gdown.download(url, str(archive), quiet=False)
    if out is None:
        raise RuntimeError(
            "Download failed (Drive quota?). Download manually in your browser, "
            "then unzip into data/raw/ and skip this step."
        )
    archive = Path(out)
    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as z:
            z.extractall(RAW_DIR)
    elif tarfile.is_tarfile(archive):
        with tarfile.open(archive) as t:
            t.extractall(RAW_DIR)
    else:
        print("File is not an archive; leaving it as is.")
    archive.unlink(missing_ok=True)
    print("Dataset ready in", RAW_DIR)


if __name__ == "__main__":
    download_and_extract()
