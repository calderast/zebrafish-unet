"""Download the C. elegans 3D nuclei dataset (Zenodo 5942575, 84 MB).

Usage:
    python download_data.py
"""

import shutil
import urllib.request
import zipfile
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"
URL = "https://zenodo.org/records/5942575/files/c_elegans_nuclei.zip?download=1"
DEST = DATA_DIR / "c_elegans"
CHECK = DEST / "c_elegans_nuclei" / "train" / "images"

if __name__ == "__main__":
    if CHECK.exists():
        print("Already downloaded, skipping.")
    else:
        DEST.mkdir(parents=True, exist_ok=True)
        zip_path = DEST / "c_elegans_nuclei.zip"

        print("Downloading C. elegans nuclei dataset (84 MB)...")
        urllib.request.urlretrieve(URL, str(zip_path))

        print("Extracting...")
        with zipfile.ZipFile(str(zip_path), "r") as z:
            z.extractall(str(DEST))
        zip_path.unlink()

        # Clean up macOS metadata if present
        macosx_dir = DEST / "__MACOSX"
        if macosx_dir.exists():
            shutil.rmtree(macosx_dir)

    print(f"Data is in: {CHECK.resolve()}")
