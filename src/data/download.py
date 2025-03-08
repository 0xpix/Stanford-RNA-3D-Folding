import os
import zipfile
from src.utils import log_message

# Define paths
COMPETITION_NAME = "stanford-rna-3d-folding"
DOWNLOAD_PATH = "data/raw"
ZIP_FILE = f"{DOWNLOAD_PATH}/{COMPETITION_NAME}.zip"

# Ensure the directory exists
os.makedirs(DOWNLOAD_PATH, exist_ok=True)

# Download the competition dataset
log_message("Downloading dataset")
os.system(f"kaggle competitions download -c {COMPETITION_NAME} -p {DOWNLOAD_PATH}")

# Unzip the dataset
log_message("Unzipping dataset")
with zipfile.ZipFile(ZIP_FILE, "r") as zip_ref:
    zip_ref.extractall(DOWNLOAD_PATH)

# Remove the zip file after extraction
os.remove(ZIP_FILE)

log_message("Dataset successfully downloaded and unzipped!\n", "DONE")
