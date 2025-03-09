import os
import zipfile
from tqdm import tqdm
from src.utils import log_message

# Define paths
COMPETITION_NAME = "stanford-rna-3d-folding"
DOWNLOAD_PATH = "data/raw"
ZIP_FILE = f"{DOWNLOAD_PATH}/{COMPETITION_NAME}.zip"

# Ensure the directory exists
os.makedirs(DOWNLOAD_PATH, exist_ok=True)

# Download the competition dataset with --quiet flag
log_message("🚀 Downloading dataset...", "INFO")
os.system(f"kaggle competitions download -c {COMPETITION_NAME} -p {DOWNLOAD_PATH} --quiet")

# Simulate a progress bar for better UX
for i in tqdm(range(100), desc="Downloading", ascii=" █", ncols=80):
    pass  # Just to show progress animation

# Unzip the dataset
log_message("📦 Unzipping dataset...", "INFO")
with zipfile.ZipFile(ZIP_FILE, "r") as zip_ref:
    zip_ref.extractall(DOWNLOAD_PATH)

# Remove the zip file after extraction
os.remove(ZIP_FILE)

log_message("✅ Dataset successfully downloaded and unzipped!\n", "DONE")
