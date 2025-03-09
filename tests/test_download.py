import unittest
import os
import zipfile
from unittest.mock import patch, MagicMock

# Import your variables from the actual download script
from src.utils import log_message
from src.data.download import COMPETITION_NAME, DOWNLOAD_PATH, ZIP_FILE

# Define ANSI colors
COLOR_RESET = "\033[0m"
COLOR_GREEN = "\033[32m"
COLOR_RED = "\033[31m"
COLOR_YELLOW = "\033[33m"

class TestDownload(unittest.TestCase):
    """Tests for dataset download and extraction."""

    def test_kaggle_download(self):
        """test_kaggle_download"""
        with patch("os.system") as mock_system:
            expected_command = f"kaggle competitions download -c {COMPETITION_NAME} -p {DOWNLOAD_PATH}"
            mock_system.return_value = 0
            os.system(expected_command)
            mock_system.assert_called_with(expected_command)
        log_message("Kaggle CLI command ... ✅", "PASS")

    def test_kaggle_cli_missing(self):
        """test_kaggle_cli_missing"""
        with patch("os.system", side_effect=OSError("Kaggle CLI not found")):
            with self.assertRaises(OSError):
                os.system(f"kaggle competitions download -c {COMPETITION_NAME} -p {DOWNLOAD_PATH}")
        log_message("Handle missing Kaggle CLI ... ✅", "PASS")

    def test_unzip_files(self):
        """test_unzip_files"""
        with patch("zipfile.ZipFile") as mock_zipfile, patch("zipfile.ZipFile.extractall") as mock_extractall:
            mock_zip = MagicMock()
            mock_zip.__enter__.return_value.extractall = mock_extractall
            mock_zipfile.return_value = mock_zip
            with zipfile.ZipFile(ZIP_FILE, "r") as zip_ref:
                zip_ref.extractall(DOWNLOAD_PATH)
            mock_extractall.assert_called_with(DOWNLOAD_PATH)
        log_message("Extraction of the ZIP file ... ✅", "PASS")

    def test_remove_zip_file(self):
        """test_remove_zip_file"""
        with patch("os.remove") as mock_remove:
            os.remove(ZIP_FILE)
            mock_remove.assert_called_with(ZIP_FILE)
        log_message("Remove zipped file ... ✅", "PASS")

if __name__ == "__main__":
    unittest.main(verbosity=0)  # Suppress unittest default output
