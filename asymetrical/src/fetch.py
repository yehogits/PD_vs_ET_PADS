"""
Data Fetching Module
--------------------
Automates the download of the PADS dataset from PhysioNet.
"""
import zipfile
import urllib.request
from src.config import Paths
from src.logger import logger_inst

DATA_URL = "https://physionet.org/static/published-projects/parkinsons-disease-smartwatch/parkinsons-disease-smartwatch-1.0.0.zip"

def fetch_data() -> None:
    paths = Paths.from_here()
    zip_target = paths.data_raw / "dataset.zip"

    # 1. Check if data is already extracted
    if any(paths.data_raw.glob("**/movement")):
        logger_inst.info("Stage 1: Data already present. Skipping download.")
        return

    # 2. Download if missing
    if not zip_target.exists():
        logger_inst.info("Stage 1: Downloading dataset (approx 735MB)...")
        opener = urllib.request.build_opener()
        opener.addheaders = [('User-agent', 'Mozilla/5.0')]
        urllib.request.install_opener(opener)
        urllib.request.urlretrieve(DATA_URL, zip_target)
        logger_inst.info("Download complete.")

    # 3. Extract
    logger_inst.info("Extracting archive...")
    try:
        with zipfile.ZipFile(zip_target, 'r') as zip_ref:
            zip_ref.extractall(paths.data_raw)
        logger_inst.info("Extraction successful.")
    except zipfile.BadZipFile:
        logger_inst.error("Downloaded ZIP is corrupt. Please delete it and try again.")