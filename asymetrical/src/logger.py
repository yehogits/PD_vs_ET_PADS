"""
Logging Module
--------------
Sets up a console logger to track the pipeline's progress.
"""
import logging
import sys

# Initialize Logger
logger_inst = logging.getLogger("PADS_Project")
logger_inst.setLevel(logging.INFO)

# Console Handler
handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%H:%M:%S')
handler.setFormatter(formatter)

# Avoid duplicate handlers if re-imported
if not logger_inst.handlers:
    logger_inst.addHandler(handler)