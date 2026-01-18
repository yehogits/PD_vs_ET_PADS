"""
Main Execution Pipeline
-----------------------
Orchestrates the PADS classification system.
"""

import tensorflow as tf
import sys

# Check for GPU
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"✅ GPU Detected: {gpus}")
else:
    print("⚠️ No GPU detected. Running on CPU.")
    

from src import dataset, fetch, exploration, modeling
from src.config import Paths
from src.logger import logger_inst
from src.utils import set_global_seed # <--- NEW IMPORT

def main():
    # 0. Reproducibility
    set_global_seed(42) # <--- FREEZE RANDOMNESS
    
    logger_inst.info("=== Starting PADS Analysis Pipeline ===")
    Paths.make_directories()
    
    # 1. Data Acquisition
    fetch.fetch_data()
    
    # 2. Processing
    dataset.run_processing_pipeline()
    
    # 3. Exploration
    exploration.run_exploration()
    
    # 4. Modeling & Audit
    modeling.run_modeling_pipeline()
    
    logger_inst.info("=== Pipeline Completed Successfully ===")

if __name__ == "__main__":
    main()