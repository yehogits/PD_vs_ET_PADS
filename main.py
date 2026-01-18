"""
Main Execution Pipeline
-----------------------
Orchestrates the PADS classification system.
"""

import sys
from src.config import Paths
from src.logger import logger_inst
from src.utils import set_global_seed 
from src import dataset, fetch, exploration

def main():

    # 0. Reproducibility
    set_global_seed(42) #FREEZE RANDOMNESS

    logger_inst.info("=== Starting PADS Analysis Pipeline ===")
    Paths.make_directories()    
    
    # 1. Data Acquisition
    fetch.fetch_data()
    
    # 2. Processing
    dataset.run_processing_pipeline()
    
    # 3. Exploration
    exploration.run_exploration()

    logger_inst.info("=== Pipeline Completed Successfully ===")

if __name__ == "__main__":
    main()
