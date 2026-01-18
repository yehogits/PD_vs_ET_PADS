"""
Utilities Module
----------------
Helper functions for reproducibility and system configuration.
"""
import os
import random
import numpy as np
import tensorflow as tf

def set_global_seed(seed=42):
    """
    Freezes the random state of the universe.
    Ensures that training results are reproducible for the report.
    """
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    
    # Force TensorFlow to use deterministic ops (may slow down slightly)
    tf.config.experimental.enable_op_determinism()
    print(f"✅ Global Seed set to {seed}. Results will be deterministic.")