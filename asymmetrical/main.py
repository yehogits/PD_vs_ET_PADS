import sys
import os

import matplotlib
matplotlib.use('Agg')

import pandas as pd
from sklearn.metrics import confusion_matrix

# Freeze Randomness
from src.utils import set_global_seed
set_global_seed(42)


current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from src.logger import logger_inst

# =================================================================
# 1. THE INTERCEPTOR
# =================================================================
import src.voting

original_voting = src.voting.run_patient_voting
intercepted_data = {}

def patched_voting(model, X_test, y_test, groups_test, paths):
    logger_inst.info("Intercepting Hybrid model data for comparison graph...")
    
    # Save the exact model object
    intercepted_data['model'] = model
    
    # Calculate the exact CM
    probs = model.predict(X_test, verbose=0).flatten()
    df = pd.DataFrame({'Subject': groups_test, 'True': y_test, 'Prob': probs})
    results = df.groupby('Subject').agg({'True': 'first', 'Prob': 'mean'}).reset_index()
    results['Pred'] = (results['Prob'] > 0.15).astype(int)
    cm = confusion_matrix(results['True'], results['Pred'])
    
    intercepted_data['cm'] = cm
    original_voting(model, X_test, y_test, groups_test, paths)

src.voting.run_patient_voting = patched_voting

# =================================================================
# 2. REST OF THE PIPELINE IMPORT
# =================================================================
try:
    try:
        from src.fetch import run_fetch_pipeline
    except ImportError:
        run_fetch_pipeline = None

    try:
        from src.dataset import run_processing_pipeline
    except ImportError:
        run_processing_pipeline = None

    try:
        from src.exploration import run_exploration
    except ImportError:
        try:
            from src.exploration import run_exploration_pipeline as run_exploration
        except ImportError:
            run_exploration = None

    from src.modeling import run_modeling_pipeline
    
    try:
        from src.audit import run_audit
    except ImportError:
        run_audit = None
        
    from src.newexperiments import run_all_new_experiments
    
except ImportError as e:
    print(f"\nCRITICAL IMPORT ERROR: {e}")
    sys.exit(1)


if __name__ == "__main__":
    logger_inst.info("=== 1. STARTING ORIGINAL PIPELINE ===")
    
    if run_fetch_pipeline:
        try: run_fetch_pipeline()
        except Exception as e: logger_inst.error(f"Fetch issue: {e}")

    if run_processing_pipeline:
        try: run_processing_pipeline()
        except Exception as e: logger_inst.error(f"Processing issue: {e}")

    if run_exploration:
        try: run_exploration()
        except Exception as e: logger_inst.error(f"Exploration issue: {e}")

    try:
        run_modeling_pipeline()
    except Exception as e:
        logger_inst.error(f"Modeling pipeline issue: {e}")

    if run_audit:
        try: run_audit()
        except Exception as e: logger_inst.error(f"Audit issue: {e}")


    logger_inst.info("=== 2. STARTING NEW PATIENT-LEVEL EXPERIMENTS ===")
    try:
        hybrid_model = intercepted_data.get('model')
        hybrid_cm = intercepted_data.get('cm')
        
        if hybrid_cm is None:
            logger_inst.error("CRITICAL: Interceptor failed to capture the Hybrid CM!")
            
        run_all_new_experiments(hybrid_model, hybrid_cm)
    except Exception as e:
        logger_inst.error(f"New experiments encountered an issue: {e}")
        
    logger_inst.info("All execution completed successfully.")