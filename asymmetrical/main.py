import sys
import os

from src.utils import set_global_seed

set_global_seed(42) # <--- FREEZE RANDOMNESS

# Ensure Python finds the 'src' folder
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

try:
    from src.logger import logger_inst
    # Import from your original pipeline
    import src.modeling
    from src.modeling import run_modeling_pipeline
    
    # Import the new experiments pipeline
    from src.newexperiments import run_all_new_experiments
except ImportError as e:
    print(f"\nCRITICAL IMPORT ERROR: {e}")
    sys.exit(1)

if __name__ == "__main__":
    logger_inst.info("=== 1. STARTING ORIGINAL PIPELINE ===")
    
    # --- The "Monkey Patch" Interceptor ---
    # Intercept run_patient_voting to extract the exact CM from the Hybrid model
    original_voting = src.modeling.run_patient_voting
    intercepted_data = {}

    def patched_voting(model, X_test, y_test, groups_test, paths):
        import pandas as pd
        from sklearn.metrics import confusion_matrix
        
        # Save the model 
        intercepted_data['model'] = model
        
        # Recreate the exact CM using your original 0.15 threshold
        probs = model.predict(X_test, verbose=0).flatten()
        df = pd.DataFrame({'Subject': groups_test, 'True': y_test, 'Prob': probs})
        results = df.groupby('Subject').agg({'True': 'first', 'Prob': 'mean'}).reset_index()
        results['Pred'] = (results['Prob'] > 0.15).astype(int)
        
        # Store the confusion matrix safely
        cm = confusion_matrix(results['True'], results['Pred'])
        intercepted_data['cm'] = cm
        
        # Run your original code normally so ALL original plots are generated
        original_voting(model, X_test, y_test, groups_test, paths)

    # Apply the patch to the original module
    src.modeling.run_patient_voting = patched_voting

    try:
        # Run the full original modeling pipeline
        run_modeling_pipeline()
    except Exception as e:
        logger_inst.error(f"Original pipeline encountered an issue: {e}")

    logger_inst.info("=== 2. STARTING NEW PATIENT-LEVEL EXPERIMENTS ===")
    try:
        # Pass the intercepted original model and exact confusion matrix to the new tests
        hybrid_model = intercepted_data.get('model')
        hybrid_cm = intercepted_data.get('cm')
        
        # Run the new models and include the Hybrid model in the final graph
        run_all_new_experiments(hybrid_model, hybrid_cm)
    except Exception as e:
        logger_inst.error(f"New experiments encountered an issue: {e}")
        
    logger_inst.info("All execution completed successfully.")