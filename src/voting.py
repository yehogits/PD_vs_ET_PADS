"""
Voting Module
-------------
Aggregates window-level predictions into a patient-level diagnosis.
"""
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from src.logger import logger_inst

# FIX: Added 'paths' argument here so it matches the call in modeling.py
def run_patient_voting(model, X_test, y_test, groups_test, paths): 
    logger_inst.info("Running Clinical Voting Audit...")
    probs = model.predict(X_test, verbose=0).flatten()
    
    df = pd.DataFrame({'Subject': groups_test, 'True': y_test, 'Prob': probs})
    
    # Vote: Average probability across windows
    results = df.groupby('Subject').agg({'True': 'first', 'Prob': 'mean'}).reset_index()
    
    # --- SAFETY NET THRESHOLD ---
    threshold = 0.15
    results['Pred'] = (results['Prob'] > threshold).astype(int)
    
    # Text Report
    print("\n=== PATIENT-LEVEL CLINICAL REPORT ===")
    print(classification_report(results['True'], results['Pred'], target_names=['PD', 'ET']))
    
    # --- VISUAL UPGRADE: CONFUSION MATRIX HEATMAP ---
    cm = confusion_matrix(results['True'], results['Pred'])
    
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Pred PD', 'Pred ET'],
                yticklabels=['True PD', 'True ET'])
    plt.title(f"Clinical Confusion Matrix\n(Threshold={threshold})")
    plt.ylabel("Actual Condition")
    plt.xlabel("Model Prediction")
    
    # Save nicely
    plt.tight_layout()
    plt.savefig(paths.figures / "clinical_confusion_matrix.png")
    plt.close()
    
    logger_inst.info(f"Confusion Matrix:\n{cm}")