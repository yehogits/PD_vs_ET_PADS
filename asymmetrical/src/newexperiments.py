import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from scipy.fft import rfft, rfftfreq
from sklearn.model_selection import GroupShuffleSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix

from src.logger import logger_inst
from src.config import Paths

# ==========================================
# 1. DATA LOADING
# ==========================================
def load_data_with_groups(paths: Paths):
    logger_inst.info("Loading data directly from processed .npy files for new experiments...")
    X = np.load(paths.data_processed / "X_train.npy")
    y = np.load(paths.data_processed / "y_train.npy")
    groups = np.load(paths.data_processed / "groups_train.npy")
    
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(gss.split(X, y, groups))
    
    return X[train_idx], y[train_idx], groups[train_idx], X[test_idx], y[test_idx], groups[test_idx]

# ==========================================
# 2. PATIENT-LEVEL EVALUATION (Calibrated %)
# ==========================================
def evaluate_patient_level(preds_sample_prob, y_test, groups_test, model_name, paths, threshold=0.5):
    unique_patients = np.unique(groups_test)
    patient_y_true = []
    patient_y_pred = []
    
    for patient in unique_patients:
        idx = np.where(groups_test == patient)[0]
        patient_y_true.append(y_test[idx[0]]) 
        pred_label = 1 if np.mean(preds_sample_prob[idx]) > threshold else 0 
        patient_y_pred.append(pred_label)
        
    cm = confusion_matrix(patient_y_true, patient_y_pred, labels=[0, 1])
    
    true_pd_total = cm[0, 0] + cm[0, 1]
    true_et_total = cm[1, 0] + cm[1, 1]
    
    true_pd_pred_et = cm[0, 1] 
    true_et_pred_pd = cm[1, 0] 
    
    perc_pd_as_et = (true_pd_pred_et / max(1, true_pd_total)) * 100
    perc_et_as_pd = (true_et_pred_pd / max(1, true_et_total)) * 100
    
    acc = accuracy_score(patient_y_true, patient_y_pred)
    logger_inst.info(f"[{model_name}] Patient-Level Accuracy: {acc:.4f} (Threshold: {threshold})")
    
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['PD', 'ET'], yticklabels=['PD', 'ET'])
    plt.title(f"Clinical CM (Patients): {model_name}\nThreshold: {threshold}")
    plt.ylabel("True Diagnosis")
    plt.xlabel("Predicted Diagnosis")
    safe_name = model_name.replace(' ', '_').replace('(', '').replace(')', '')
    plt.savefig(paths.reports / f"cm_patient_{safe_name}.png")
    plt.close()
    
    return perc_et_as_pd, perc_pd_as_et

# ==========================================
# 3. FFT & CLASSICAL MODELS
# ==========================================
def extract_hybrid_features(X_tensor: np.ndarray, fs=100.0):
    N, T, C = X_tensor.shape
    stats = np.hstack([np.mean(X_tensor, axis=1), np.std(X_tensor, axis=1), np.max(X_tensor, axis=1)])
    yf = np.abs(rfft(X_tensor, axis=1))
    xf = rfftfreq(T, 1 / fs)
    idx_band = np.where((xf >= 3) & (xf <= 15))[0]
    fft_feats = yf[:, idx_band, :].reshape(N, -1)
    return np.hstack([stats, fft_feats])

def run_classical_benchmarks(X_train, y_train, X_test, y_test, groups_test, paths):
    logger_inst.info(">>> RUNNING CLASSICAL BASELINES <<<")
    X_train_feat = extract_hybrid_features(X_train)
    X_test_feat = extract_hybrid_features(X_test)
    
    models_dict = {
        "FFT (Logistic Reg)": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight='balanced')),
        "Random Forest": RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42),
        "SVM (RBF)": make_pipeline(StandardScaler(), SVC(class_weight='balanced', probability=True))
    }
    
    thresholds = {
        "FFT (Logistic Reg)": 0.50, 
        "Random Forest": 0.15,      
        "SVM (RBF)": 0.15           
    }
        
    results = {}
    for name, model in models_dict.items():
        logger_inst.info(f"Training {name}...")
        model.fit(X_train_feat, y_train)
        preds_prob = model.predict_proba(X_test_feat)[:, 1] if hasattr(model, "predict_proba") else model.predict(X_test_feat)
        
        thresh = thresholds.get(name, 0.5)
        missed_et_perc, missed_pd_perc = evaluate_patient_level(preds_prob, y_test, groups_test, name, paths, threshold=thresh)
        results[name] = {"True ET predicted as PD (%)": missed_et_perc, "True PD predicted as ET (%)": missed_pd_perc}
    return results

# ==========================================
# 4. DEEP LEARNING (Small 1D CNN)
# ==========================================
def build_small_cnn(input_shape):
    inputs = layers.Input(shape=input_shape)
    x = layers.Conv1D(32, 5, activation='relu')(inputs)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Conv1D(64, 5, activation='relu')(x)
    x = layers.GlobalMaxPooling1D()(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)
    model = models.Model(inputs, outputs, name="Small_1D_CNN")
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# ==========================================
# 5. ORCHESTRATOR
# ==========================================
def run_all_new_experiments(hybrid_model=None, hybrid_cm=None):
    paths = Paths()
    X_train, y_train, groups_train, X_test, y_test, groups_test = load_data_with_groups(paths)
    input_shape = (X_train.shape[1], X_train.shape[2])
    
    final_comparisons = {}

    # 1. ADD ORIGINAL HYBRID MODEL TO COMPARISON
    if hybrid_cm is not None:
        logger_inst.info("--- Adding intercepted Original Hybrid CNN-LSTM to comparison ---")
        true_pd_total = hybrid_cm[0, 0] + hybrid_cm[0, 1]
        true_et_total = hybrid_cm[1, 0] + hybrid_cm[1, 1]
        true_pd_pred_et = hybrid_cm[0, 1]
        true_et_pred_pd = hybrid_cm[1, 0]
        
        perc_pd_as_et = (true_pd_pred_et / max(1, true_pd_total)) * 100
        perc_et_as_pd = (true_et_pred_pd / max(1, true_et_total)) * 100
        
        final_comparisons["Original Hybrid CNN-LSTM"] = {
            "True ET predicted as PD (%)": perc_et_as_pd, 
            "True PD predicted as ET (%)": perc_pd_as_et
        }

    # 2. RUN CLASSICAL MODELS (FFT, RF, SVM)
    results_classical = run_classical_benchmarks(X_train, y_train, X_test, y_test, groups_test, paths)
    final_comparisons.update(results_classical)
    
    # 3. RUN SMALL 1D CNN
    logger_inst.info(f"\n{'='*30}\nTraining Small 1D CNN\n{'='*30}")
    small_cnn = build_small_cnn(input_shape)
    n_pd, n_et = np.sum(y_train == 0), np.sum(y_train == 1)
    class_weights = {0: 1.0, 1: n_pd / max(1, n_et)}
    early_stop = callbacks.EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True)
    
    small_cnn.fit(
        X_train, y_train, 
        epochs=35, 
        batch_size=64, 
        validation_split=0.15, 
        class_weight=class_weights, 
        callbacks=[early_stop], 
        verbose=0
    )
    preds_prob = small_cnn.predict(X_test, verbose=0).flatten()
    
    missed_et_perc, missed_pd_perc = evaluate_patient_level(preds_prob, y_test, groups_test, "Small 1D CNN", paths, threshold=0.15)
    final_comparisons["Small 1D CNN"] = {"True ET predicted as PD (%)": missed_et_perc, "True PD predicted as ET (%)": missed_pd_perc}

    # 4. FINAL COMPARISON PLOT
    df_compare = pd.DataFrame(final_comparisons).T
    df_compare.reset_index(inplace=True)
    df_compare.rename(columns={'index': 'Model'}, inplace=True)
    
    df_melted = df_compare.melt(id_vars='Model', var_name='Misdiagnosis Type', value_name='Percentage of Patients (%)')
    
    plt.figure(figsize=(14, 7))
    sns.barplot(data=df_melted, x='Model', y='Percentage of Patients (%)', hue='Misdiagnosis Type', palette=['#e74c3c', '#3498db'])
    plt.title("Patient-Level Misdiagnoses by Model (Relative %)")
    plt.xticks(rotation=15, ha="right")
    plt.ylabel("% of Misdiagnosed Patients")
    plt.ylim(0, 100)
    plt.tight_layout()
    plt.savefig(paths.reports / "patient_misdiagnoses_comparison.png")
    plt.close()
    
    logger_inst.info("All new experiments completed! Check reports/patient_misdiagnoses_comparison.png")

if __name__ == "__main__":
    run_all_new_experiments()