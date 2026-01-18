"""
Modeling Module (Bilateral)
---------------------------
Architecture: 12-Channel Early Fusion.
Inputs: (400, 12) - Both wrists simultaneously.
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras import layers, models, callbacks
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GroupShuffleSplit
from src.config import Paths
from src.logger import logger_inst
from src.audit import visualize_latent_space, run_adversarial_stress_test
from src.interpretation import plot_filter_physics, plot_clinical_saliency
from src.voting import run_patient_voting

def build_model(input_shape: tuple) -> models.Model:
    model = models.Sequential([
        # 1. Physics Kernel (12 Channels Input)
        # Increased filters 64->96 to handle bilateral correlations
        layers.Conv1D(96, kernel_size=50, activation='relu', padding='same', input_shape=input_shape),
        layers.LayerNormalization(),
        layers.MaxPool1D(2),
        layers.Dropout(0.1),

        # 2. Detail Kernel
        layers.Conv1D(128, kernel_size=10, activation='relu', padding='same'),
        layers.LayerNormalization(),
        layers.MaxPool1D(2),
        layers.Dropout(0.1),

        # 3. Temporal Memory
        layers.LSTM(64, return_sequences=False),
        layers.Dropout(0.1),

        # 4. Classifier
        layers.Dense(32, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def run_modeling_pipeline():
    paths = Paths.from_here()
    logger_inst.info("Loading bilateral tensors...")
    X = np.load(paths.data_processed / "X_train.npy")
    y = np.load(paths.data_processed / "y_train.npy")
    groups = np.load(paths.data_processed / "groups_train.npy")

    # Patient-Level Split
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(splitter.split(X, y, groups))
    
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    groups_test = groups[test_idx]

    # Weighted for 100% Recall
    class_weights = {0: 1.0, 1: 6.0} 

    model = build_model((X_train.shape[1], X_train.shape[2]))
    early_stop = callbacks.EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True)

    logger_inst.info(f"Training on {len(X_train)} bilateral samples...")
    model.fit(
        X_train, y_train,
        validation_split=0.15,
        epochs=35,
        batch_size=64,
        class_weight=class_weights,
        callbacks=[early_stop],
        verbose=1
    )

    # Audits
    visualize_latent_space(model, X_test, y_test, paths)
    run_adversarial_stress_test(model, X_test, y_test, paths)
    plot_filter_physics(model, paths)
    plot_clinical_saliency(model, X_test, y_test, paths)
    
    # Pass 'paths' to voting (as fixed in the last error)
    run_patient_voting(model, X_test, y_test, groups_test, paths)
