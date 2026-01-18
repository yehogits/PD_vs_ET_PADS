"""
Modeling Module
---------------
Defines the CNN-LSTM Hybrid Architecture.
Strategy:
1. CNN (Wide Kernel): Detects the 4Hz-7Hz wave shape.
2. LSTM: Detects the temporal rhythm.
3. Class Weights (1:6): Forces the model to pay 6x more attention to ET to fix imbalance.
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras import layers, models, callbacks
from sklearn.metrics import classification_report, confusion_matrix, recall_score
from sklearn.model_selection import GroupShuffleSplit
from src.config import Paths
from src.logger import logger_inst
from src.audit import visualize_latent_space, run_adversarial_stress_test
from src.interpretation import plot_filter_physics, plot_clinical_saliency
from src.voting import run_patient_voting

def build_model(input_shape: tuple) -> models.Model:
    model = models.Sequential([
        # 1. Physics Kernel (50 steps = 0.5s, captures ~2Hz cycles)
        layers.Conv1D(64, kernel_size=50, activation='relu', input_shape=input_shape, padding='same'),
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
    logger_inst.info("Loading tensors...")
    X = np.load(paths.data_processed / "X_train.npy")
    y = np.load(paths.data_processed / "y_train.npy")
    groups = np.load(paths.data_processed / "groups_train.npy")

    # Patient-Level Split (Crucial to prevent data leakage)
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(splitter.split(X, y, groups))
    
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    groups_test = groups[test_idx]

    # --- CONFIGURATION FOR 100% RECALL ---
    # Weight 6.0: The "Sweet Spot" that balances signal vs noise.
    class_weights = {0: 1.0, 1: 6.0} 

    model = build_model((X_train.shape[1], X_train.shape[2]))
    early_stop = callbacks.EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True)

    logger_inst.info(f"Training on {len(X_train)} samples with Class Weights {class_weights}...")
    model.fit(
        X_train, y_train,
        validation_split=0.15,
        epochs=35,
        batch_size=64,
        class_weight=class_weights,
        callbacks=[early_stop],
        verbose=1
    )

    # --- AUDIT & INTERPRETATION ---
    # 1. Geometry (Islands vs Strings)
    visualize_latent_space(model, X_test, y_test, paths)
    
    # 2. Robustness (Can noise break it?)
    run_adversarial_stress_test(model, X_test, y_test, paths)
    
    # 3. Physics (Did it learn frequency?)
    plot_filter_physics(model, paths)
    
    # 4. Saliency (What did it look at?)
    plot_clinical_saliency(model, X_test, y_test, paths)
    
    # 5. Clinical Voting (The "Safety Net")
    run_patient_voting(model, X_test, y_test, groups_test, paths)