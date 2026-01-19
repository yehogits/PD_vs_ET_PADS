# Audit Module: Validates model reliability beyond accuracy.

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras import models
import umap
from src.config import Paths
from src.logger import logger_inst

def visualize_latent_space(model, X_test, y_test, paths: Paths):

    # Extracts the 'Bottleneck' features (output of the Dense layer before probability) and projects them into 2D using UMAP.
    
    logger_inst.info("Running Geometric Audit (UMAP)...")
    
    # Extract output from the second-to-last layer (Dense 32)
    extractor = models.Model(inputs=model.inputs, outputs=model.layers[-2].output)
    
    # Predict to get the "Brain Waves" (Latent Features)
    latent_features = extractor.predict(X_test, verbose=0)
    
    # Reduce dimensions
    reducer = umap.UMAP(random_state=42)
    embedding = reducer.fit_transform(latent_features)
    
    plot_labels = ["ET" if y==1 else "PD" for y in y_test]

    # Plot
    plt.figure(figsize=(10, 8))
    sns.scatterplot(
        x=embedding[:, 0], y=embedding[:, 1], 
        hue=plot_labels, palette={'PD': 'blue', 'ET': 'red'}, alpha=0.6,
        style=plot_labels, markers={'PD': 'o', 'ET': 'X'}
    )
    plt.title("Geometric Analysis: Latent Space Topology (UMAP)")
    
    plt.xlabel("UMAP Dimension 1")
    plt.ylabel("UMAP Dimension 2")
    
    save_path = paths.reports / "geometric_audit_umap.png"
    plt.savefig(save_path)
    logger_inst.info(f"UMAP Projection saved to {save_path}")
    plt.close()

def run_adversarial_stress_test(model, X_test, y_test, paths: Paths):
    
    # Applies invisible noise (epsilon) to PD samples to see if they flip to ET.
    
    logger_inst.info("Running Adversarial Stress-Test (FGSM)...")
    
    # Test on 100 PD patients (Label 0)
    pd_indices = np.where(y_test == 0)[0][:100]
    if len(pd_indices) == 0: return

    X_sample = tf.convert_to_tensor(X_test[pd_indices], dtype=tf.float32)
    y_sample = tf.convert_to_tensor(y_test[pd_indices].reshape(-1, 1), dtype=tf.float32)

    # Calculate Gradient of Loss w.r.t Input Image
    with tf.GradientTape() as tape:
        tape.watch(X_sample)
        prediction = model(X_sample)
        loss = tf.keras.losses.binary_crossentropy(y_sample, prediction)

    gradient = tape.gradient(loss, X_sample)
    signed_grad = tf.sign(gradient)
    
    perturbations = [0.01, 0.05, 0.1, 0.2] # Noise levels
    flip_rates = []

    for eps in perturbations:
        X_adv = X_sample + eps * signed_grad
        preds = model.predict(X_adv, verbose=0)
        
        # Count flips to ET (> 0.5)
        flips = np.sum(preds > 0.5)
        flip_rate = (flips / len(X_sample)) * 100
        flip_rates.append(flip_rate)
        
    plt.figure(figsize=(8, 5))
    plt.plot(perturbations, flip_rates, marker='o', color='red', linewidth=2)
    plt.title("Adversarial Robustness Curve (FGSM)")
    plt.xlabel("Perturbation Magnitude (Epsilon)")
    plt.ylabel("% of Misdiagnoses (PD -> ET)")
    plt.grid(True)
    
    save_path = paths.reports / "adversarial_audit.png"
    plt.savefig(save_path)
    logger_inst.info(f"Adversarial Report saved to {save_path}")
    plt.close()