# Interpretation Module: Opens the "Black Box".

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from scipy.fft import fft, fftfreq
from src.config import Paths
from src.logger import logger_inst

def plot_filter_physics(model, paths: Paths):
    # Do the CNN filters act like Frequency Analyzers?
    logger_inst.info("Running Physics Audit...")
    try:
        weights = model.layers[0].get_weights()[0]
    except: return
        
    kernels = np.mean(weights, axis=1)
    n = 50
    xf = fftfreq(n, 1/100)[:n//2]
    avg_response = np.mean([np.abs(fft(kernels[:, i])[:n//2]) for i in range(64)], axis=0)
    
    plt.figure(figsize=(10, 6))
    plt.plot(xf, avg_response, color='red', linewidth=3, label='Learned Filter Response')
    plt.axvspan(3, 6, color='blue', alpha=0.1, label='PD Band (3-6 Hz)')
    plt.axvspan(6, 12, color='green', alpha=0.1, label='ET Band (6-12 Hz)')
    
    plt.title("Physics Proof: Did the CNN learn Tremor Frequencies?")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Filter Activation Magnitude")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Zoom in on the relevant band
    plt.xlim(0, 25) 
    
    plt.savefig(paths.reports / "interpretation_filter_physics.png")
    plt.close()

def plot_clinical_saliency(model, X_test, y_test, paths: Paths):
    # Grad-CAM for 1D.
    logger_inst.info("Running Saliency Audit...")
    
    probs = model.predict(X_test, verbose=0).flatten()
    
    # Take the highest probability ET case, even if it's below 0.85
    et_indices = np.where(y_test == 1)[0]
    if len(et_indices) == 0: return

    # Sort ET cases by probability and pick the best one
    best_candidate_idx = et_indices[np.argmax(probs[et_indices])]
    
    idx = best_candidate_idx
    tensor = tf.convert_to_tensor([X_test[idx]], dtype=tf.float32)
    
    with tf.GradientTape() as tape:
        tape.watch(tensor)
        preds = model(tensor)
    
    grads = tape.gradient(preds, tensor)
    saliency = tf.reduce_max(tf.abs(grads), axis=-1)[0].numpy()
    
    time_axis = np.arange(400) / 100.0
    
    plt.figure(figsize=(12, 6))
    plt.plot(time_axis, X_test[idx][:,0], color='black', alpha=0.6, label='Acceleration X')
    plt.scatter(time_axis, X_test[idx][:,0], c=saliency, cmap='hot', label='AI Attention', s=15)
    
    plt.colorbar(label='Importance (Red = High)')
    plt.title(f"Clinical Saliency: Which movements triggered the diagnosis?\n(Patient #{idx} - Prob: {probs[idx]:.2f})")
    plt.xlabel("Time (seconds)") 
    plt.ylabel("Acceleration (Normalized)")
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)
    
    plt.savefig(paths.reports / "interpretation_saliency.png")
    plt.close()