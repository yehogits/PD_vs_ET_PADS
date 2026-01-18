"""
Exploration Module
------------------
Audits the data quality before training.
Includes Signal Visualization and Spectral Density (PSD) analysis.
"""
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.signal import welch
from src.config import Paths
from src.logger import logger_inst
from src.dataset import apply_physics_filter

def plot_spectral_density(paths: Paths):
    """ The 'Physics Check': Do the classes have different frequencies? """
    logger_inst.info("Generating Spectral Density (PSD) Audit...")
    try:
        X = np.load(paths.data_processed / "X_train.npy")
        y = np.load(paths.data_processed / "y_train.npy")
    except: return

    # Calculate magnitude of acceleration for PD vs ET
    X_pd = np.sqrt(np.sum(X[y == 0][:, :, :3]**2, axis=2))
    X_et = np.sqrt(np.sum(X[y == 1][:, :, :3]**2, axis=2))

    # Welch's Method for PSD
    freqs, psd_pd = welch(X_pd, fs=100, nperseg=200)
    freqs, psd_et = welch(X_et, fs=100, nperseg=200)

    plt.figure(figsize=(10, 6))
    plt.plot(freqs, np.mean(psd_pd, axis=0), label="Parkinson's (PD)", color='blue')
    plt.plot(freqs, np.mean(psd_et, axis=0), label="Essential Tremor (ET)", color='red')
    plt.title("Tremor Signature: Power Spectral Density (PSD)")
    plt.xlabel("Frequency (Hz)")
    plt.xlim(0, 20)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(paths.figures / "spectral_audit.png")
    plt.close()

def run_exploration():
    paths = Paths.from_here()
    plot_spectral_density(paths)
    # (Additional standard plots can remain here if needed)