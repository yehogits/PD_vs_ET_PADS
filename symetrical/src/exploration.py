"""
Exploration Module
------------------
Audits the data quality (Bilateral Compatible).
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch
from src.config import Paths
from src.logger import logger_inst

def plot_spectral_density(paths: Paths):
    logger_inst.info("Generating Spectral Density (PSD) Audit...")
    try:
        X = np.load(paths.data_processed / "X_train.npy")
        y = np.load(paths.data_processed / "y_train.npy")
    except: return

    # Calculate magnitude of Watch A (Cols 0,1,2)
    # If we wanted Watch B, we would use [:, :, 6:9]
    X_pd = np.sqrt(np.sum(X[y == 0][:, :, :3]**2, axis=2))
    X_et = np.sqrt(np.sum(X[y == 1][:, :, :3]**2, axis=2))

    freqs, psd_pd = welch(X_pd, fs=100, nperseg=200)
    freqs, psd_et = welch(X_et, fs=100, nperseg=200)

    plt.figure(figsize=(10, 6))
    plt.plot(freqs, np.mean(psd_pd, axis=0), label="Parkinson's (PD)", color='blue')
    plt.plot(freqs, np.mean(psd_et, axis=0), label="Essential Tremor (ET)", color='red')
    plt.title("Tremor Signature: Power Spectral Density (Watch A)")
    plt.xlabel("Frequency (Hz)")
    plt.xlim(0, 20)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(paths.figures / "spectral_audit.png")
    plt.close()

def run_exploration():
    paths = Paths.from_here()
    plot_spectral_density(paths)
