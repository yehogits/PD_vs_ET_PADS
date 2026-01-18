# PADS: Parkinson's Disease vs. Essential Tremor Classification

## Project Overview
This project presents a **Deep Learning "Safety Net"** for differentiating Parkinson's Disease (PD) from Essential Tremor (ET) using smartwatch inertial data. Unlike standard black-box models, this system validates its decisions using **Geometric Topology**, **Adversarial Stress-Testing**, and **Physics-Based Filter Analysis**.

## Key Innovations
1.  **Physics-Aware Architecture:** The CNN utilizes a wide receptive field (0.5s) which spontaneously learned to isolate the **4Hz-6Hz tremor frequency** without explicit supervision (validated via FFT audit).
2.  **Geometric Validity:** UMAP projection reveals that ET patients form distinct topological "islands" separate from the PD manifold, proving the model detects a real phenotype.
3.  **The "Safety Net" Strategy:**
    * Optimized for **100% Sensitivity (Recall)** using a Class Weight of 1:6 and a Voting Threshold of 0.15.
    * **Clinical Rationale:** The system acts as a rule-out screener. We prioritize catching *every* ET case, accepting a higher False Positive rate (PD flagged as ET) to ensure no pathological tremor is dismissed.

## Pipeline Components
* **Preprocessing:** 2.0Hz High-Pass filter (gravity removal) + 100x Gain Boost.
* **Model:** Hybrid **1D-CNN + LSTM** (Spatiotemporal feature extraction).
* **Audit Modules:**
    * `audit.py`: Adversarial Robustness (FGSM) & Latent Space (UMAP).
    * `interpretation.py`: Saliency Mapping & Filter Frequency Response.
    * `voting.py`: Patient-level aggregation.

## Installation & Run
1.  **Clone:** `git clone <repo>`
2.  **Install:** `pip install -r requirements.txt` (Ensure `tensorflow`, `umap-learn`, `scikit-learn`, `pandas`, `scipy` are included).
3.  **Run:** `python main.py`

## Results
* **Patient-Level Recall:** 100% (6/6 ET patients identified).
* **Adversarial Robustness:** <2% Error rate under 20% signal noise.
