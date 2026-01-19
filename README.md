# PADS: Parkinson's Disease vs. Essential Tremor Classification

## Project Overview
**PADS** is a physics-aware Deep Learning system designed to distinguish between **Parkinson's Disease (PD)** and **Essential Tremor (ET)** using smartwatch accelerometer data.

Unlike standard "black-box" classifiers, this project implements a **"Safety Net" Architecture**:
1.  **Clinical Priority:** It optimizes for **100% Sensitivity (Recall)** to ensure no pathological tremor is missed (ET is weighted 6x higher than PD).
2.  **Physics Verification:** It validates that the model actually learns the 4-6Hz tremor band using FFT analysis of the CNN filters.
3.  **Geometric Validity:** It uses UMAP to prove that the model separates patients into distinct topological manifolds.

## Key Features

### 1. Dual-Pipeline Architecture
The system supports two distinct experimental modes via a unified CLI:
* **Asymmetrical (Single Watch):** Analyzes raw movement data from a single limb to detect tremor signatures independent of coordination.
* **Symmetrical (Bilateral):** Synchronizes data from **two smartwatches** (Left & Right) to a common 100Hz clock. This captures inter-limb coordination features crucial for distinguishing PD (often asymmetrical) from ET (often symmetrical).

### 2. Physics-Aware Preprocessing
* **Bandpass Filter (2Hz - 20Hz):** Removes gravity (DC component) and high-frequency sensor noise, isolating the human tremor band.
* **Gain Boost (x100):** Amplifies micro-tremors to ensure the Neural Network can detect subtle onset patterns.

### 3. "Glass Box" Auditing
The pipeline includes built-in interpretability modules:
* **Spectral Density (PSD):** Verifies the input data contains distinct frequency peaks.
* **Adversarial Stress-Test (FGSM):** Bombards the model with invisible noise to measure robustness.
* **Saliency Mapping:** Visualizes exactly which movement segments triggered the diagnosis.

---

## Repository Structure

The project is organized into two parallel pipelines managed by a root executor.

```text
├── main.py                     # Entry point (Interactive CLI)
├── data/                       # Stores raw and processed tensors
├── reports/                    # Generated figures and logs
├── asymetrical/                # PIPELINE 1: Single-Limb Analysis
│   ├── src/
│   │   ├── modeling.py         # CNN-LSTM Architecture
│   │   ├── audit.py            # UMAP & FGSM tests
│   │   └── ...
├── symetrical/                 # PIPELINE 2: Bilateral Analysis
│   ├── src/
│   │   ├── dataset.py          # Contains 'synchronize_streams' logic
│   │   ├── voting.py           # Patient-level consensus logic
│   │   └── ...
└── pyproject.toml              # Dependencies & Configuration
```

## Clone the Repository

git clone [https://github.com/yehogits/PDvsET_PADS.git](https://github.com/yehogits/PDvsET_PADS.git)
cd PDvsET_PADS


## Install Dependencies

pip install tensorflow numpy pandas scipy umap-learn matplotlib seaborn


## Usage

**Run**

python main.py

**Follow the prompt:** 

"
Which process would you like to run?
1: Asymmetrical (Single Watch Analysis)
2: Symmetrical (Dual Watch Synchronization & Analysis)

Enter choice (1 or 2):
"


## Pipeline Steps (Automated)
Once selected, the system runs the following stages automatically:

1. Fetch: Downloads the PADS dataset from PhysioNet (if missing).
2. Process: Applies Physics Filtering, Gain Boost, and Windowing (4s windows). Symmetrical mode will auto-sync Left/Right streams here.
3. Explore: Generates PSD plots to audit data quality.
4. Train: Trains the Hybrid CNN-LSTM model with Class_Weight = {0: 1.0, 1: 6.0}.
5. Audit: Runs UMAP projection, Adversarial attacks, and Clinical Saliency maps.
6. Report: Outputs a Patient-Level diagnosis (Confusion Matrix & Classification Report).


## Model Architecture
The core model (src/modeling.py) is a Hybrid 1D-CNN + LSTM:

Layer,Specs,Purpose
Input: "(400, 6) or (400, 12)" ; 4 seconds of accelerometer/gyro data.
Conv1D: "k=50 (0.5s), 64/96 filters" ; Captures low-frequency tremor shapes (Physics Kernel).
Conv1D: "k=10, 128 filters" ; Captures high-frequency detail textures.
LSTM: 64 units ; Analyzes the temporal rhythm and consistency.
Dense: Sigmoid Output ; Binary Classification (PD vs. ET).


## Results & Artifacts:

After running the pipeline, check the reports/ directories for:

- geometric_audit_umap.png: Proof of topological separation between diseases.
- interpretation_filter_physics.png: Verification that the CNN learned the 4-6Hz band.
- clinical_confusion_matrix.png: Final diagnostic performance.

Current Benchmark (Symmetrical):
- Recall (Sensitivity): 100% (All ET patients correctly identified).
- Robustness: <4% Error rate under adversarial noise.


## License
This project uses the PADS dataset (PhysioNet). Code is open-source.