"""
Preprocessing Module (FIXED)
----------------------------
Transforms raw sensor logs into clean 3D tensors for the CNN.
FIX: Restored nested loop to capture ALL patient sessions (not just the first one).

KEY DECISIONS:
1. 2.0Hz High-Pass Filter: Removes gravity and voluntary hand movements.
2. 20.0Hz Low-Pass Filter: Removes sensor noise.
3. Gain Boost (x100): Amplifies micro-tremors for the Neural Net.
"""
import json
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.signal import butter, filtfilt
from src.config import Paths
from src.logger import logger_inst

def apply_physics_filter(data: np.ndarray, fs: float = 100.0) -> np.ndarray:
    """ Applies 2.0Hz - 20.0Hz Bandpass Filter (The Tremor Band). """
    nyq = 0.5 * fs
    b, a = butter(4, [2.0 / nyq, 20.0 / nyq], btype='band')
    return filtfilt(b, a, data, axis=0)

def create_windows(data: np.ndarray, window: int, step: int) -> np.ndarray:
    """ Slides a window over the signal. Returns shape (N_windows, 400, 6). """
    # Faster list comprehension
    return np.array([data[i : i + window] for i in range(0, len(data) - window + 1, step)])

def run_processing_pipeline() -> None:
    paths = Paths.from_here()
    logger_inst.info("Scanning for patient files...")
    
    # 1. Map Patient IDs to Conditions (PD=0, ET=1)
    label_map = {}
    for pf in paths.data_raw.rglob("patient_*.json"):
        try:
            d = json.load(open(pf))
            # FIX: Added .strip() back to be safe against whitespace
            sid = str(d.get('id')).strip()
            cond = d.get('condition')
            if cond == "Essential Tremor":
                label_map[sid] = 1
            elif cond == "Parkinson's":
                label_map[sid] = 0
        except: continue

    # 2. Processing Loop
    all_x, all_y, all_groups = [], [], []
    WIN, STEP = 400, 200  # 4 seconds, 50% overlap
    COLS = ['Time', 'Acc_X', 'Acc_Y', 'Acc_Z', 'Gyro_X', 'Gyro_Y', 'Gyro_Z']
    
    # Locate all session files
    obs_files = list(paths.data_raw.rglob("observation_*.json"))
    # Robust way to find the timeseries folder
    ts_roots = list(paths.data_raw.rglob("movement/timeseries"))
    if not ts_roots:
        logger_inst.critical("Could not find 'movement/timeseries' folder!")
        return
    ts_root = ts_roots[0]

    for obs_f in obs_files:
        try:
            obs = json.load(open(obs_f))
            sid = str(obs.get('subject_id')).strip()
            if sid not in label_map: continue
            
            # FIX: Loop through ALL sessions, not just [0]
            for session in obs.get('session', []):
                for rec in session.get('records', []):
                    f_name = rec.get('file_name')
                    if not f_name: continue
                    
                    f_path = ts_root / Path(f_name).name
                    if not f_path.exists(): continue
                    
                    # Load & Check Length
                    try:
                        df = pd.read_csv(f_path, header=None, names=COLS)
                    except: continue
                    
                    if len(df) < WIN: continue

                    # 1. Physics Filter
                    signals = apply_physics_filter(df[COLS[1:]].values)
                    # 2. Gain Boost (Crucial for Neural Net visibility)
                    signals = signals * 100.0
                    
                    windows = create_windows(signals, WIN, STEP)
                    
                    if len(windows) > 0:
                        all_x.append(windows)
                        all_y.append(np.full(len(windows), label_map[sid]))
                        all_groups.append(np.full(len(windows), sid))
        except Exception as e:
            continue

    if len(all_x) == 0:
        logger_inst.critical("No data processed! Check paths.")
        return

    # Save Vectors
    X = np.concatenate(all_x)
    y = np.concatenate(all_y)
    groups = np.concatenate(all_groups)
    
    logger_inst.info(f"Processed Dataset: {X.shape}")
    logger_inst.info(f"Class Balance: PD={np.sum(y==0)}, ET={np.sum(y==1)}")
    
    np.save(paths.data_processed / "X_train.npy", X)
    np.save(paths.data_processed / "y_train.npy", y)
    np.save(paths.data_processed / "groups_train.npy", groups)