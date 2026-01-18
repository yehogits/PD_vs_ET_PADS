"""
Preprocessing Module (Bilateral)
--------------------------------
Transforms raw sensor logs into clean 3D tensors for the CNN.
MODE: SYMMETRIC (12 Channels: 6 Left + 6 Right).
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
    """ Slides a window over the signal. Returns shape (N, 400, 12). """
    return np.array([data[i : i + window] for i in range(0, len(data) - window + 1, step)])

def synchronize_streams(df1: pd.DataFrame, df2: pd.DataFrame) -> np.ndarray:
    """
    Synchronizes two smartwatch streams to a common 100Hz timeline.
    Returns a (N, 12) numpy array.
    """
    # 1. Find the common time window (Intersection)
    start_time = max(df1['Time'].min(), df2['Time'].min())
    end_time = min(df1['Time'].max(), df2['Time'].max())
    
    if start_time >= end_time:
        return None

    # 2. Create a master 100Hz clock
    # We create a new index from start to end with 10ms steps (100Hz)
    common_time = np.arange(start_time, end_time, 0.01)
    
    # 3. Interpolate both streams to this clock
    # We drop the 'Time' column after setting it as index for interpolation
    def reindex_stream(df):
        df = df.drop_duplicates(subset='Time').set_index('Time')
        # Reindex to common clock and interpolate values linearly
        df_resampled = df.reindex(df.index.union(common_time)).interpolate(method='linear')
        # Keep only the common timestamps
        return df_resampled.loc[common_time].values

    s1 = reindex_stream(df1) # Shape (N, 6)
    s2 = reindex_stream(df2) # Shape (N, 6)
    
    # 4. Stack columns (Channel 0-5 = Watch A, Channel 6-11 = Watch B)
    return np.hstack([s1, s2]) # Shape (N, 12)

def run_processing_pipeline() -> None:
    paths = Paths.from_here()
    logger_inst.info("Scanning for paired patient files (Bilateral Mode)...")
    
    label_map = {}
    for pf in paths.data_raw.rglob("patient_*.json"):
        try:
            d = json.load(open(pf))
            sid = str(d.get('id')).strip()
            cond = d.get('condition')
            if cond == "Essential Tremor": label_map[sid] = 1
            elif cond == "Parkinson's": label_map[sid] = 0
        except: continue

    all_x, all_y, all_groups = [], [], []
    WIN, STEP = 400, 200  
    COLS = ['Time', 'Acc_X', 'Acc_Y', 'Acc_Z', 'Gyro_X', 'Gyro_Y', 'Gyro_Z']
    
    ts_roots = list(paths.data_raw.rglob("movement/timeseries"))
    if not ts_roots: return
    ts_root = ts_roots[0]

    obs_files = list(paths.data_raw.rglob("observation_*.json"))
    
    for obs_f in obs_files:
        try:
            obs = json.load(open(obs_f))
            sid = str(obs.get('subject_id')).strip()
            if sid not in label_map: continue
            
            for session in obs.get('session', []):
                records = session.get('records', [])
                
                # PAIRING LOGIC: We need at least 2 files (Left & Right)
                if len(records) < 2: continue
                
                # Sort by filename to ensure Device A is always channels 0-5
                # This prevents random flipping of Left/Right inputs
                records.sort(key=lambda x: x.get('file_name', ''))
                
                # Take the first two (Assuming they are the pair)
                rec1, rec2 = records[0], records[1]
                
                f1 = ts_root / Path(rec1.get('file_name')).name
                f2 = ts_root / Path(rec2.get('file_name')).name
                
                if not f1.exists() or not f2.exists(): continue
                
                try:
                    df1 = pd.read_csv(f1, header=None, names=COLS)
                    df2 = pd.read_csv(f2, header=None, names=COLS)
                except: continue
                
                # 1. Synchronize & Merge (6ch -> 12ch)
                merged_signal = synchronize_streams(df1, df2)
                if merged_signal is None or len(merged_signal) < WIN: continue
                
                # 2. Physics Filter (Applied to all 12 channels)
                # Note: We filter the merged array directly
                filtered_signal = apply_physics_filter(merged_signal)
                
                # 3. Gain Boost
                filtered_signal = filtered_signal * 100.0
                
                windows = create_windows(filtered_signal, WIN, STEP)
                
                if len(windows) > 0:
                    all_x.append(windows)
                    all_y.append(np.full(len(windows), label_map[sid]))
                    all_groups.append(np.full(len(windows), sid))
                    
        except Exception as e: continue

    if len(all_x) == 0:
        logger_inst.critical("No data processed! Check paths.")
        return

    X = np.concatenate(all_x)
    y = np.concatenate(all_y)
    groups = np.concatenate(all_groups)
    
    logger_inst.info(f"Bilateral Dataset: {X.shape} (Note: Last dim should be 12)")
    logger_inst.info(f"Class Balance: PD={np.sum(y==0)}, ET={np.sum(y==1)}")
    
    np.save(paths.data_processed / "X_train.npy", X)
    np.save(paths.data_processed / "y_train.npy", y)
    np.save(paths.data_processed / "groups_train.npy", groups)
