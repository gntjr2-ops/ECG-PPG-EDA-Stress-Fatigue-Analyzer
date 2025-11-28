# filters.py
import numpy as np
from scipy.signal import butter, filtfilt

def bandpass(x, fs, lo, hi, order=3):
    lo = max(lo, 1e-3)
    hi = min(hi, fs/2-1e-3)
    b, a = butter(order, [lo/(fs/2), hi/(fs/2)], btype="band")
    return filtfilt(b, a, x)

def lowpass(x, fs, fc, order=3):
    fc = min(fc, fs/2-1e-3)
    b, a = butter(order, fc/(fs/2), btype="low")
    return filtfilt(b, a, x)

def moving_avg(x, n):
    if n <= 1:
        return x.copy()
    k = np.ones(int(n), dtype=float) / float(n)
    return np.convolve(x, k, mode="same")

def zscore(x):
    m, s = np.mean(x), np.std(x) + 1e-8
    return (x - m) / s
