# features.py
import numpy as np
from filters import zscore
from scipy.signal import welch

# ---------- ECG/PPG ----------
def ibi_from_peaks(peaks, fs):
    if peaks is None or len(peaks) < 2:
        return None
    return np.diff(peaks) / fs

def hr_from_ibi(ibi):
    if ibi is None or len(ibi) == 0:
        return None
    return 60.0 / np.mean(ibi)

def sdnn(ibi):
    if ibi is None or len(ibi) == 0:
        return None
    return float(np.std(ibi))

def rmssd(ibi):
    if ibi is None or len(ibi) < 2:
        return None
    return float(np.sqrt(np.mean(np.diff(ibi) ** 2)))

def lfhf_ratio_from_rr(rr_series, fs_rr=4.0):
    if rr_series is None or len(rr_series) < 8:
        return None
    rr = rr_series - np.mean(rr_series)
    if np.allclose(rr, 0):
        return None

    f, pxx = welch(rr, fs=fs_rr, nperseg=min(256, len(rr)))
    lf_band = (f >= 0.04) & (f <= 0.15)
    hf_band = (f >= 0.15) & (f <= 0.40)

    lf = np.trapz(pxx[lf_band], f[lf_band]) if np.any(lf_band) else 0.0
    hf = np.trapz(pxx[hf_band], f[hf_band]) if np.any(hf_band) else 0.0

    if hf <= 0.0:
        return None
    return lf / hf

def ptt_from_pairs(r_peaks, ppg_feet, fs):
    if r_peaks is None or ppg_feet is None or len(r_peaks) == 0 or len(ppg_feet) == 0:
        return None
    i = j = 0
    diffs = []
    while i < len(r_peaks) and j < len(ppg_feet):
        if ppg_feet[j] <= r_peaks[i]:
            j += 1
            continue
        diffs.append((ppg_feet[j] - r_peaks[i]) / fs)
        i += 1
        j += 1
    return float(np.mean(diffs)) if diffs else None

# ---------- EDA ----------
def scl_level(eda_tonic):
    return float(np.mean(eda_tonic)) if eda_tonic is not None and len(eda_tonic) else None

def scr_frequency(eda_tonic, fs, thr_add=0.02):
    """
    매우 간단한 SCR 빈도 추정: tonic 평균 + thr_add 초과 상승 이벤트 카운트
    """
    if eda_tonic is None or len(eda_tonic) < 2:
        return None
    mean = np.mean(eda_tonic)
    thr = mean + thr_add
    up = (eda_tonic[:-1] < thr) & (eda_tonic[1:] >= thr)
    freq_hz = np.sum(up) / (len(eda_tonic) / fs)
    return float(freq_hz)
