# preprocess.py
import numpy as np
from scipy.signal import find_peaks
from filters import bandpass, lowpass

# --- 전처리 ---
def preprocess_ecg(ecg, fs):
    # QRS 대역 근사(5~30Hz)
    return bandpass(ecg, fs, 5.0, 30.0, order=3)

def preprocess_ppg(ppg, fs):
    # 맥파 대역 근사(0.5~8Hz)
    return bandpass(ppg, fs, 0.5, 8.0, order=3)

def preprocess_eda(eda, fs):
    # 저역통과(tonic 추출) + 고주파 노이즈 억제
    return lowpass(eda, fs, fc=2.0, order=3)

# --- 검출 ---
def detect_r_peaks(ecg_f, fs):
    # 간이 peak: QRS 폭 ~ 80~120ms, 최소 간격 250ms 가정
    distance = int(0.25 * fs)
    height = np.percentile(ecg_f, 75)
    peaks, _ = find_peaks(ecg_f, distance=distance, height=height)
    return peaks

def detect_ppg_peaks(ppg_f, fs):
    # 주 피크 (systolic peak)
    distance = int(0.35 * fs)
    height = np.percentile(ppg_f, 70)
    peaks, _ = find_peaks(ppg_f, distance=distance, height=height)
    return peaks

def detect_ppg_foot(ppg_f, fs):
    """
    PPG foot(발 기점) 근사: 파형의 상승 시작점(1차 미분 최대 전의 영교차) 간이검출
    간단히는 이동 최소값 근방에서 상승 시작으로 근사
    """
    n = len(ppg_f)
    win = max(3, int(0.15 * fs))
    # 이동 최소(rough baseline)
    mins = np.minimum.accumulate(ppg_f)  # 간단 근사
    # 상승 구간: ppg_f - 최소값이 작고 이후 기울기 양수
    d1 = np.gradient(ppg_f)
    cand = np.where((d1 > 0) & (ppg_f - mins < np.std(ppg_f)))[0]
    # 간격 제한(맥박 주기 ~ 0.35s 이상)
    feet = []
    last = -10**9
    min_dist = int(0.35 * fs)
    for idx in cand:
        if idx - last >= min_dist:
            feet.append(idx)
            last = idx
    return np.array(feet, dtype=int)
