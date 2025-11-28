# pipeline.py
import numpy as np
from typing import Dict
from preprocess import (
    preprocess_ecg, preprocess_ppg, preprocess_eda,
    detect_r_peaks, detect_ppg_peaks, detect_ppg_foot
)
from features import (
    ibi_from_peaks, hr_from_ibi, sdnn, rmssd, lfhf_ratio_from_rr,
    ptt_from_pairs, scl_level, scr_frequency
)
from heuristics import classify_heuristic


class HeuristicStressPipeline:
    def __init__(self, fs_ecg=128, fs_ppg=128, fs_eda=32, win_sec=60):
        """
        fs_ecg: ECG 샘플링 레이트(Hz)
        fs_ppg: PPG 샘플링 레이트(Hz)
        fs_eda: EDA 샘플링 레이트(Hz)
        win_sec: 윈도우 길이(초)
        """
        self.fs_ecg = fs_ecg
        self.fs_ppg = fs_ppg
        self.fs_eda = fs_eda
        self.win_ecg = win_sec * fs_ecg
        self.win_ppg = win_sec * fs_ppg
        self.win_eda = win_sec * fs_eda

    # ===========================================================
    # 1) 일반 모드: 신호 → 전처리 → 피크 검출 → 지표 추출 → 판정
    # ===========================================================
    def process_window(self, ecg, ppg, eda) -> Dict:
        # 1) 전처리
        ecg_f = preprocess_ecg(ecg, self.fs_ecg)
        ppg_f = preprocess_ppg(ppg, self.fs_ppg)
        eda_f = preprocess_eda(eda, self.fs_eda)

        # 2) 피크/foot 검출
        rpk = detect_r_peaks(ecg_f, self.fs_ecg)
        ppk = detect_ppg_peaks(ppg_f, self.fs_ppg)
        foot = detect_ppg_foot(ppg_f, self.fs_ppg)

        # 3) ECG 지표
        ibi = ibi_from_peaks(rpk, self.fs_ecg)
        HR   = hr_from_ibi(ibi)
        SDNN = sdnn(ibi)
        RMSSD= rmssd(ibi)
        LFHF = lfhf_ratio_from_rr(ibi, fs_rr=4.0)

        # 4) PPG 지표
        PTT  = ptt_from_pairs(rpk, foot, self.fs_ppg)

        # 5) EDA 지표
        SCL  = scl_level(eda_f)
        SCR  = scr_frequency(eda_f, self.fs_eda, thr_add=0.02)

        metrics = dict(HR=HR, SDNN=SDNN, RMSSD=RMSSD,
                       LFHF=LFHF, PTT=PTT, SCL=SCL, SCR=SCR)

        # 6) 휴리스틱 판정
        label, reason = classify_heuristic(metrics)

        return {
            "Label": label,
            "Reason": reason,
            **metrics,
            "R_peaks": rpk,
            "PPG_peaks": ppk,
            "PPG_feet": foot
        }

    # ===========================================================
    # 2) Ground Truth 모드: 합성기에서 제공하는 피크/foot 직접 사용
    # ===========================================================
    def process_window_with_peaks(self, ecg, ppg, eda, r_peaks, ppg_feet):
        """
        검출기를 거치지 않고, 합성 Ground Truth 피크를 직접 사용.
        """
        # EDA 전처리
        eda_f = preprocess_eda(eda, self.fs_eda)

        # ECG 지표
        ibi = ibi_from_peaks(r_peaks, self.fs_ecg)
        HR   = hr_from_ibi(ibi)
        SDNN = sdnn(ibi)
        RMSSD= rmssd(ibi)
        LFHF = lfhf_ratio_from_rr(ibi, fs_rr=4.0)

        # PTT (GT foot 사용)
        PTT  = ptt_from_pairs(r_peaks, ppg_feet, self.fs_ppg)

        # EDA 지표
        SCL  = scl_level(eda_f)
        SCR  = scr_frequency(eda_f, self.fs_eda, thr_add=0.02)

        metrics = dict(HR=HR, SDNN=SDNN, RMSSD=RMSSD,
                       LFHF=LFHF, PTT=PTT, SCL=SCL, SCR=SCR)

        label, reason = classify_heuristic(metrics)

        return {
            "Label": label,
            "Reason": reason,
            **metrics,
            "R_peaks": r_peaks,
            "PPG_feet": ppg_feet
        }
