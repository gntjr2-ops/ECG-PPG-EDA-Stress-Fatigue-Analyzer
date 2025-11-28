# demo.py
import numpy as np
from pipeline import HeuristicStressPipeline
from synth import synth_signals_with_gt

def safe_fmt(val, fmt="{:.3f}"):
    """None 값은 'N/A'로 안전하게 출력"""
    return fmt.format(val) if val is not None else "N/A"

if __name__ == "__main__":
    # 윈도우 길이를 60초 이상으로 → LF 성분이 잘 잡힘
    pipe = HeuristicStressPipeline(fs_ecg=128, fs_ppg=128, fs_eda=32, win_sec=60)

    for mode in ["normal", "stress", "fatigue"]:
        ecg, ppg, eda, rpk, feet, rr = synth_signals_with_gt(
            mode=mode, fs=128, fs_eda=32, win_sec=60, seed=7
        )

        res = pipe.process_window_with_peaks(ecg, ppg, eda, rpk, feet)

        print(f"\n=== MODE: {mode.upper()} ===")
        print(f"Label: {res['Label']} | Reason: {res['Reason']}")
        print(f"HR={safe_fmt(res.get('HR'), '{:.1f}')} BPM, "
              f"SDNN={safe_fmt(res.get('SDNN'))} s, "
              f"LF/HF={safe_fmt(res.get('LFHF'), '{:.2f}')}")
        print(f"PTT={safe_fmt(res.get('PTT'))} s, "
              f"SCL={safe_fmt(res.get('SCL'))}, "
              f"SCR={safe_fmt(res.get('SCR'))}")
