# synth.py
import numpy as np

def synth_rr_series(hr_bpm, win_sec, sdnn_target, lf_amp=0.02, hf_amp=0.01,
                    lf_hz=0.08, hf_hz=0.25, seed=0):
    """
    RR(=IBI) 시계열을 직접 합성.
    - 평균 RR = 60 / HR
    - 저주파/고주파 변조를 섞어 SDNN과 LF/HF 특성을 제어
    """
    rng = np.random.default_rng(seed)
    n_beats = int(round(hr_bpm/60.0 * win_sec))
    base_rr = 60.0 / hr_bpm  # 초

    t = np.linspace(0, win_sec, n_beats, endpoint=False)
    rr = (base_rr
          + lf_amp*np.sin(2*np.pi*lf_hz*t)     # LF 변조
          + hf_amp*np.sin(2*np.pi*hf_hz*t)     # HF 변조
          + rng.normal(scale=sdnn_target*0.25, size=n_beats))  # 미세 잡음

    rr = np.clip(rr, 0.3, 2.0)  # 안전범위
    return rr

def rr_to_peaks(rr, fs):
    """
    RR(초) 누적합 → R-peak 인덱스(샘플)
    """
    times = np.cumsum(rr)       # 각 박자 누적 시간
    times -= times[0]           # 0부터 시작
    idx = np.unique(np.maximum(0, np.round(times*fs).astype(int)))
    return idx

def synth_signals_with_gt(mode="normal", fs=128, fs_eda=32, win_sec=30, seed=42):
    """
    Ground Truth 포함 합성:
      - r_peaks: RR로부터 직접 생성
      - ppg_feet: r_peaks + delay(PTT)로 직접 지정
      - HR/SDNN/LFHF/PTT/SCR이 규칙을 만족하도록 파라미터 설계
    """
    rng = np.random.default_rng(seed)

    if mode == "stress":
        hr_bpm, ptt_s = 92, 0.20
        rr = synth_rr_series(hr_bpm, win_sec, sdnn_target=0.01,
                            lf_amp=0.05, hf_amp=0.001, seed=seed)  
        # ↑ LF 성분을 크게, HF는 작게 → LF/HF ↑
        scr_rate = 0.10
    elif mode == "fatigue":
        hr_bpm, ptt_s = 60, 0.28
        rr = synth_rr_series(hr_bpm, win_sec, sdnn_target=0.2,
                             lf_amp=0.030, hf_amp=0.030, seed=seed+1) # 변동↑(SDNN↑), LF/HF 낮춤
        scr_rate = 0.02
    else:  # normal
        hr_bpm, ptt_s = 75, 0.24
        rr = synth_rr_series(hr_bpm, win_sec, sdnn_target=0.07,
                             lf_amp=0.020, hf_amp=0.015, seed=seed+2) # 중간
        scr_rate = 0.04

    # --- Ground Truth peaks ---
    r_peaks = rr_to_peaks(rr, fs)
    delay = int(round(ptt_s * fs))
    ppg_feet = r_peaks + delay

    # --- ECG 합성 (R-피크를 가우시안으로) ---
    N = fs * win_sec
    ecg = np.zeros(N, dtype=np.float32)
    width = int(0.02*fs)
    k = np.arange(-width, width+1)
    gauss = np.exp(-0.5*(k/(0.007*fs))**2)
    for rp in r_peaks:
        if rp - width < 0 or rp + width >= N: continue
        ecg[rp-width:rp+width+1] += gauss
    ecg += 0.005*rng.standard_normal(N)

    # --- PPG 합성 (foot 이후 지수감쇠 맥파) ---
    ppg = np.zeros(N, dtype=np.float32)
    tail = int(0.30*fs)
    for ft in ppg_feet:
        if ft >= N: continue
        end = min(N, ft+tail)
        idx = np.arange(end-ft)
        wave = np.exp(-idx/(0.08*fs))
        ppg[ft:end] += wave.astype(np.float32)
    ppg = (ppg - ppg.mean()) / (ppg.std()+1e-8)
    ppg += 0.01*rng.standard_normal(N)

    # --- EDA 합성 (tonic + SCR 이벤트) ---
    N_eda = fs_eda * win_sec
    t_eda = np.arange(N_eda) / fs_eda
    eda = 0.5 + 0.02*np.sin(2*np.pi*0.01*t_eda) + 0.01*rng.standard_normal(N_eda)
    n_scr = int(max(1, scr_rate * win_sec))  # 약간 과장해 이벤트 생성
    for _ in range(n_scr):
        onset = rng.integers(0, N_eda - int(0.8*fs_eda))
        L = int(0.8*fs_eda)
        k = np.arange(L)
        scr = (1 - np.exp(-k/(0.15*fs_eda))) * np.exp(-k/(0.7*fs_eda))
        amp = 0.05 if mode=="stress" else (0.03 if mode=="normal" else 0.02)
        eda[onset:onset+L] += amp*scr[:L]

    return ecg, ppg, eda.astype(np.float32), r_peaks, ppg_feet, rr
