# heuristics.py
from typing import Dict, Tuple

DEFAULT_RULES = {
    "stress": {
        "hr_min": 85.0,          # BPM 이상
        "sdnn_max": 0.05,        # s 미만
        "lfhf_min": 0.0,         # 이상
        "ptt_max": 0.22,         # s 미만
        "scr_min": 0.05          # Hz 초과
    },
    "fatigue": {
        "hr_max": 65.0,          # BPM 이하
        "sdnn_min": 0.05,        # s 초과
        "lfhf_max": 1.5,         # 미만
        "ptt_min": 0.25,         # s 이상
        "scr_max": 0.03          # Hz 미만
    }
}

def classify_heuristic(metrics: Dict, rules: Dict = None) -> Tuple[str, str]:
    """
    metrics: dict {
      'HR','SDNN','LFHF','PTT','SCL','SCR'
    }
    return: (label, explanation)
    """
    if rules is None:
        rules = DEFAULT_RULES

    hr   = metrics.get("HR")
    sdnn = metrics.get("SDNN")
    lfhf = metrics.get("LFHF")
    ptt  = metrics.get("PTT")
    scl  = metrics.get("SCL")
    scr  = metrics.get("SCR")

    expl = []

    # Stress rule
    s = rules["stress"]
    stress_ok = True
    if hr is None or hr < s["hr_min"]:     stress_ok = False
    else: expl.append(f"HR={hr:.1f}≥{s['hr_min']}")

    if sdnn is None or sdnn >= s["sdnn_max"]: stress_ok = False
    else: expl.append(f"SDNN={sdnn:.3f}<{s['sdnn_max']}")

    if lfhf is None or lfhf <= s["lfhf_min"]: stress_ok = False
    else: expl.append(f"LF/HF={lfhf:.2f}>{s['lfhf_min']}")

    if ptt is None or ptt >= s["ptt_max"]: stress_ok = False
    else: expl.append(f"PTT={ptt:.3f}<{s['ptt_max']}")

    if scr is None or scr <= s["scr_min"]: stress_ok = False
    else: expl.append(f"SCR={scr:.3f}>{s['scr_min']}")

    if stress_ok:
        return "Stress", " & ".join(expl)

    # Fatigue rule
    expl = []
    f = rules["fatigue"]
    fatigue_ok = True
    if hr is None or hr > f["hr_max"]:     fatigue_ok = False
    else: expl.append(f"HR={hr:.1f}≤{f['hr_max']}")

    if sdnn is None or sdnn <= f["sdnn_min"]: fatigue_ok = False
    else: expl.append(f"SDNN={sdnn:.3f}>{f['sdnn_min']}")

    if lfhf is None or lfhf >= f["lfhf_max"]: fatigue_ok = False
    else: expl.append(f"LF/HF={lfhf:.2f}<{f['lfhf_max']}")

    if ptt is None or ptt <= f["ptt_min"]: fatigue_ok = False
    else: expl.append(f"PTT={ptt:.3f}>{f['ptt_min']}")

    if scr is None or scr >= f["scr_max"]: fatigue_ok = False
    else: expl.append(f"SCR={scr:.3f}<{f['scr_max']}")

    if fatigue_ok:
        return "Fatigue", " & ".join(expl)

    return "Normal", "조건 미충족 → 정상 범주로 판정"
