"""Noisy parity (k=2, m=1): margin of the lag-1 sum (<= C_2) over sqrt(2kN ln N). Run: python3 src/verification/noisy_parity.py"""
import json, numpy as np
from pathlib import Path
eps, k = 0.01, 2
d = (2**eps - 1) / 2
rng = np.random.default_rng(20261012)
out = {"eps": eps, "delta": d, "limit": 2**eps - 1, "seed": 20261012, "rows": []}
for N, T in [(10**5, 20), (10**6, 20), (10**7, 20), (10**8, 4)]:
    S = np.array([abs(int(N - 2 * (rng.random(N) < 0.5 - d).sum())) for _ in range(T)])
    m = S / np.sqrt(2 * k * N * np.log(N))
    out["rows"].append({"N": N, "trials": T, "sum_over_N_mean": float((S / N).mean()),
                        "sum_over_N_min": float((S / N).min()),
                        "margin_mean": float(m.mean()), "margin_min": float(m.min()), "rejected": int((m>1).sum())})
    print(out["rows"][-1])
json.dump(out, open(Path(__file__).with_name("noisy-parity-results.json"), "w"), indent=1)
