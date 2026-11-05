"""Fair-bit rejection at the cutoff 0.01 also holds for the cyclic (FFT) autocorrelation.

Statistic: max over nonzero lags of |ACF|, linear (zero-padded, /N) and cyclic (/N).
Run: python3 src/verification/check_cyclic_acf.py
"""
import json
from pathlib import Path
import numpy as np

N, trials = 100000, 400
rng = np.random.Generator(np.random.PCG64(20261009))
lin, cyc = [], []
for _ in range(trials):
    x = 1 - 2 * rng.integers(0, 2, N)
    f = np.fft.rfft(x)
    cyc.append(np.max(np.abs(np.fft.irfft(f * np.conj(f), N)[1:])) / N)
    g = np.fft.rfft(x, 2 * N)
    lin.append(np.max(np.abs(np.fft.irfft(g * np.conj(g), 2 * N)[1:N])) / N)
res = {"N": N, "trials": trials, "seed": 20261009, "threshold": 0.01,
       "linear_min": float(min(lin)), "cyclic_min": float(min(cyc)),
       "linear_below_threshold": int(sum(v < 0.01 for v in lin)),
       "cyclic_below_threshold": int(sum(v < 0.01 for v in cyc))}
print(json.dumps(res, indent=2))
Path(__file__).with_name("cyclic-acf-results.json").write_text(json.dumps(res, indent=2) + "\n")
