"""Random search: how small can (1 - h_inf)/kappa_m be for stationary memory-m chains?

h_inf is exact (maximum mean cycle, Karp); kappa_m as in main.tex.
Run: python3 src/verification/check_linear_gap.py
"""
import json, math
from pathlib import Path
import numpy as np

rng = np.random.default_rng(20261009)


def setup(m):
    size = 2 ** m
    signs = 1 - 2 * np.array([[(w >> j) & 1 for j in range(m)] for w in range(size)])
    chars = np.array([np.prod(signs[:, [j for j in range(m) if a & (1 << j)]], axis=1) for a in range(size)])
    return size, chars


def evaluate(q, size, chars):
    trans = np.zeros((size, size))
    for w in range(size):
        trans[w, (2 * w) % size] += 1 - q[w]
        trans[w, (2 * w + 1) % size] += q[w]
    a = trans.T - np.eye(size); a[-1, :] = 1
    rhs = np.zeros(size); rhs[-1] = 1
    pi = np.linalg.solve(a, rhs)
    kappa = np.max(np.abs(chars @ (pi * (1 - 2 * q))))
    w_mat = np.full((size, size), -np.inf)
    for w in range(size):
        for y, nxt in ((0, (2 * w) % size), (1, (2 * w + 1) % size)):
            p = 1 - q[w] if y == 0 else q[w]
            w_mat[w, nxt] = max(w_mat[w, nxt], math.log2(p))
    d = np.full((size + 1, size), -np.inf); d[0, :] = 0
    for k in range(1, size + 1):
        for v in range(size):
            d[k, v] = np.max(d[k - 1, :] + w_mat[:, v])
    best = max(min((d[size, v] - d[k, v]) / (size - k) for k in range(size) if d[k, v] > -np.inf)
               for v in range(size) if d[size, v] > -np.inf)
    return kappa, -best


out = {}
for m in (1, 2, 3):
    size, chars = setup(m)
    for scale in (0.02, 0.005, 0.001):
        best = np.inf
        for _ in range(1500):
            q = 0.5 + rng.uniform(-scale, scale, size) * rng.choice([0, 1], size)
            kappa, h = evaluate(q, size, chars)
            if kappa > 0:
                best = min(best, (1 - h) / kappa)
        out[f"m={m},scale={scale}"] = float(best)
print(json.dumps(out, indent=2))
Path(__file__).with_name("linear-gap-results.json").write_text(json.dumps(out, indent=2) + "\n")
