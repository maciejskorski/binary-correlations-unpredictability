"""iid+trap chain: exact h_inf (max-plus iteration) vs closed form, F_m floor and quadratic ceiling.

Run: python3 src/verification/check_trap_chain.py
"""
import json, math, itertools
from pathlib import Path
import numpy as np


def chain(m, g):
    words = list(itertools.product([-1, 1], repeat=m + 1))
    p = {z: (1 - g) / 2 ** (m + 1) for z in words}
    p[(-1,) * (m + 1)] = (1 + (2 ** (m + 1) - 1) * g) / 2 ** (m + 1)
    return p


def h_inf(m, g, steps=3000):
    p = chain(m, g)
    ctx = list(itertools.product([-1, 1], repeat=m))
    idx = {c: i for i, c in enumerate(ctx)}
    pc = {c: p[c + (-1,)] + p[c + (1,)] for c in ctx}
    w = np.full((2 ** m, 2 ** m), -np.inf)
    for c in ctx:
        for y in (-1, 1):
            w[idx[c], idx[c[1:] + (y,)]] = math.log2(p[c + (y,)] / pc[c])
    v = np.zeros(2 ** m)
    for _ in range(steps):
        v = np.max(w + v[None, :], axis=1)
    return -v.max() / steps


def kappa(m, g):
    p = chain(m, g)
    best = 0.0
    for r in range(m + 1):
        for A in itertools.combinations(range(m), r):
            e = sum(q * z[m] * np.prod([z[j] for j in A]) for z, q in p.items())
            best = max(best, abs(e))
    return best


def F(m, u):
    if (2 ** m - 1) * u >= 1:
        return 0.0
    return max(0.0, 1 - math.log2(1 + 2 ** m * u / (1 - (2 ** m - 1) * u)))


rows = []
for m in (1, 2, 3, 4):
    for g in (1e-3, 1e-2, 3e-2):
        h = h_inf(m, g)
        closed = 1 - math.log2(1 + 2 ** m * g / (1 + (2 ** m - 1) * g))
        k = kappa(m, g)
        ok = abs(h - closed) < 2e-3 and abs(k - g) < 1e-12 and F(m, k) <= closed <= 1 - k * k / (2 * math.log(2))
        rows.append(dict(m=m, gamma=g, h_exact=round(h, 5), h_closed=round(closed, 5), kappa=round(k, 12), ok=bool(ok)))
        print(rows[-1])
assert all(r["ok"] for r in rows)
json.dump(rows, open(Path(__file__).with_name("trap-chain-results.json"), "w"), indent=1)


# Parities at all lags/orders: Walsh transform of the exact n-bit window law (n = 12).
def window_parities(m, g, n=12, sign=+1):
    p = chain(m, g)
    if sign < 0:  # extremizer: sign-flipped moments
        p = {z: 2 / 2 ** (m + 1) - q for z, q in p.items()}
    pc = {c: p[c + (-1,)] + p[c + (1,)] for c in itertools.product([-1, 1], repeat=m)}
    pmf = np.zeros(2 ** n)
    for i, bits in enumerate(itertools.product([0, 1], repeat=n)):
        x = tuple(1 - 2 * b for b in bits)
        pr = pc[x[:m]]
        for t in range(m, n):
            pr *= p[x[t - m:t] + (x[t],)] / pc[x[t - m:t]]
        pmf[i] = pr
    h = 1
    while h < len(pmf):  # fast Walsh-Hadamard: all E chi_A at once
        pmf = pmf.reshape(-1, 2, h)
        pmf = np.stack([pmf[:, 0] + pmf[:, 1], pmf[:, 0] - pmf[:, 1]], axis=1).reshape(-1)
        h *= 2
    return np.max(np.abs(pmf[1:]))


for m in (1, 2, 3, 4):
    for g in (0.01, 0.05, 0.2):
        assert window_parities(m, g) <= g + 1e-12, (m, g)
        if (2 ** (m + 1) - 1) * g < 1:
            assert window_parities(m, g, sign=-1) <= g + 1e-12, (m, g)
print("all parities in 12-bit windows <= gamma (trap chain and extremizer)")
