"""Necessary sample-size thresholds for the direct h_inf >= 0.99 certificate.

direct : Necessary condition obtained by summing the required context counts.
         Not sufficient for acceptance, and not a universal sample lower bound.
corr   : Exploratory calculation, excluded from the paper table: an *idealised*
         i.i.d.-type Hoeffding radius for the 2^m
         parity moments (+-1 summands); not proved for dependent data.
Run: python3 src/verification/certificate_size.py
"""
import json, math
from pathlib import Path

alpha = 0.05
beta_star = (2 ** 0.01 - 1) / 2


def direct(m):
    lo, hi = 1, 10 ** 13
    while lo < hi:
        mid = (lo + hi) // 2
        if math.sqrt(math.log(2 ** (m + 1) * mid / alpha) / (2 * mid / 2 ** m)) <= beta_star:
            hi = mid
        else:
            lo = mid + 1
    return lo


def corr(m):
    t = 2 ** 0.01 - 1
    u = t / (2 ** m + (2 ** m - 1) * t)          # F_m(u) = 0.99
    return 2 * math.log(2 ** (m + 1) / alpha) / u ** 2


rows = {m: {"direct": direct(m), "corr_idealised": round(corr(m))} for m in range(1, 9)}
for m, r in rows.items():
    print(m, "%.2e" % r["direct"], "%.2e" % r["corr_idealised"])
Path(__file__).with_name("certificate-size-results.json").write_text(
    json.dumps({"alpha": alpha, "beta_star": beta_star, "rows": rows}, indent=2) + "\n")
