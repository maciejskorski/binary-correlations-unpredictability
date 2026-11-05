"""Independent finite checks of the prediction, Maurer, and entropy formulas.

These checks supplement, rather than replace, the analytic proofs in main.tex.
Run: python3 src/verification/check_bounds.py
"""
import itertools
import json
import math
from pathlib import Path

import numpy as np


def check_prediction():
    rng = np.random.default_rng(9182)
    checked = 0
    for m in range(1, 6):
        states = np.array(list(itertools.product((-1, 1), repeat=m)))
        chars = np.array([
            np.prod(states[:, [j for j in range(m) if mask & (1 << j)]], axis=1)
            for mask in range(1 << m)
        ])
        for _ in range(200):
            joint = rng.dirichlet(np.full(2 ** (m + 1), 0.3)).reshape(-1, 2)
            p = joint.sum(axis=1)
            h = joint[:, 1] - joint[:, 0]
            g = h / p
            a, b = chars @ p, chars @ h
            adv = np.abs(h).sum() / 2
            assert np.max(np.abs(b)) / 2 <= adv + 1e-12
            assert adv <= min(1, np.linalg.norm(b)) / 2 + 1e-12
            assert np.allclose(chars.T @ b, (chars.T @ a) * g, atol=1e-12)
            a1, b1 = np.abs(a[1:]).sum(), np.abs(b).sum()
            if a1 < 1:
                assert np.max(np.abs(g)) / 2 <= b1 / (2 * (1 - a1)) + 1e-12
            checked += 1
    return checked


def check_maurer():
    checked = 0
    for k, total in ((1, 8), (2, 5)):
        for blocks in itertools.product(range(2 ** k), repeat=total):
            last, gaps = {}, []
            for n, value in enumerate(blocks, start=1):
                gaps.append(n - last.get(value, 0))
                last[value] = n
            for r in range(1, total):
                length = total - r
                statistic = sum(math.log2(a) for a in gaps[r:]) / length
                counts = np.bincount(blocks[r:], minlength=2 ** k)
                p = counts / length
                entropy = -sum(v * math.log2(v) for v in p if v)
                upper = entropy + math.log2(total / length)
                pinsker = k + math.log2(total / length) - np.abs(p - 2 ** -k).sum() ** 2 / (2 * math.log(2))
                assert statistic <= upper + 1e-12 <= pinsker + 2e-12
                for t in range(2, r + 2):
                    collisions = sum(
                        sum(blocks[n] == blocks[n - i] for n in range(r, total)) / length
                        for i in range(1, t)
                    )
                    lower = max(0, 1 - collisions) * math.log2(t)
                    assert lower <= statistic + 1e-12
                checked += 1
    return checked


def check_noise_entropy():
    checked = 0
    for lags in ((1,), (1, 3), (1, 2, 4)):
        m, n, delta = max(lags), 10, 0.05
        probabilities = []
        for bits in itertools.product((0, 1), repeat=n):
            probability = 2 ** -m
            for i in range(m, n):
                noise = bits[i]
                for d in lags:
                    noise ^= bits[i - d]
                probability *= 0.5 + delta if noise == 0 else 0.5 - delta
            probabilities.append(probability)
        assert abs(sum(probabilities) - 1) < 1e-12
        actual = -math.log2(max(probabilities))
        expected = m + (n - m) * (1 - math.log2(1 + 2 * delta))
        assert abs(actual - expected) < 1e-12
        checked += len(probabilities)
    return checked


if __name__ == "__main__":
    result = {
        "prediction_distributions": check_prediction(),
        "maurer_block_sequences_and_initializations": check_maurer(),
        "noisy_parity_paths": check_noise_entropy(),
        "status": "passed",
    }
    Path(__file__).with_name("bounds-check-results.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
