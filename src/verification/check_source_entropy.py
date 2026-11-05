"""Check source-entropy bounds against exact maximum-cycle entropy rates.
Run with Python and NumPy. Numerical checks supplement the analytic proof.
"""
import json
from pathlib import Path
import numpy as np

rng = np.random.default_rng(20261003)
checked = 0
for m in range(1, 4):
    size = 2 ** m
    contexts = np.array([[(w >> j) & 1 for j in range(m)] for w in range(size)])
    signs = 1 - 2 * contexts
    chars = np.array([np.prod(signs[:, [j for j in range(m) if a & (1 << j)]], axis=1) for a in range(size)])
    for scale in (0.001, 0.01, 0.05, 0.2, 0.49):
        for trial in range(30):
            q = 0.5 + rng.uniform(-scale, scale, size)
            trans = np.zeros((size, size))
            for w in range(size):
                trans[w, (2 * w) % size] = 1 - q[w]
                trans[w, (2 * w + 1) % size] = q[w]
            equations = trans.T - np.eye(size)
            equations[-1, :] = 1
            rhs = np.zeros(size)
            rhs[-1] = 1
            pi = np.linalg.solve(equations, rhs)
            c = float(np.max(np.abs(chars @ (pi * (1 - 2 * q)))))
            means = []
            def visit(start, current, seen, weight):
                for nxt in np.flatnonzero(trans[current]):
                    new_weight = weight + np.log2(trans[current, nxt])
                    if nxt == start:
                        means.append(new_weight / len(seen))
                    elif nxt not in seen and nxt >= start:
                        visit(start, int(nxt), seen | {int(nxt)}, new_weight)
            for start in range(size):
                visit(start, start, {start}, 0.0)
            rate = -max(means)
            upper = 1 - c * c / (2 * np.log(2))
            assert rate <= upper + 1e-10
            if (size - 1) * c < 1:
                lower = max(0, 1 - np.log2(1 + size * c / (1 - (size - 1) * c)))
                assert lower <= rate + 1e-10
            checked += 1
result = {'stationary_markov_chains_checked': checked, 'memories': [1, 2, 3], 'status': 'passed'}
Path(__file__).with_name('source-entropy-check-results.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result, indent=2))
