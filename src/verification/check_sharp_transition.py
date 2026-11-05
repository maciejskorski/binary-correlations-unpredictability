"""Check sharp order-resolved transition bounds and stationary extremizers.
The proof is analytic; these finite checks independently verify its formulas.
"""
import itertools
import json
import math
from pathlib import Path
import numpy as np


def extremizer(m, bounds):
    states = np.array(list(itertools.product((-1, 1), repeat=m + 1)))
    mass = np.ones(len(states))
    for r in range(1, m + 2):
        for subset in itertools.combinations(range(m + 1), r):
            mass += (-1) ** (r - 1) * bounds[r - 1] * np.prod(states[:, subset], axis=1)
    mass /= 2 ** (m + 1)
    assert mass.min() > -1e-12
    assert abs(mass.sum() - 1) < 1e-12
    prefix = mass.reshape(2 ** m, 2).sum(axis=1)
    suffix = mass.reshape(2, 2 ** m).sum(axis=0)
    assert np.allclose(prefix, suffix, atol=1e-12)
    for r in range(1, m + 2):
        for subset in itertools.combinations(range(m + 1), r):
            moment = np.dot(mass, np.prod(states[:, subset], axis=1))
            assert abs(moment - (-1) ** (r - 1) * bounds[r - 1]) < 1e-11
    joint = mass.reshape(2 ** m, 2)
    biases = np.abs(joint[:, 1] / prefix - 0.5)
    u = sum(math.comb(m, r) * bounds[r - 1] for r in range(1, m + 1))
    v = sum(math.comb(m, r - 1) * bounds[r - 1] for r in range(1, m + 2))
    target = v / (2 * (1 - u))
    assert abs(biases[0] - target) < 1e-11
    assert abs(biases.max() - target) < 1e-11
    return target


rng = np.random.default_rng(20261008)
checked = 0
for m in range(1, 7):
    for trial in range(30):
        bounds = rng.uniform(0, 1, m + 1)
        total = sum(math.comb(m + 1, r) * bounds[r - 1] for r in range(1, m + 2))
        bounds *= rng.uniform(0.01, 0.99) / total
        extremizer(m, bounds)
        checked += 1
    for saturation in (0.001, 0.1, 0.9, 0.999):
        c = saturation / (2 ** (m + 1) - 1)
        delta = extremizer(m, np.full(m + 1, c))
        old = 2 ** (m - 1) * c / (1 - 2 ** (m - 1) * c)
        assert delta >= old - 1e-12
        if m >= 2:
            assert delta > old
        checked += 1
    bounds = np.zeros(m + 1)
    bounds[-1] = 0.2
    assert abs(extremizer(m, bounds) - 0.1) < 1e-11
    checked += 1
m, rho, epsilon = 4, 0.02, 0.001
improved = (rho + (2 ** m - 1) * epsilon) / (2 * (1 - (2 ** m - 1) * epsilon))
naive = min(0.5, 2 ** (m - 1) * rho / (1 - (2 ** m - 1) * rho))
prior = 2 ** (m - 1) * rho / (1 - 2 ** (m - 1) * rho)
result = {
    'status': 'passed', 'stationary_extremizers_checked': checked,
    'memories': list(range(1, 7)),
    'example': {'m': m, 'top_order_bound': rho, 'lower_order_bound': epsilon,
                'order_resolved_bound': improved, 'max_only_bound_clipped': naive,
                'prior_expression': prior},
    'scope': 'Sharp for local population moment profiles; no claim of sharpness for arbitrary finite sample maxima.'
}
Path(__file__).with_name('sharp-transition-results.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result, indent=2))
