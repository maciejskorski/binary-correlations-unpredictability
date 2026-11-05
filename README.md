[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3](https://img.shields.io/badge/python-3-blue.svg)](src/)

# Bridging Correlation Measures and Unpredictability

Supplementary material for *Bridging Correlation Measures and Unpredictability: Sharp Bounds and Insights for TRNG Validation* (Skorski & Günlü).

## Do small correlations certify high unpredictability?

Correlation measures (Mauduit–Sárközy) describe a *sequence*; the min-entropy rate $h_\infty$ describes the *law* that generated it.
We link the two for binary sources with memory $m$, and show that fair-coin correlation thresholds, recently proposed for testing true random number generators, are not well-suited for this task.

## Two main contributions

### 1. Correlations bound unpredictability

An *order-$r$ correlation* is the average parity (product of $\pm1$ bits) of $r$ bits at fixed lags: the bias for $r=1$, the autocorrelation for $r=2$.
Let $\gamma_r$ be the largest absolute order-$r$ correlation among the $m+1$ most recent bits (the memory window), and $\kappa_m=\max_r\gamma_r$.

**Correlations bound prediction.** The best next-bit advantage satisfies

$$\mathrm{Adv}\le\frac{\sum_{r=1}^{m+1}\binom{m}{r-1}\gamma_r}{2\bigl(1-\sum_{r=1}^{m}\binom{m}{r}\gamma_r\bigr)},$$

minimax optimal, with a sharp transition. It refutes a single-order bound used in prior TRNG-validation work.

**Correlations sandwich entropy.** The min-entropy deficit is linear in $\kappa_m$:

$$\frac{\kappa_m}{(m+1)\ln 2}-O(\kappa_m^2)\le 1-h_\infty\le\frac{2^m\kappa_m}{\ln 2}+O(\kappa_m^2).$$

Both constants are optimal: the *runs-of-ones* source attains the lower one, the *all-ones trap* source the upper one.

**Samples inherit the bounds.** Empirical parity biases are correlation sums, so the bounds transfer from stationary laws to samples (ergodic limits and finite-sample guarantees).

### 2. Fair-coin thresholds over-reject

The *noisy-parity* source $S_n=S_{n-d_1}\oplus\dots\oplus S_{n-d_{k-1}}\oplus\xi_n$ with noise bias $\delta$ has

$$h_\infty=1-\log_2(1+2\delta)\quad\text{yet}\quad\widehat C_k\ge 2\delta-o(1),$$

so the rule $C_k\le\Gamma\sqrt{2kN\ln N}$ rejects it with probability tending to one.

**Finite-sample limits.** At $N=10^5$, all 400 fair-bit samples violate $\widehat C_2<0.01$ (smallest witness $0.0109$). Conversely, certifying $h_\infty\ge0.99$ needs $1.5\cdot10^6$ to $1.5\cdot10^7$ samples ($\alpha=0.05$, $m=1,\dots,4$).

## Experiment: noisy parity over-rejection

Source $S_n=S_{n-1}\oplus\xi_n$ ($m=1$, $k=2$), $\varepsilon=0.01$, $\delta=(2^\varepsilon-1)/2$, so $h_\infty=0.99$.
The margin is $C_2/\sqrt{4N\ln N}$; a margin above $1$ means rejection.

| $N$ | runs | mean $C_2/N$ | mean margin | rejected |
|---|---|---|---|---|
| $10^5$ | 20 | 0.00779 | 0.36 | 0/20 |
| $10^6$ | 20 | 0.00694 | 0.93 | 5/20 |
| $10^7$ | 20 | 0.00702 | 2.77 | 20/20 |
| $10^8$ | 4 | 0.00689 | 8.02 | 4/4 |

The limit is $C_2/N\to2\delta\approx0.00696$. The effect is asymptotic: the margin crosses $1$ only around $N\approx10^6$.

**Reproduce** (seconds): run [`src/noisy_parity_overrejection.ipynb`](src/noisy_parity_overrejection.ipynb) or `python3 src/verification/noisy_parity.py`. The seed is fixed, and reruns match the stored `src/verification/noisy-parity-results.json` exactly.

## Code

- [`src/noisy_parity_overrejection.ipynb`](src/noisy_parity_overrejection.ipynb): the experiment above.
- [`src/proof_steps_check.ipynb`](src/proof_steps_check.ipynb): numerical checks of the key proof steps. Symbolic (Wolfram) counterparts, with stored outputs: [`proof_steps.wl`](src/verification/proof_steps.wl) and [`bridge_steps.wl`](src/verification/bridge_steps.wl) / [`.nb`](src/verification/bridge_steps.nb).
- [`src/verification/`](src/verification/): Python scripts with stored JSON results; run from the repository root, e.g. `python3 src/verification/calibration.py`.
  - Contribution 1: `check_bounds.py`, `check_sharp_transition.py`, `check_source_entropy.py`, `check_trap_chain.py`, `check_linear_gap.py`, `check_cycle_linear.py`, and `check_bridge.py` (sample-to-source bridge; sample maxima use lags up to 10, so they lower-bound $C_r$).
  - Contribution 2: `noisy_parity.py`, `calibration.py`, `check_cyclic_acf.py`, `certificate_size.py`.
