"""Check the sample-to-source bridge (Theorem `thm:bridge`) numerically. NumPy only.
Run from the repository root: python3 src/verification/check_bridge.py
Sample maxima use lags in [0, DMAX] only, so they LOWER-bound the full C_r.
"""
import itertools, json, math
from pathlib import Path
import numpy as np

rng = np.random.default_rng(20261104)
out = {}

def subsets(n):
    return [B for r in range(1, n + 1) for B in itertools.combinations(range(n), r)]

def source(m, s, g=None):
    """Random memory-m source: g(w)=E[x_t|w] in [-s,s]; returns g, stationary pi over windows."""
    W = np.array(list(itertools.product([1, -1], repeat=m)))     # window (x_{t-1},...,x_{t-m})
    g = rng.uniform(-s, s, len(W)) if g is None else g
    idx = {tuple(w): i for i, w in enumerate(W)}
    P = np.zeros((len(W), len(W)))
    for i, w in enumerate(W):
        for y, p in ((1, (1 + g[i]) / 2), (-1, (1 - g[i]) / 2)):
            P[i, idx[(y,) + tuple(w[:-1])]] += p
    ev, evec = np.linalg.eig(P.T)
    pi = np.real(evec[:, np.argmin(abs(ev - 1))]); pi /= pi.sum()
    return W, g, pi, P

def biases(W, g, pi, m):
    b = {}
    for B in subsets(m + 1):
        A = [j - 1 for j in B if j > 0]
        chi = np.prod(W[:, A], axis=1) if A else np.ones(len(W))
        b[B] = float(np.sum(pi * chi * (g if 0 in B else 1)))
    return b

# 1. h_safe(u) >= 1 + log2(1 - 2^m u)
worst = 0
for m in range(1, 9):
    for v in np.linspace(1e-4, 0.5, 5000):
        u = v / 2 ** m
        hs = max(0, 1 - math.log2(1 + v / (1 - (2 ** m - 1) * u)))
        gap = hs - (1 + math.log2(1 - v)); assert gap > -1e-12; worst = max(worst, gap) if v <= .1 else worst
out["hsafe_lower_bound_max_gap_v<=0.1"] = worst

# 2. kappa_m <= 2Adv <= 2^{m/2} kappa_m on random sources
ratios = []
for m in (1, 2, 3):
    for s in (.02, .1, .4, .9):
        for _ in range(40):
            W, g, pi, _P = source(m, s)
            kap = max(abs(v) for v in biases(W, g, pi, m).values()); adv2 = float(np.sum(pi * abs(g)))  # 2Adv=E|g|
            assert kap <= adv2 + 1e-12 and adv2 <= 2 ** (m / 2) * kap + 1e-12
            ratios.append(adv2 / kap)
out["2Adv_over_kappa_range"] = [min(ratios), max(ratios)]

# 3. sample chain: C* >= (N-m)/N kappa-hat (deterministic) and C* vs 2Adv
def sample(P, m, n, W):
    st = 0; u = rng.random(n)
    cum = np.cumsum(P, axis=1); seq = np.empty(n, dtype=np.int64)
    for t in range(n):
        st = int(np.searchsorted(cum[st], u[t])); seq[t] = st
    return W[seq][:, 0].astype(float)         # newest symbol of each window

DMAX, N = 10, 100000
chain = []
for m in (1, 2):
    W, g, pi, P = source(m, .3)
    x = sample(P, m, N, W); n = N - m
    kb = {B: np.mean(np.prod([x[m - j: N - j] for j in B], axis=0)) for B in subsets(m + 1)}
    khat = max(abs(v) for v in kb.values())
    C = 0.0
    for r in range(1, m + 2):
        for D in itertools.combinations(range(DMAX + 1), r):
            M = N - max(D)
            C = max(C, np.max(np.abs(np.cumsum(np.prod([x[d:d + M] for d in D], axis=0)))) / N)
    kap = max(abs(v) for v in biases(W, g, pi, m).values()); adv2 = float(np.sum(pi * abs(g)))
    assert C >= (N - m) / N * khat - 1e-12
    chain.append(dict(m=m, kappa=kap, kappa_hat=khat, C_star_restricted=C, two_Adv=adv2, bound_2m2_kappa=2 ** (m / 2) * kap))
out["sample_chain_N=1e5"] = chain

# 4. mixing constant T = sum tau_k of the block chain; Doeblin interval coverage
def block_chain(m, p):             # p[w]: P(y=+1|w), w=(x_{t-1}..x_{t-m}); block=(x_t..x_{t-m})
    Z = np.array(list(itertools.product([1, -1], repeat=m + 1))); idx = {tuple(z): i for i, z in enumerate(Z)}
    Wi = {tuple(w): i for i, w in enumerate(itertools.product([1, -1], repeat=m))}
    P = np.zeros((len(Z), len(Z)))
    for i, z in enumerate(Z):
        pw = p[Wi[tuple(z[:m])]]
        for y, q in ((1, pw), (-1, 1 - pw)): P[i, idx[(y,) + tuple(z[:m])]] += q
    return P
def mixing_constant(P):
    T, Q = 0.0, np.eye(len(P))
    for _ in range(2000):
        tau = 1.0 if _ == 0 else 0.5 * max(np.abs(Q[i] - Q[j]).sum() for i in range(len(P)) for j in range(i))
        T += tau; Q = Q @ P
        if tau < 1e-13: break
    return T
Ts = []
for m in (1, 2, 3):
    for pm in (.5, .45, .4):
        p = rng.uniform(pm, 1 - pm, 2 ** m); p[0] = pm; T = mixing_constant(block_chain(m, p))
        assert m + 1 - 1e-9 <= T <= (m + 1) / (2 * pm) ** (m + 1) + 1e-9
        Ts.append(dict(m=m, pmin=pm, T=T, doeblin_bound=(m + 1) / (2 * pm) ** (m + 1)))
out["mixing_constant"] = Ts

alpha, m, pm, N, hits, ratio = .05, 1, .4, 20000, 0, []
W, g, pi, P = source(m, 0, g=np.array([.2, -.2]))          # transitions in [.4,.6]: Doeblin with pmin=.4
kap = max(abs(v) for v in biases(W, g, pi, m).values()); n = N - m
e = (m + 1) / (2 * pm) ** (m + 1) * (math.sqrt(2 * math.log(2 ** (m + 2) / alpha) / n) + 4 / n)
for _ in range(200):
    x = sample(P, m, N, W)
    kh = max(abs(np.mean(np.prod([x[m - j: N - j] for j in B], axis=0))) for B in subsets(m + 1))
    hits += abs(kh - kap) <= e; ratio.append(abs(kh - kap) / e)
out["interval_coverage"] = dict(m=m, pmin=pm, N=N, alpha=alpha, e=e, kappa=kap, coverage=hits / 200, mean_err_over_e=float(np.mean(ratio)))

# 5. AIS example (m=1, T=2, alpha=0.01): N so that khat+e <= v*/2^m with 1+log2(1-v*)... deficit 0.02
v = 1 - 2 ** -0.02; kmax = v / 2; T, L = 2, math.log(8 / 0.01)
xs = np.linspace(0, 8, 400001); phi = np.sqrt(2 / np.pi) * np.exp(-xs ** 2 / 2); Phi = np.interp(xs, xs[::100], [math.erf(t / math.sqrt(2)) for t in xs[::100]]); Emax3 = float(np.trapezoid(3 * xs * phi * Phi ** 2, xs))
out["AIS_0.98_m=1"] = dict(v_star=v, N_radius_only=(T * math.sqrt(2 * L) / kmax) ** 2,
                           N_typical_fair=((T * math.sqrt(2 * L) + Emax3) / kmax) ** 2, Emax_3abs_normals=Emax3)
Path("src/verification/bridge-results.json").write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=1)[:2500])
