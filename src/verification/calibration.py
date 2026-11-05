"""Reproduce the finite-sample rejection witness; requires NumPy.

Run: python3 src/verification/calibration.py
GPU: python3 src/verification/calibration.py --backend torch --device cuda --batch-size 16
Install the optional PyTorch dependency in your GPU environment before using it.
The statistic computed here is a LOWER BOUND on the full C_2, not C_2 itself.
Large-N maxima located by zero-padded convolution are verified by integer sums.
The paper deliberately makes no algorithmic claim about computing full C_2.
"""
import itertools
import json
import platform
import argparse
from pathlib import Path

import numpy as np


def lag_sums(x):
    n = len(x)
    size = 1 << (2 * n - 1).bit_length()
    spectrum = np.fft.rfft(x.astype(float), n=size)
    values = np.fft.irfft(spectrum * spectrum.conj(), n=size)[1:n]
    rounded = np.rint(values)
    assert np.max(np.abs(values - rounded)) < 1e-5
    return rounded.astype(np.int64)


def exact_c2(x):
    """All shifts and all windows, via ranges of prefix sums for each lag."""
    best = 0
    for d in range(1, len(x)):
        prefix = np.r_[0, np.cumsum(x[:-d] * x[d:], dtype=np.int64)]
        best = max(best, int(prefix.max() - prefix.min()))
    return best


def verify_small():
    checked = 0
    for n in range(2, 10):
        for signs in itertools.product((-1, 1), repeat=n):
            x = np.array(signs, dtype=np.int64)
            expected = np.array([np.dot(x[:-d], x[d:]) for d in range(1, n)])
            actual = lag_sums(x)
            assert np.array_equal(actual, expected)
            assert np.max(np.abs(actual)) <= exact_c2(x)
            checked += 1
    return checked


def torch_lag_sums(x, device):
    import torch

    n = x.shape[-1]
    size = 1 << (2 * n - 1).bit_length()
    tensor = torch.as_tensor(x, dtype=torch.float64, device=device)
    spectrum = torch.fft.rfft(tensor, n=size, dim=-1)
    values = torch.fft.irfft(spectrum * spectrum.conj(), n=size, dim=-1)[..., 1:n]
    rounded = torch.round(values)
    if float(torch.max(torch.abs(values - rounded)).item()) >= 1e-5:
        raise ArithmeticError("Transform error exceeds the integer-rounding tolerance")
    return rounded.to(dtype=torch.int64).cpu().numpy()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--backend", choices=("numpy", "torch"), default="numpy")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto",
                        help="PyTorch: auto selects CUDA if available, otherwise CPU; MPS lacks float64")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--n", type=int, default=100_000)
    parser.add_argument("--trials", type=int, default=400)
    parser.add_argument("--seed", type=int, default=20260930)
    parser.add_argument("--threshold", type=float, default=0.01)
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("calibration-results.json"))
    args = parser.parse_args()
    if args.n < 2 or args.trials < 1 or args.batch_size < 1 or not 0 < args.threshold < 1:
        parser.error("Require n >= 2, trials and batch-size >= 1, and 0 < threshold < 1")
    device, backend_version = "cpu", np.__version__
    if args.backend == "torch":
        try:
            import torch
        except ImportError:
            parser.error("PyTorch is not installed. Install it in your GPU environment, or use --backend numpy.")
        device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
        if device == "cuda" and not torch.cuda.is_available():
            parser.error("CUDA is unavailable; use --device cpu or --backend numpy")
        backend_version = torch.__version__
        check_rng = np.random.Generator(np.random.PCG64(0))
        for check_n in (3, 8, 31, 128, 1000):
            batch = 2 * check_rng.integers(0, 2, size=(4, check_n), dtype=np.int64) - 1
            actual = torch_lag_sums(batch, device)
            assert np.array_equal(actual, np.stack([lag_sums(x) for x in batch]))
    elif args.device == "cuda":
        parser.error("--device cuda requires --backend torch")
    checks = verify_small()
    n, trials, seed = args.n, args.trials, args.seed
    rng = np.random.Generator(np.random.PCG64(seed))
    witnesses = []
    for start in range(0, trials, args.batch_size):
        size = min(args.batch_size, trials - start)
        batch = 2 * rng.integers(0, 2, size=(size, n), dtype=np.int64) - 1
        sums_batch = torch_lag_sums(batch, device) if args.backend == "torch" else [lag_sums(x) for x in batch]
        for offset, (x, sums) in enumerate(zip(batch, sums_batch)):
            d = int(np.argmax(np.abs(sums))) + 1
            integer_sum = int(np.dot(x[:-d], x[d:]))
            assert integer_sum == int(sums[d - 1])
            witnesses.append({"trial": start + offset, "lag": d, "sum": integer_sum})
    below = sum(abs(w["sum"]) / n < args.threshold for w in witnesses)
    output = {
        "statistic": "T=max_{1<=d<N}|sum_{i=1}^{N-d}x_i*x_{i+d}|/N <= C_2/N",
        "N": n, "trials": trials, "seed": seed, "rng": "PCG64",
        "numpy": np.__version__, "python": platform.python_version(),
        "backend": args.backend, "backend_version": backend_version,
        "device": device, "batch_size": args.batch_size, "threshold": args.threshold,
        "exhaustive_small_sequences_checked": checks,
        "witnesses_below_threshold": below,
        "minimum_witness": min(abs(w["sum"]) for w in witnesses) / n,
        "acceptance_probability_95pct_upper_if_zero": 1 - 0.05 ** (1 / trials) if below == 0 else None,
        "witnesses": witnesses,
    }
    path = args.output
    path.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({k: v for k, v in output.items() if k != "witnesses"}, indent=2))


if __name__ == "__main__":
    main()
