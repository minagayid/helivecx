"""scripts/run_benchmark.py

Run the HelivecX benchmark and optionally visualise results.
"""
from __future__ import annotations

import argparse
import os
import warnings

import numpy as np

from helivecx import HelivecX, benchmark_compression


def main() -> None:
    parser = argparse.ArgumentParser(description="HelivecX benchmark")
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--plot", action="store_true", help="Save benchmark plot")
    args = parser.parse_args()

    dim = args.dim
    rng = np.random.default_rng(args.seed)
    comp = HelivecX(dim=dim, seed=args.seed)

    mse_list, snr_list, ratio_list = [], [], []
    print(f"Running {args.trials} trials, dim={dim} ...")
    for _ in range(args.trials):
        vec = rng.normal(size=dim)
        metrics = benchmark_compression(comp, vec)
        mse_list.append(metrics.mse)
        snr_list.append(metrics.snr_db)
        ratio_list.append(metrics.compression_ratio)

    print(f"\nHelivecX Benchmark Results")
    print("=" * 40)
    print(f"  Dimension         : {dim}")
    print(f"  Trials            : {args.trials}")
    print(f"  MSE (mean)        : {sum(mse_list)/len(mse_list):.6e}")
    print(f"  MSE (min/max)     : {min(mse_list):.6e} / {max(mse_list):.6e}")
    print(f"  SNR  (mean)       : {sum(snr_list)/len(snr_list):.2f} dB")
    print(f"  SNR  (min/max)    : {min(snr_list):.2f} / {max(snr_list):.2f} dB")
    print(f"  Ratio (mean)      : {sum(ratio_list)/len(ratio_list):.2f}")
    print(f"  Ratio (min/max)   : {min(ratio_list):.2f} / {max(ratio_list):.2f}")

    if args.plot:
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            warnings.warn("matplotlib not installed; skipping plot.  Install with 'pip install matplotlib'.")
            return

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        axes[0].hist(mse_list, bins=30, color="C0", edgecolor="black")
        axes[0].set_title("MSE Distribution")
        axes[0].set_xlabel("MSE")

        axes[1].hist(snr_list, bins=30, color="C1", edgecolor="black")
        axes[1].set_title("SNR Distribution (dB)")
        axes[1].set_xlabel("SNR (dB)")

        axes[2].hist(ratio_list, bins=30, color="C2", edgecolor="black")
        axes[2].set_title("Compression Ratio Distribution")
        axes[2].set_xlabel("Ratio")

        plt.tight_layout()
        outfile = os.path.expanduser("~/Desktop/helivecx/benchmark.png")
        plt.savefig(outfile)
        print(f"Plot saved to {outfile}")


if __name__ == "__main__":
    main()
