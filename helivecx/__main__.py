"""CLI entry-point for HelivecX.

Usage examples::

    python -m helivecx compress 128 input.npy output.helx
    python -m helivecx decompress output.helx reconstructed.npy
    python -m helivecx bench 128 --trials 100

"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

from .core import HelivecX, benchmark_compression


def _compress(args: argparse.Namespace) -> int:
    dim: int = args.dim
    vec = np.load(args.input)
    if vec.shape[0] != dim:
        print(f"Error: vector dimension {vec.shape[0]} != {dim}", file=sys.stderr)
        return 1
    comp = HelivecX(dim=dim, seed=args.seed)
    helix, meta = comp.compress(vec)
    comp.save(helix, meta, args.output)
    print(f"Compressed {args.input} -> {args.output}")
    return 0


def _decompress(args: argparse.Namespace) -> int:
    helix, meta = HelivecX.load(args.input)
    comp = HelivecX(dim=meta["dim"], seed=None)
    rec = comp.decompress(helix, meta)
    np.save(args.output, rec)
    print(f"Decompressed {args.input} -> {args.output}")
    return 0


def _bench(args: argparse.Namespace) -> int:
    dim: int = args.dim
    trials: int = args.trials
    comp = HelivecX(dim=dim, seed=42)
    mse_list, snr_list, ratio_list = [], [], []
    for _ in range(trials):
        vec = np.random.randn(dim)
        metrics = benchmark_compression(comp, vec)
        mse_list.append(metrics.mse)
        snr_list.append(metrics.snr_db)
        ratio_list.append(metrics.compression_ratio)

    print(f"Benchmark: {trials} trials, dim={dim}")
    print(f"  MSE   : {sum(mse_list)/len(mse_list):.6e}  (min={min(mse_list):.6e}, max={max(mse_list):.6e})")
    print(f"  SNR   : {sum(snr_list)/len(snr_list):.2f} dB  (min={min(snr_list):.2f}, max={max(snr_list):.2f})")
    print(f"  Ratio : {sum(ratio_list)/len(ratio_list):.2f} (min={min(ratio_list):.2f}, max={max(ratio_list):.2f})")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="helivecx", description="HelivecX CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    # compress
    p_compress = sub.add_parser("compress", help="Compress a .npy vector to .helx")
    p_compress.add_argument("dim", type=int)
    p_compress.add_argument("input", type=str)
    p_compress.add_argument("output", type=str)
    p_compress.add_argument("--seed", type=int, default=42)
    p_compress.set_defaults(func=_compress)

    # decompress
    p_decompress = sub.add_parser("decompress", help="Decompress a .helx to .npy")
    p_decompress.add_argument("input", type=str)
    p_decompress.add_argument("output", type=str)
    p_decompress.set_defaults(func=_decompress)

    # bench
    p_bench = sub.add_parser("bench", help="Run benchmark trials")
    p_bench.add_argument("dim", type=int)
    p_bench.add_argument("--trials", type=int, default=100)
    p_bench.set_defaults(func=_bench)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
