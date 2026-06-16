"""CLI entry-point for HelivecX (v2 indexed-byte).

Usage examples::

    python -m helivecx compress 128 input.npy output.helx
    python -m helivecx decompress output.helx reconstructed.npy
    python -m helivecx bench 128 --trials 100
    python -m helivecx inspect output.helx

"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

from .core import HelivecX, benchmark_compression
from .indexed import inspect_byte, byte_type


def _compress(args: argparse.Namespace) -> int:
    dim: int = args.dim
    vec = np.load(args.input)
    if vec.shape[0] != dim:
        print(f"Error: vector dimension {vec.shape[0]} != {dim}", file=sys.stderr)
        return 1
    comp = HelivecX(dim=dim, seed=args.seed, alpha_channels=args.alpha_channels)
    indexed_data, meta = comp.compress(vec)
    comp.save(indexed_data, meta, args.output)
    print(f"Compressed {args.input} -> {args.output}")
    print(f"  Indexed bytes: {len(indexed_data)}")
    print(f"  Alpha channels: {meta['alpha_channels']}")
    return 0


def _decompress(args: argparse.Namespace) -> int:
    indexed_data, meta = HelivecX.load(args.input)
    comp = HelivecX(dim=meta["dim"], seed=None, alpha_channels=meta.get("alpha_channels", 1))
    rec = comp.decompress(indexed_data, meta)
    np.save(args.output, rec)
    print(f"Decompressed {args.input} -> {args.output}")
    return 0


def _bench(args: argparse.Namespace) -> int:
    dim: int = args.dim
    trials: int = args.trials
    comp = HelivecX(dim=dim, seed=42, alpha_channels=args.alpha_channels)
    mse_list, snr_list, ratio_list = [], [], []
    for _ in range(trials):
        vec = np.random.randn(dim)
        metrics = benchmark_compression(comp, vec)
        mse_list.append(metrics.mse)
        snr_list.append(metrics.snr_db)
        ratio_list.append(metrics.compression_ratio)

    print(f"Benchmark: {trials} trials, dim={dim}, alpha_channels={args.alpha_channels}")
    print(f"  MSE   : {sum(mse_list)/len(mse_list):.6e}  (min={min(mse_list):.6e}, max={max(mse_list):.6e})")
    print(f"  SNR   : {sum(snr_list)/len(snr_list):.2f} dB  (min={min(snr_list):.2f}, max={max(snr_list):.2f})")
    print(f"  Ratio : {sum(ratio_list)/len(ratio_list):.2f} (min={min(ratio_list):.2f}, max={max(ratio_list):.2f})")
    return 0


def _inspect(args: argparse.Namespace) -> int:
    """Inspect indexed bytes in a .helx file."""
    indexed_data, meta = HelivecX.load(args.input)
    print(f"File: {args.input}")
    print(f"  Encoding: {meta.get('encoding', 'unknown')}")
    print(f"  Dim: {meta.get('dim', '?')}")
    print(f"  Num channels: {meta.get('num_channels', '?')}")
    print(f"  Alpha channels: {meta.get('alpha_channels', '?')}")
    print(f"  Indexed bytes: {len(indexed_data)}")

    # Show first N bytes
    n = min(args.bytes, len(indexed_data))
    print(f"\nFirst {n} indexed bytes:")
    for i in range(n):
        info = inspect_byte(indexed_data[i])
        if info["type"] == "numeric":
            print(f"  [{i:4d}] {info['raw']:3d} = NUMERIC  value={info['value']}")
        else:
            print(f"  [{i:4d}] {info['raw']:3d} = ALPHA    char='{info['char']}' idx={info['value']}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="helivecx", description="HelivecX CLI (indexed-byte v2)")
    sub = parser.add_subparsers(dest="command", required=True)

    # compress
    p_compress = sub.add_parser("compress", help="Compress a .npy vector to .helx")
    p_compress.add_argument("dim", type=int)
    p_compress.add_argument("input", type=str)
    p_compress.add_argument("output", type=str)
    p_compress.add_argument("--seed", type=int, default=42)
    p_compress.add_argument("--alpha-channels", type=int, default=1,
                            help="Number of helix columns to encode as alpha (default: 1)")
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
    p_bench.add_argument("--alpha-channels", type=int, default=1)
    p_bench.set_defaults(func=_bench)

    # inspect
    p_inspect = sub.add_parser("inspect", help="Inspect indexed bytes in a .helx file")
    p_inspect.add_argument("input", type=str)
    p_inspect.add_argument("--bytes", type=int, default=20,
                           help="Number of indexed bytes to show (default: 20)")
    p_inspect.set_defaults(func=_inspect)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
