# HelivecX – Helix Vector Encoding & Compression

A DNA-inspired data compression algorithm for high-dimensional vectors.

## Overview

HelivecX compresses high-dimensional vectors by exploiting geometric transformations inspired by the double-helix structure of DNA. The pipeline is:

1. **Hypersphere Embedding** – project the vector onto the unit hypersphere.
2. **Random Orthogonal Rotation** – remove coordinate bias with a QR-based rotation (determinant +1).
3. **Helical Encoding** – fold the rotated trajectory into a generalised 3-D helix, preserving topological information.
4. **Quantisation** – store helix coordinates as 8-bit integers, yielding a compact, lossy representation with a well-defined reconstruction path.

## Installation

```bash
cd helivecx
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Requires Python **>= 3.10**.

## Quick Start (Python)

```python
import numpy as np
from helivecx import HelivecX, benchmark_compression

dim = 128
comp = HelivecX(dim=dim, seed=42)
vec = np.random.randn(dim)

# Compress & decompress
helix, meta = comp.compress(vec)
recovered = comp.decompress(helix, meta)

# Evaluate
metrics = benchmark_compression(comp, vec)
print(metrics)  # CompressionMetrics(mse=..., snr_db=..., ratio=...)
```

## CLI

```bash
# Compress a .npy vector to .helx
python -m helivecx compress 128 input.npy output.helx

# Decompress back
python -m helivecx decompress output.helx reconstructed.npy

# Run a benchmark suite
python -m helivecx bench 128 --trials 1000
```

## Benchmarks

Run the included benchmark script:

```bash
python scripts/run_benchmark.py --dim 128 --trials 100 --plot
```

Typical results for dim=128:
- **MSE**     : ~1e-3 (varies with quantisation)
- **SNR**     : ~30 dB
- **Ratio**   : ~2.5× (depends on dimensionality and metadata)

## Tests

```bash
pytest tests/ -v
```

## API Reference

### `HelivecX`

- `compress(vec)` → `(helix_quant, metadata)`
- `decompress(helix_quant, metadata)` → `recovered_vec`
- `save(helix, metadata, path)` – binary `.helx` format
- `HelivecX.load(path)` → `(helix, metadata)`
- `benchmark_compression(instance, vec)` → `CompressionMetrics`

## TODO / Future Work

- [ ] Support batched compression (matrix of vectors)
- [ ] Explore non-linear helical parameterisations
- [ ] Optional lossless mode (store full float32 instead of 8-bit)
- [ ] CUDA / Numba speed-ups for large dimensions

## License

MIT
