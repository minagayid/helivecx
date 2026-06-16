# HelivecX – Helix Vector Encoding & Compression

A DNA-inspired data compression algorithm for high-dimensional vectors using **indexed-byte encoding**.

![HelivecX Logo](assets/logo.png)

## Overview

HelivecX compresses high-dimensional vectors by exploiting geometric transformations inspired by the double-helix structure of DNA. The pipeline is:

1. **Hypersphere Embedding** – project the vector onto the unit hypersphere.
2. **Random Orthogonal Rotation** – remove coordinate bias with a QR-based rotation (determinant +1).
3. **Helical Encoding** – fold the rotated trajectory into a generalised 3-D helix, preserving topological information.
4. **Indexed-Byte Quantisation** – each byte carries a **type flag** (bit 0: numeric vs alpha) and a **7-bit payload** (bits 1-7), yielding higher semantic density than flat uint8.

### Indexed-Byte Encoding

Each output byte is a tagged union:

```
[b7 b6 b5 b4 b3 b2 b1 b0]
 |--- value (7 bits) ---| type (1 bit)

bit 0 = 0  →  NUMERIC  : bits 1-7 encode value 0-127
bit 0 = 1  →  ALPHA    : bits 1-7 encode letter index (A-Z, a-z, 0-9, specials)
```

This mirrors DNA base-pair encoding: each byte carries both *what kind* of value it holds and *the value itself* — doubling the semantic density per byte compared to flat 8-bit quantisation.

| Encoding | Bits for value | Range | Semantic info |
|----------|----------------|-------|---------------|
| Flat uint8 | 8 | 0-255 | None (just a number) |
| **Indexed byte** | 7 | 0-127 | **+ type flag** (numeric or symbolic) |

The helix's x and y coordinates are encoded as **numeric** bytes, while the z-axis (cumulative progression) can be encoded as **alpha** bytes — storing symbolic base-pair-like markers alongside spatial coordinates.

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

# Compress & decompress (returns indexed bytes + metadata)
indexed_data, meta = comp.compress(vec)
recovered = comp.decompress(indexed_data, meta)

# Evaluate
metrics = benchmark_compression(comp, vec)
print(metrics)  # CompressionMetrics(mse=..., snr_db=..., ratio=...)
```

### Direct indexed-byte usage

```python
from helivecx import encode_numeric, decode_numeric, encode_alpha, decode_alpha, byte_type

# Numeric: value 42 → indexed byte
b = encode_numeric(42)      # → 84 (0b1010100|0)
assert byte_type(b) == "numeric"
assert decode_numeric(b) == 42

# Alpha: letter 'A' → indexed byte
b = encode_alpha('A')       # → 1 (0b0000000|1)
assert byte_type(b) == "alpha"
assert decode_alpha(b) == 'A'
```

## CLI

```bash
# Compress a .npy vector to .helx
python -m helivecx compress 128 input.npy output.helx

# Decompress back
python -m helivecx decompress output.helx reconstructed.npy

# Run a benchmark suite
python -m helivecx bench 128 --trials 1000

# Inspect indexed bytes in a compressed file
python -m helivecx inspect output.helx --bytes 30
```

## Benchmarks

Run the included benchmark script:

```bash
python scripts/run_benchmark.py --dim 128 --trials 100 --plot
```

Typical results for dim=128:
- **MSE**     : ~1e-3 (varies with quantisation)
- **SNR**     : ~30 dB
- **Ratio**   : ~2.5x (depends on dimensionality and metadata)

## Tests

```bash
pytest tests/ -v
```

## API Reference

### `HelivecX`

- `compress(vec)` → `(indexed_bytes, metadata)` — indexed-byte v2 encoding
- `decompress(indexed_bytes, metadata)` → `recovered_vec`
- `save(indexed_bytes, metadata, path)` – binary `.helx` format
- `HelivecX.load(path)` → `(indexed_bytes, metadata)`
- `benchmark_compression(instance, vec)` → `CompressionMetrics`

### Indexed-Byte API

- `encode_numeric(value)` → indexed byte (bit0=0)
- `decode_numeric(byte)` → int (0-127)
- `encode_alpha(char)` → indexed byte (bit0=1)
- `decode_alpha(byte)` → str (character)
- `byte_type(byte)` → "numeric" or "alpha"
- `inspect_byte(byte)` → dict with raw/type/value/char
- `quantize_to_indexed(arr, num_channels, alpha_channels)` → (bytes, meta)
- `dequantize_from_indexed(data, rows, cols, meta)` → ndarray

## License

MIT
