"""HelivecX - Helix Vector Encoding & Compression.

A DNA-inspired data compression algorithm for high-dimensional vectors,
using indexed-byte encoding where each byte carries a type flag (numeric
vs alpha) and a 7-bit payload for higher semantic density.

Core pipeline:
    1. Hypersphere normalization
    2. Random orthogonal rotation (QR decomposition)
    3. Helical encoding (DNA-style folding)
    4. Indexed-byte quantization (type-flagged 7-bit payloads)

The indexed-byte scheme mirrors DNA base-pair encoding: bit 0 indicates
whether the payload is a numeric coordinate (0) or a symbolic alpha code
(1), and bits 1-7 carry the value — much like how each base pair in DNA
encodes both the nucleotide type and its structural role.

Example:
    >>> import numpy as np
    >>> from helivecx import HelivecX
    >>> comp = HelivecX(dim=128)
    >>> data, meta = comp.compress(np.random.randn(128))
    >>> rec = comp.decompress(data, meta)
"""

from .core import (
    HelivecX,
    normalize,
    random_rotation_matrix,
    helical_encode,
    helical_decode,
    benchmark_compression,
    CompressionMetrics,
)
from .indexed import (
    encode_numeric,
    decode_numeric,
    encode_alpha,
    decode_alpha,
    byte_type,
    inspect_byte,
    encode_indexed,
    decode_indexed,
    quantize_to_indexed,
    dequantize_from_indexed,
    NUMERIC_MAX,
)

__all__ = [
    "HelivecX",
    "normalize",
    "random_rotation_matrix",
    "helical_encode",
    "helical_decode",
    "benchmark_compression",
    "CompressionMetrics",
    # Indexed-byte API
    "encode_numeric",
    "decode_numeric",
    "encode_alpha",
    "decode_alpha",
    "byte_type",
    "inspect_byte",
    "encode_indexed",
    "decode_indexed",
    "quantize_to_indexed",
    "dequantize_from_indexed",
    "NUMERIC_MAX",
]

__version__ = "2.0.0"
