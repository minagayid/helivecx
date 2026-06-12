"""HelivecX - Helix Vector Encoding & Compression.

A DNA-inspired data compression algorithm for high-dimensional vectors.

Core pipeline:
    1. Hypersphere normalization
    2. Random orthogonal rotation (QR decomposition)
    3. Helical encoding (DNA-style folding)
    4. 8-bit quantization with metadata for reconstruction

Example:
    >>> import numpy as np
    >>> from helivecx import HelivecX
    >>> comp = HelivecX(dim=128)
    >>> helix, meta = comp.compress(np.random.randn(128))
    >>> rec = comp.decompress(helix, meta)
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

__all__ = [
    "HelivecX",
    "normalize",
    "random_rotation_matrix",
    "helical_encode",
    "helical_decode",
    "benchmark_compression",
    "CompressionMetrics",
]

__version__ = "1.0.0"
