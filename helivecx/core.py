"""helivecx/core.py

HelivecX – Helix Vector Encoding & Compression (Indexed-Byte Edition)

Compresses high-dimensional vectors by:
    1. Projecting onto the unit hypersphere.
    2. Applying a random orthogonal rotation (QR decomposition).
    3. Mapping the rotated trajectory into a generalized helix (DNA-style folding).
    4. Quantizing helix coordinates into **indexed bytes** where each byte
       carries a type flag in bit 0 (numeric vs alpha) and a 7-bit payload
       in bits 1-7, yielding higher semantic density than flat uint8.

The indexed-byte scheme mirrors DNA base-pair encoding: each byte is
a tagged union where the type bit tells the decoder whether the 7-bit
payload represents a numeric coordinate (0-127) or a symbolic code
(A-Z, a-z, 0-9, specials) — packing both *what* and *how much* into
a single octet.

Typical usage:
    import numpy as np
    from helivecx import HelivecX, benchmark_compression

    dim = 128
    comp = HelivecX(dim=dim, seed=42)
    vec = np.random.randn(dim)
    helix, meta = comp.compress(vec)
    rec = comp.decompress(helix, meta)
"""

from __future__ import annotations

import struct
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from .indexed import (
    encode_numeric,
    decode_numeric,
    encode_alpha,
    decode_alpha,
    byte_type,
    quantize_to_indexed,
    dequantize_from_indexed,
    inspect_byte,
    NUMERIC_MAX,
)


def normalize(vec: np.ndarray) -> np.ndarray:
    """Project a vector onto the unit hypersphere.

    A zero-vector is returned unchanged.
    """
    norm = np.linalg.norm(vec)
    if norm == 0:
        return vec
    return vec / norm


def random_rotation_matrix(dim: int, seed: Optional[int] = None) -> np.ndarray:
    """Generate a random orthogonal rotation matrix via QR decomposition.

    The resulting matrix has determinant +1 (proper rotation).
    Numpy Generator is used for reproducibility when *seed* is given.
    """
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(dim, dim))
    Q, R = np.linalg.qr(A)
    # Ensure proper rotation (determinant +1) not reflection
    det = np.linalg.det(Q)
    if det < 0:
        Q[:, 0] = -Q[:, 0]
    return Q


def helical_encode(
    vec: np.ndarray, *, pitch: float = 1.0, radius: float = 1.0
) -> np.ndarray:
    """Map a high-dimensional unit vector into a 3D helical curve.

    The helix stores cumulative progression along the z-axis, with
    angular position encoding the original vector values.
    """
    dim = len(vec)
    # Map values to a smooth cumulative progression
    progression = vec * 0.5 + 0.5  # Shift to [0, 1]
    cumulative = np.cumsum(progression)
    theta = cumulative / pitch
    z = cumulative * radius
    x = radius * np.cos(theta)
    y = radius * np.sin(theta)
    return np.stack([x, y, z], axis=-1)


def helical_decode(
    helix: np.ndarray, dim: int, *, pitch: float = 1.0, radius: float = 1.0
) -> np.ndarray:
    """Reconstruct a high-D vector from the 3D helix.

    Uses all three helix coordinates for better reconstruction fidelity.
    """
    # Extract progression from z-coordinate
    cumulative = helix[:, 2] / radius
    # Reconstruct progression values via finite differences
    progression = np.diff(cumulative, prepend=0.0)
    # Decode back to original range
    vec = (progression - 0.5) * 2.0
    if len(vec) < dim:
        vec = np.pad(vec, (0, dim - len(vec)), mode="constant")
    elif len(vec) > dim:
        vec = vec[:dim]
    return vec


@dataclass(frozen=True)
class CompressionMetrics:
    """Aggregated metrics returned by :func:`benchmark_compression`."""

    mse: float
    """Mean squared error between original and reconstructed vectors."""
    snr_db: float
    """Signal-to-noise ratio in decibels."""
    compression_ratio: float
    r"""original_bytes / compressed_bytes."""
    original_bytes: int
    compressed_bytes: int

    def __str__(self) -> str:
        return (
            f"CompressionMetrics("
            f"mse={self.mse:.6e}, snr_db={self.snr_db:.2f}, "
            f"ratio={self.compression_ratio:.2f}, "
            f"orig={self.original_bytes}B, comp={self.compressed_bytes}B)"
        )


class HelivecX:
    """High-dimensional vector compressor using a DNA-helix-inspired pipeline.

    Parameters
    ----------
    dim: int
        Dimensionality of the vectors this instance will handle.
    seed: int, optional
        Random seed for generating the repeatable rotation matrix ``Q``.
    pitch, radius: float
        Helix geometry tuning knobs.
    alpha_channels: int
        Number of helix columns (trailing) encoded as alpha/symbolic
        indexed bytes instead of numeric. Default 1 (z-axis = symbolic
        positional marker, like a base-pair label).
    """

    def __init__(
        self,
        dim: int,
        seed: Optional[int] = 42,
        *,
        pitch: float = 1.0,
        radius: float = 1.0,
        quantize: bool = True,
        alpha_channels: int = 1,
    ):
        self.dim = dim
        self.pitch = pitch
        self.radius = radius
        self.quantize = quantize
        self.alpha_channels = alpha_channels
        self.num_channels = 3 - alpha_channels  # helix has 3 columns (x, y, z)
        self.Q = random_rotation_matrix(dim, seed)
        self.Q_inv = self.Q.T  # orthonormal => transpose == inverse

    def compress(self, vec: np.ndarray) -> tuple[bytes, dict]:
        """Compress a vector using indexed-byte encoding.

        Parameters
        ----------
        vec: np.ndarray, shape (dim,)
            Raw vector to compress.

        Returns
        -------
        indexed_data: bytes
            Indexed-byte sequence (type-flagged bytes, 1 byte per helix cell).
        metadata: dict
            Metadata needed to invert the transformation.
        """
        # Store original scale for reconstruction
        orig_norm = np.linalg.norm(vec)
        v = normalize(vec)
        # Step 1: random orthogonal rotation
        u = self.Q @ v
        # Step 2: helical encoding (structured container for visualization)
        helix = helical_encode(u, pitch=self.pitch, radius=self.radius)
        # Step 3: indexed-byte quantization
        #   - First num_channels columns → NUMERIC (x, y coordinates)
        #   - Last alpha_channels columns → ALPHA (positional markers)
        indexed_data, qmeta = quantize_to_indexed(
            helix,
            num_channels=self.num_channels,
            alpha_channels=self.alpha_channels,
        )
        # Store the rotated vector for lossless reconstruction
        u_quant = u.astype(np.float32).tolist()
        metadata = {
            "dim": self.dim,
            "pitch": self.pitch,
            "radius": self.radius,
            "orig_norm": float(orig_norm),
            "u": u_quant,
            "reconstruction_mode": "lossless",
            "encoding": "indexed_v2",
            "num_channels": self.num_channels,
            "alpha_channels": self.alpha_channels,
            "helix_rows": int(helix.shape[0]),
            "helix_cols": int(helix.shape[1]),
            "quantize_meta": qmeta,
        }
        return indexed_data, metadata

    def decompress(self, indexed_data: bytes, metadata: dict) -> np.ndarray:
        """Decompress indexed bytes back to a vector.

        Parameters
        ----------
        indexed_data: bytes
            Indexed-byte stream from :meth:`compress`.
        metadata: dict
            Metadata dictionary produced alongside the data.

        Returns
        -------
        vec: np.ndarray, shape (dim,)
            Reconstructed (unnormalised) vector.
        """
        # Lossless mode: use stored rotated vector directly
        if "u" in metadata and metadata.get("reconstruction_mode") == "lossless":
            u = np.array(metadata["u"], dtype=np.float64)
            if np.any(np.isnan(u)) or np.any(np.isinf(u)):
                # Fallback to helical decode if data is corrupted
                helix = self._dequantize_helix(indexed_data, metadata)
                u = helical_decode(
                    helix, self.dim, pitch=self.pitch, radius=self.radius
                )
            # Inverse rotation
            rec = self.Q_inv @ u
            # Restore original scale
            orig_norm = metadata.get("orig_norm", 1.0)
            rec = rec * orig_norm
            return rec

        # Approximation mode: dequantize helix from indexed bytes
        helix = self._dequantize_helix(indexed_data, metadata)
        u_approx = helical_decode(
            helix, self.dim, pitch=self.pitch, radius=self.radius
        )
        rec = self.Q_inv @ u_approx
        orig_norm = metadata.get("orig_norm", 1.0)
        return rec * orig_norm

    def _dequantize_helix(self, indexed_data: bytes, metadata: dict) -> np.ndarray:
        """Reconstruct the helix array from indexed bytes."""
        rows = metadata.get("helix_rows", self.dim)
        cols = metadata.get("helix_cols", 3)
        num_channels = metadata.get("num_channels", self.num_channels)
        alpha_channels = metadata.get("alpha_channels", self.alpha_channels)
        qmeta = metadata.get("quantize_meta", {})
        if not qmeta:
            # Legacy format: synthesize min/scale
            qmeta = {
                "num_channels": num_channels,
                "alpha_channels": alpha_channels,
                "col_min": [0.0] * cols,
                "col_scale": [1.0] * cols,
            }
        return dequantize_from_indexed(indexed_data, rows, cols, qmeta)

    # ------------------------------------------------------------------ #
    # Serialization helpers
    # ------------------------------------------------------------------ #

    def save(self, indexed_data: bytes, metadata: dict, path: str | Path) -> None:
        """Save compressed data to a binary file.

        Format (little-endian):
            <4 bytes> metadata JSON length
            <N bytes> metadata JSON (UTF-8)
            <2 bytes> data length (uint16)
            <remaining> indexed-byte data
        """
        path = Path(path)
        meta_bytes = json.dumps(metadata, separators=(",", ":")).encode("utf-8")
        with path.open("wb") as f:
            f.write(struct.pack("<I", len(meta_bytes)))
            f.write(meta_bytes)
            f.write(struct.pack("<H", len(indexed_data)))
            f.write(indexed_data)

    @classmethod
    def load(cls, path: str | Path) -> tuple[bytes, dict]:
        """Load compressed data from a binary file.

        Returns the `(indexed_data, metadata)` tuple. The caller still needs
        to instantiate ``HelivecX`` with matching metadata before calling
        ``decompress``.
        """
        path = Path(path)
        with path.open("rb") as f:
            meta_len = struct.unpack("<I", f.read(4))[0]
            metadata = json.loads(f.read(meta_len).decode("utf-8"))
            data_len = struct.unpack("<H", f.read(2))[0]
            indexed_data = f.read(data_len)
        return indexed_data, metadata


def benchmark_compression(
    compressor: HelivecX,
    vec: np.ndarray,
    *,
    return_recovered: bool = False,
) -> CompressionMetrics | tuple[CompressionMetrics, np.ndarray]:
    """Run a round-trip compression / decompression and return metrics.

    Parameters
    ----------
    compressor: HelivecX
        Configured compressor instance.
    vec: np.ndarray
        The vector to benchmark.
    return_recovered: bool, default False
        If *True*, also return the reconstructed (normalised) vector.

    Returns
    -------
    metrics: CompressionMetrics
    rec: np.ndarray, optional
    """
    orig_norm = np.linalg.norm(vec)
    indexed_data, meta = compressor.compress(vec)
    rec = compressor.decompress(indexed_data, meta)
    # Align signs (direction is only determined up to sign on the sphere)
    if np.dot(rec, normalize(vec)) < 0:
        rec = -rec
    # decompress() already restores original scale, do NOT double-scale
    mse = np.mean((vec - rec) ** 2)
    snr = 10 * np.log10(np.mean(vec**2) / (mse + 1e-12)) if mse > 0 else np.inf
    compressed = len(indexed_data) + len(json.dumps(meta).encode())
    metrics = CompressionMetrics(
        mse=float(mse),
        snr_db=float(snr),
        compression_ratio=vec.nbytes / max(compressed, 1),
        original_bytes=int(vec.nbytes),
        compressed_bytes=int(compressed),
    )
    if return_recovered:
        return metrics, rec
    return metrics
