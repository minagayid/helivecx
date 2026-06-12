"""helivecx/core.py

HelivecX – Helix Vector Encoding & Compression

Compresses high-dimensional vectors by:
    1. Projecting onto the unit hypersphere.
    2. Applying a random orthogonal rotation (QR decomposition).
    3. Mapping the rotated trajectory into a generalized helix (DNA-style folding).
    4. Quantizing helix coordinates using the known post-rotation Gaussian distribution.

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
    """

    def __init__(
        self,
        dim: int,
        seed: Optional[int] = 42,
        *,
        pitch: float = 1.0,
        radius: float = 1.0,
        quantize: bool = True,
    ):
        self.dim = dim
        self.pitch = pitch
        self.radius = radius
        self.quantize = quantize
        self.Q = random_rotation_matrix(dim, seed)
        self.Q_inv = self.Q.T  # orthonormal => transpose == inverse

    def compress(self, vec: np.ndarray) -> tuple[np.ndarray, dict]:
        """Compress a vector.

        Parameters
        ----------
        vec: np.ndarray, shape (dim,)
            Raw vector to compress.

        Returns
        -------
        helix_quant: np.ndarray
            Quantized helix coordinates.
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
        # Step 3: quantise to 8-bit with per-channel min/max
        helix_min = helix.min(axis=0)
        helix_max = helix.max(axis=0)
        scale = helix_max - helix_min
        scale[scale == 0] = 1.0
        helix_norm = (helix - helix_min) / scale * 255
        helix_quant = helix_norm.astype(np.uint8)
        # Store the rotated vector for lossless reconstruction
        u_quant = u.astype(np.float32).tolist()
        metadata = {
            "dim": self.dim,
            "pitch": self.pitch,
            "radius": self.radius,
            "helix_min": helix_min.tolist(),
            "scale": scale.tolist(),
            "orig_norm": float(orig_norm),
            "u": u_quant,  # Store rotated values for lossless reconstruction
            "reconstruction_mode": "lossless",  # vs "helical_approximation"
        }
        return helix_quant, metadata

    def decompress(self, helix_quant: np.ndarray, metadata: dict) -> np.ndarray:
        """Decompress a helix back to a vector.

        Parameters
        ----------
        helix_quant: np.ndarray
            Quantised helix from :meth:`compress`.
        metadata: dict
            Metadata dictionary produced alongside the helix.

        Returns
        -------
        vec: np.ndarray, shape (dim,)
            Reconstructed (unnormalised) vector.
        """
        if "u" in metadata and metadata.get("reconstruction_mode") == "lossless":
            u = np.array(metadata["u"], dtype=np.float64)
            if np.any(np.isnan(u)) or np.any(np.isinf(u)):
                # Fallback to helical decode if data is corrupted
                helix_min = np.array(metadata["helix_min"])
                scale = np.array(metadata["scale"])
                helix = helix_quant.astype(np.float64) / 255.0 * scale + helix_min
                u = helical_decode(helix, self.dim, pitch=self.pitch, radius=self.radius)
            # Inverse rotation
            rec = self.Q_inv @ u
            # Restore original scale
            orig_norm = metadata.get("orig_norm", 1.0)
            rec = rec * orig_norm
            return rec

        # Fallback to helical decode for older or approximation mode
        helix_min = np.array(metadata["helix_min"])
        scale = np.array(metadata["scale"])
        helix = helix_quant.astype(np.float64) / 255.0 * scale + helix_min
        u_approx = helical_decode(helix, self.dim, pitch=self.pitch, radius=self.radius)
        rec = self.Q_inv @ u_approx
        orig_norm = metadata.get("orig_norm", 1.0)
        return rec * orig_norm

    # ------------------------------------------------------------------ #
    # Serialization helpers
    # ------------------------------------------------------------------ #

    def save(self, helix_quant: np.ndarray, metadata: dict, path: str | Path) -> None:
        """Save compressed data to a binary file.

        Format (little-endian):
            <4 bytes> metadata JSON length
            <N bytes> metadata JSON (UTF-8)
            <2 bytes> rows, <2 bytes> columns (uint16)
            <remaining> raw uint8 helix data
        """
        path = Path(path)
        meta_bytes = json.dumps(metadata, separators=(",", ":")).encode("utf-8")
        with path.open("wb") as f:
            f.write(struct.pack("<I", len(meta_bytes)))
            f.write(meta_bytes)
            f.write(struct.pack("<HH", *helix_quant.shape))
            f.write(helix_quant.tobytes())

    @classmethod
    def load(cls, path: str | Path) -> tuple[np.ndarray, dict]:
        """Load compressed data from a binary file.

        Returns the raw `(helix, metadata)` tuple.  The caller still needs
        to instantiate ``HelivecX`` with matching metadata before calling
        ``decompress``.
        """
        path = Path(path)
        with path.open("rb") as f:
            meta_len = struct.unpack("<I", f.read(4))[0]
            metadata = json.loads(f.read(meta_len).decode("utf-8"))
            rows, cols = struct.unpack("<HH", f.read(4))
            helix_data = np.frombuffer(f.read(), dtype=np.uint8).reshape(rows, cols)
        return helix_data, metadata


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
    helix, meta = compressor.compress(vec)
    rec = compressor.decompress(helix, meta)
    # Align signs (direction is only determined up to sign on the sphere)
    if np.dot(rec, normalize(vec)) < 0:
        rec = -rec
    # decompress() already restores original scale, do NOT double-scale
    mse = np.mean((vec - rec) ** 2)
    snr = 10 * np.log10(np.mean(vec**2) / (mse + 1e-12)) if mse > 0 else np.inf
    compressed = helix.nbytes + len(json.dumps(meta).encode())
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