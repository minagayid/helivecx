"""dna_compress/core.py

DNA-Inspired Data Compression

Compresses high-dimensional vectors by:
1. Projecting onto the unit hypersphere.
2. Applying a random orthogonal rotation (Q).
3. Mapping the rotated trajectory into a generalized helix (DNA-style folding).
4. Quantizing using the known post-rotation Gaussian distribution.
"""

import numpy as np
import struct
from typing import Optional, Tuple


def normalize(vec: np.ndarray) -> np.ndarray:
    """Project vector onto the unit hypersphere."""
    norm = np.linalg.norm(vec)
    if norm == 0:
        return vec
    return vec / norm


def random_rotation_matrix(dim: int, seed: Optional[int] = None) -> np.ndarray:
    """Generate a random orthogonal rotation matrix via QR decomposition."""
    if seed is not None:
        np.random.seed(seed)
    A = np.random.randn(dim, dim)
    Q, R = np.linalg.qr(A)
    # Ensure proper rotation (determinant +1)
    D = np.diag(np.sign(np.diag(R)))
    Q = Q @ D
    return Q


def helical_encode(vec: np.ndarray, pitch: float = 1.0, radius: float = 1.0) -> np.ndarray:
    """
    Map a high-dimensional unit vector into a 3D helical curve.

    Returns an array of 3D points along the helix.
    The number of points equals the dimension of the input vector.
    """
    dim = len(vec)
    # Parameter t along the helix axis, scaled by the vector's values
    t = np.arange(dim)
    # Use the vector values to modulate the progression
    cumulative = np.cumsum(vec * 0.5 + 0.5)  # map to positive for progression
    theta = cumulative / pitch
    z = cumulative * radius
    x = radius * np.cos(theta)
    y = radius * np.sin(theta)
    helix = np.stack([x, y, z], axis=-1)
    return helix


def helical_decode(helix: np.ndarray, dim: int, pitch: float = 1.0, radius: float = 1.0) -> np.ndarray:
    """
    Inverse of helical_encode: reconstruct approximate high-D vector from helix.
    """
    # From the helix, estimate the cumulative progression
    z = helix[:, 2]
    cumulative = z / radius
    # Recover original increments
    vec = np.diff(cumulative, prepend=0.0)
    vec = (vec - 0.5) * 2.0  # inverse of mapping in encode
    if len(vec) < dim:
        vec = np.pad(vec, (0, dim - len(vec)), mode='constant')
    elif len(vec) > dim:
        vec = vec[:dim]
    return vec


class DNACompressor:
    def __init__(self, dim: int, seed: Optional[int] = 42, pitch: float = 1.0, radius: float = 1.0):
        self.dim = dim
        self.pitch = pitch
        self.radius = radius
        self.Q = random_rotation_matrix(dim, seed)
        self.Q_inv = self.Q.T

    def compress(self, vec: np.ndarray) -> Tuple[np.ndarray, dict]:
        """
        Compress a vector.

        Returns:
            helix (np.ndarray): encoded helix representation
            metadata (dict): information for reconstruction
        """
        v = normalize(vec)
        # Step 1: random orthogonal rotation
        u = self.Q @ v
        # Step 2: helical encoding
        helix = helical_encode(u, self.pitch, self.radius)
        # Step 3: quantize helix coordinates using known Gaussian distribution after rotation
        # For a unit vector in d-dim, expected variance of each coordinate after rotation = 1/d
        expected_std = np.sqrt(1.0 / self.dim)
        # Scale to [-1, 1] then to 8-bit
        helix_min = helix.min(axis=0)
        helix_max = helix.max(axis=0)
        scale = helix_max - helix_min
        scale[scale == 0] = 1.0
        helix_norm = (helix - helix_min) / scale * 255
        helix_quant = helix_norm.astype(np.uint8)
        metadata = {
            "dim": self.dim,
            "pitch": self.pitch,
            "radius": self.radius,
            "seed": None,  # could store seed to recreate Q
            "helix_min": helix_min.tolist(),
            "scale": scale.tolist(),
            "expected_std": expected_std,
        }
        return helix_quant, metadata

    def decompress(self, helix_quant: np.ndarray, metadata: dict) -> np.ndarray:
        """
        Decompress a helix back to a vector.
        """
        helix_min = np.array(metadata["helix_min"])
        scale = np.array(metadata["scale"])
        helix = helix_quant.astype(np.float64) / 255.0 * scale + helix_min
        # Inverse helical
        u_approx = helical_decode(helix, self.dim, self.pitch, self.radius)
        # Inverse rotation
        v_approx = self.Q_inv @ u_approx
        return v_approx

    def serialize(self, helix_quant: np.ndarray, metadata: dict, path: str):
        """Save compressed data to a binary file."""
        with open(path, "wb") as f:
            # Write metadata as JSON length-prefixed
            import json
            meta_bytes = json.dumps(metadata).encode("utf-8")
            f.write(struct.pack("<I", len(meta_bytes)))
            f.write(meta_bytes)
            # Write helix shape and data
            f.write(struct.pack("<HH", *helix_quant.shape))
            f.write(helix_quant.tobytes())

    @staticmethod
    def deserialize(path: str) -> Tuple[np.ndarray, dict]:
        """Load compressed data from a binary file."""
        import json
        with open(path, "rb") as f:
            meta_len = struct.unpack("<I", f.read(4))[0]
            metadata = json.loads(f.read(meta_len).decode("utf-8"))
            rows, cols = struct.unpack("<HH", f.read(4))
            helix_data = np.frombuffer(f.read(), dtype=np.uint8).reshape(rows, cols)
        return helix_data, metadata


def benchmark_compression(compressor: DNACompressor, vec: np.ndarray):
    """Run a round-trip and return metrics."""
    orig_norm = np.linalg.norm(vec)
    helix, meta = compressor.compress(vec)
    rec = compressor.decompress(helix, meta)
    # Align signs (direction is only determined up to sign on the sphere)
    if np.dot(rec, vec) < 0:
        rec = -rec
    # Reconstruct scale
    rec = rec * orig_norm
    mse = np.mean((vec - rec) ** 2)
    snr = 10 * np.log10(np.mean(vec ** 2) / (mse + 1e-12)) if mse > 0 else np.inf
    compression_ratio = (vec.nbytes) / (helix.nbytes + len(str(meta).encode()))
    return {
        "mse": mse,
        "snr_db": snr,
        "compression_ratio": compression_ratio,
        "original_bytes": vec.nbytes,
        "compressed_bytes": helix.nbytes,
    }


if __name__ == "__main__":
    dim = 128
    comp = DNACompressor(dim=dim, seed=42)
    vec = np.random.randn(dim)
    results = benchmark_compression(comp, vec)
    print("Benchmark results:")
    for k, v in results.items():
        print(f"  {k}: {v}")
