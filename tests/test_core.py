"""tests/test_core.py

Comprehensive tests for the HelivecX compression system (v2 indexed-byte).
"""
import numpy as np
import pytest

from helivecx.core import (
    HelivecX,
    normalize,
    random_rotation_matrix,
    benchmark_compression,
    CompressionMetrics,
)
from helivecx.indexed import (
    encode_numeric,
    decode_numeric,
    encode_alpha,
    decode_alpha,
    byte_type,
    encode_indexed,
    decode_indexed,
    quantize_to_indexed,
    dequantize_from_indexed,
    inspect_byte,
    NUMERIC_MAX,
)


# --------------------------------------------------------------------------- #
# Normalisation
# --------------------------------------------------------------------------- #

class TestNormalize:
    def test_zero_vector(self):
        assert np.array_equal(normalize(np.zeros(5)), np.zeros(5))

    def test_unit_length(self):
        v = np.array([3.0, 4.0])
        n = normalize(v)
        assert np.isclose(np.linalg.norm(n), 1.0)

    def test_idempotent(self):
        v = np.random.randn(50)
        n1 = normalize(v)
        n2 = normalize(n1)
        assert np.allclose(n1, n2)


# --------------------------------------------------------------------------- #
# Rotation
# --------------------------------------------------------------------------- #

class TestRotation:
    def test_orthogonality(self):
        Q = random_rotation_matrix(20, seed=7)
        assert np.allclose(Q @ Q.T, np.eye(20))

    def test_determinant(self):
        Q = random_rotation_matrix(20, seed=7)
        assert np.isclose(np.linalg.det(Q), 1.0)

    def test_norm_preservation(self):
        Q = random_rotation_matrix(20, seed=11)
        v = np.random.randn(20)
        u = Q @ v
        assert np.isclose(np.linalg.norm(v), np.linalg.norm(u))

    def test_seed_reproducibility(self):
        Q1 = random_rotation_matrix(16, seed=123)
        Q2 = random_rotation_matrix(16, seed=123)
        assert np.array_equal(Q1, Q2)


# --------------------------------------------------------------------------- #
# Indexed-byte encoding
# --------------------------------------------------------------------------- #

class TestIndexedByte:
    """Test the indexed-byte encoding scheme (bit 0 = type flag)."""

    def test_numeric_roundtrip(self):
        for val in [0, 1, 63, 127]:
            encoded = encode_numeric(val)
            assert byte_type(encoded) == "numeric"
            assert decode_numeric(encoded) == val

    def test_alpha_roundtrip(self):
        for ch in ["A", "Z", "a", "z", "0", "9", "_"]:
            encoded = encode_alpha(ch)
            assert byte_type(encoded) == "alpha"
            assert decode_alpha(encoded) == ch

    def test_type_flag_numeric(self):
        """All numeric bytes must have bit 0 = 0."""
        for val in range(128):
            encoded = encode_numeric(val)
            assert encoded & 1 == 0, f"Numeric {val} has bit0=1"

    def test_type_flag_alpha(self):
        """All alpha bytes must have bit 0 = 1."""
        for ch in "AZaz09_":
            encoded = encode_alpha(ch)
            assert encoded & 1 == 1, f"Alpha '{ch}' has bit0=0"

    def test_numeric_out_of_range(self):
        with pytest.raises(ValueError):
            encode_numeric(-1)
        with pytest.raises(ValueError):
            encode_numeric(128)

    def test_inspect_byte_numeric(self):
        info = inspect_byte(encode_numeric(42))
        assert info["type"] == "numeric"
        assert info["value"] == 42

    def test_inspect_byte_alpha(self):
        info = inspect_byte(encode_alpha("A"))
        assert info["type"] == "alpha"
        assert info["char"] == "A"

    def test_encode_decode_indexed_list(self):
        values = [10, 20, 30, 5]
        types = ["numeric", "numeric", "alpha", "numeric"]
        data = encode_indexed(values, types)
        assert isinstance(data, bytes)
        dec_vals, dec_types = decode_indexed(data)
        assert dec_types == ["numeric", "numeric", "alpha", "numeric"]
        assert dec_vals[0] == 10
        assert dec_vals[1] == 20
        assert dec_vals[3] == 5

    def test_quantize_dequantize_indexed(self):
        arr = np.array([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]])
        data, meta = quantize_to_indexed(arr, num_channels=2, alpha_channels=1)
        recovered = dequantize_from_indexed(data, 2, 3, meta)
        # Should reconstruct approximately (7-bit quantization has error)
        assert recovered.shape == (2, 3)
        assert np.allclose(recovered, arr, atol=0.1)


# --------------------------------------------------------------------------- #
# HelivecX round-trip (indexed-byte v2)
# --------------------------------------------------------------------------- #

class TestHelivecX:
    def test_compress_returns_bytes(self):
        """v2 compress returns bytes (indexed), not ndarray."""
        dim = 128
        comp = HelivecX(dim=dim, seed=42)
        vec = np.random.randn(dim)
        indexed_data, meta = comp.compress(vec)
        assert isinstance(indexed_data, bytes)
        assert meta["encoding"] == "indexed_v2"

    def test_roundtrip_correlation(self):
        dim = 128
        comp = HelivecX(dim=dim, seed=42)
        vec = np.random.randn(dim)
        indexed_data, meta = comp.compress(vec)
        rec = comp.decompress(indexed_data, meta)
        v_norm = normalize(vec)
        if np.dot(rec, v_norm) < 0:
            rec = -rec
        corr = np.dot(rec, v_norm)
        assert corr > 0.85

    def test_serialization_roundtrip(self, tmp_path):
        dim = 64
        comp = HelivecX(dim=dim, seed=99)
        vec = np.random.randn(dim)
        indexed_data, meta = comp.compress(vec)

        path = tmp_path / "test.helx"
        comp.save(indexed_data, meta, str(path))
        indexed_data2, meta2 = HelivecX.load(str(path))
        assert indexed_data == indexed_data2
        assert meta == meta2

    def test_different_dimensions(self):
        for dim in [8, 16, 32, 64, 128, 256]:
            comp = HelivecX(dim=dim, seed=0)
            vec = np.random.randn(dim)
            indexed_data, meta = comp.compress(vec)
            rec = comp.decompress(indexed_data, meta)
            assert rec.shape == (dim,)

    def test_alpha_channels_config(self):
        """alpha_channels=2 means 2 helix columns encoded as alpha."""
        comp = HelivecX(dim=64, seed=0, alpha_channels=2)
        vec = np.random.randn(64)
        indexed_data, meta = comp.compress(vec)
        assert meta["alpha_channels"] == 2
        assert meta["num_channels"] == 1

    def test_benchmark_returns_metrics(self):
        comp = HelivecX(dim=128, seed=0)
        vec = np.random.randn(128)
        metrics = benchmark_compression(comp, vec)
        assert isinstance(metrics, CompressionMetrics)
        assert metrics.compression_ratio > 0
        assert metrics.snr_db > 0

    def test_benchmark_with_recovered(self):
        comp = HelivecX(dim=64, seed=1)
        vec = np.random.randn(64)
        metrics, rec = benchmark_compression(comp, vec, return_recovered=True)
        assert isinstance(metrics, CompressionMetrics)
        assert rec.shape == (64,)


# --------------------------------------------------------------------------- #
# Edge cases
# --------------------------------------------------------------------------- #

class TestEdgeCases:
    def test_small_dimension(self):
        comp = HelivecX(dim=3, seed=0)
        vec = np.array([1.0, 2.0, 3.0])
        indexed_data, meta = comp.compress(vec)
        rec = comp.decompress(indexed_data, meta)
        assert rec.shape == (3,)

    def test_large_dimension(self):
        dim = 2048
        comp = HelivecX(dim=dim, seed=42)
        vec = np.random.randn(dim)
        indexed_data, meta = comp.compress(vec)
        rec = comp.decompress(indexed_data, meta)
        assert rec.shape == (dim,)

    def test_all_zeros(self):
        comp = HelivecX(dim=16, seed=0)
        vec = np.zeros(16)
        indexed_data, meta = comp.compress(vec)
        rec = comp.decompress(indexed_data, meta)
        assert rec.shape == (16,)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
