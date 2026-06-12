"""dna_compress/tests/test_core.py

Tests for the DNA-inspired compression system.
"""
import numpy as np
import pytest
import os
from dna_compress.core import DNACompressor, normalize, random_rotation_matrix, benchmark_compression


class TestNormalize:
    def test_zero_vector(self):
        assert np.array_equal(normalize(np.zeros(5)), np.zeros(5))

    def test_unit_length(self):
        v = np.array([3.0, 4.0])
        n = normalize(v)
        assert np.isclose(np.linalg.norm(n), 1.0)


class TestRotation:
    def test_random_rotation_orthogonality(self):
        Q = random_rotation_matrix(10, seed=0)
        assert np.allclose(Q @ Q.T, np.eye(10))
        assert np.isclose(np.linalg.det(Q), 1.0)

    def test_random_rotation_preserves_norm(self):
        Q = random_rotation_matrix(10, seed=1)
        v = np.random.randn(10)
        u = Q @ v
        assert np.isclose(np.linalg.norm(v), np.linalg.norm(u))


class TestDNACompressor:
    def test_roundtrip_identity_approx(self):
        dim = 64
        comp = DNACompressor(dim=dim, seed=42)
        vec = np.random.randn(dim)
        helix, meta = comp.compress(vec)
        rec = comp.decompress(helix, meta)
        # Normalize both for comparison (direction on sphere)
        v_norm = normalize(vec)
        # Align sign
        if np.dot(rec, v_norm) < 0:
            rec = -rec
        corr = np.dot(rec, v_norm)
        assert corr > 0.85, f"Correlation too low: {corr}"

    def test_serialize_deserialize(self, tmp_path):
        dim = 32
        comp = DNACompressor(dim=dim, seed=7)
        vec = np.random.randn(dim)
        helix, meta = comp.compress(vec)
        path = tmp_path / "test.dna"
        comp.serialize(helix, meta, str(path))
        helix2, meta2 = DNACompressor.deserialize(str(path))
        assert np.array_equal(helix, helix2)
        assert meta == meta2

    def test_benchmark(self):
        dim = 128
        comp = DNACompressor(dim=dim, seed=0)
        vec = np.random.randn(dim)
        res = benchmark_compression(comp, vec)
        assert "mse" in res
        assert "snr_db" in res
        assert "compression_ratio" in res
        assert res["compression_ratio"] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
