"""tests/test_core.py

Comprehensive tests for the HelivecX compression system.
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
# HelivecX round-trip
# --------------------------------------------------------------------------- #

class TestHelivecX:
    def test_roundtrip_correlation(self):
        dim = 128
        comp = HelivecX(dim=dim, seed=42)
        vec = np.random.randn(dim)
        helix, meta = comp.compress(vec)
        rec = comp.decompress(helix, meta)

        v_norm = normalize(vec)
        if np.dot(rec, v_norm) < 0:
            rec = -rec
        corr = np.dot(rec, v_norm)
        assert corr > 0.85

    def test_serialization_roundtrip(self, tmp_path):
        dim = 64
        comp = HelivecX(dim=dim, seed=99)
        vec = np.random.randn(dim)
        helix, meta = comp.compress(vec)

        path = tmp_path / "test.helx"
        comp.save(helix, meta, str(path))
        helix2, meta2 = HelivecX.load(str(path))
        assert np.array_equal(helix, helix2)
        assert meta == meta2

    def test_different_dimensions(self):
        for dim in [8, 16, 32, 64, 128, 256]:
            comp = HelivecX(dim=dim, seed=0)
            vec = np.random.randn(dim)
            helix, meta = comp.compress(vec)
            rec = comp.decompress(helix, meta)
            assert rec.shape == (dim,)

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
        helix, meta = comp.compress(vec)
        rec = comp.decompress(helix, meta)
        assert rec.shape == (3,)

    def test_large_dimension(self):
        dim = 2048
        comp = HelivecX(dim=dim, seed=42)
        vec = np.random.randn(dim)
        helix, meta = comp.compress(vec)
        rec = comp.decompress(helix, meta)
        assert rec.shape == (dim,)

    def test_all_zeros(self):
        comp = HelivecX(dim=16, seed=0)
        vec = np.zeros(16)
        helix, meta = comp.compress(vec)
        rec = comp.decompress(helix, meta)
        assert rec.shape == (16,)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
