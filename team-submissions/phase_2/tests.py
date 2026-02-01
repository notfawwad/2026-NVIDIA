"""
LABS Optimizer Test Suite
==========================
Comprehensive tests for physical correctness and code coverage

Team: SHU Quantum Solvers
"""

import pytest
import numpy as np
import cupy as cp
from labs_optimizer import (
    LABSEnergyCalculator,
    MTSClassicalSearch,
    LABSResult,
    WarmStartQAOA
)


class TestPhysicalCorrectness:
    """Tests for physical symmetries and ground truth validation"""

    def test_z2_inversion_symmetry(self):
        """
        CRITICAL: Z2 Inversion Symmetry
        Energy must be invariant under global spin flip: E(S) = E(-S)
        """
        N = 15
        np.random.seed(42)
        sequence = np.random.choice([-1, 1], size=N)

        energy_original = LABSEnergyCalculator.compute_energy_cpu(sequence)
        energy_inverted = LABSEnergyCalculator.compute_energy_cpu(-sequence)

        assert abs(energy_original - energy_inverted) < 1e-10, \
            f"Z2 symmetry violated: E(S)={energy_original} != E(-S)={energy_inverted}"

    def test_spatial_reversal_symmetry(self):
        """
        CRITICAL: Spatial Reversal Symmetry
        Energy must be invariant under sequence reversal: E(S) = E(S[::-1])
        """
        N = 15
        np.random.seed(42)
        sequence = np.random.choice([-1, 1], size=N)

        energy_original = LABSEnergyCalculator.compute_energy_cpu(sequence)
        energy_reversed = LABSEnergyCalculator.compute_energy_cpu(sequence[::-1])

        assert abs(energy_original - energy_reversed) < 1e-10, \
            f"Reversal symmetry violated: E(S)={energy_original} != E(S[::-1])={energy_reversed}"

    def test_barker_13_ground_truth(self):
        """
        CRITICAL: Ground Truth Validation
        Known Barker-13 sequence must yield exactly E = 6.0
        """
        # Barker-13: +1 +1 +1 +1 +1 -1 -1 +1 +1 -1 +1 -1 +1
        barker_13 = np.array([1, 1, 1, 1, 1, -1, -1, 1, 1, -1, 1, -1, 1])

        energy = LABSEnergyCalculator.compute_energy_cpu(barker_13)

        assert abs(energy - 6.0) < 1e-10, \
            f"Barker-13 ground truth violated: E={energy}, expected E=6.0"

    def test_energy_bounds(self):
        """
        Energy must be non-negative and bounded: 0 <= E <= N^2
        """
        for N in [10, 15, 20]:
            np.random.seed(N)
            sequence = np.random.choice([-1, 1], size=N)
            energy = LABSEnergyCalculator.compute_energy_cpu(sequence)

            assert energy >= 0, f"Energy {energy} is negative for N={N}"
            assert energy <= N * N, f"Energy {energy} exceeds N^2={N*N} for N={N}"

    def test_merit_factor_bounds(self):
        """
        Merit factor F = N^2 / (2E) must satisfy F >= 1 for valid sequences
        """
        for N in [10, 15, 20]:
            np.random.seed(N)
            sequence = np.random.choice([-1, 1], size=N)
            energy = LABSEnergyCalculator.compute_energy_cpu(sequence)
            merit_factor = LABSEnergyCalculator.compute_merit_factor(sequence, energy)

            assert merit_factor >= 1.0, \
                f"Merit factor {merit_factor} < 1.0 for N={N}, E={energy}"


class TestGPUCPUConsistency:
    """Verify GPU implementations match CPU reference"""

    def test_single_energy_gpu_cpu_match(self):
        """Single energy computation: GPU must match CPU exactly"""
        if not cp.cuda.is_available():
            pytest.skip("GPU not available")

        N = 20
        np.random.seed(42)
        sequence = np.random.choice([-1, 1], size=N)

        energy_cpu = LABSEnergyCalculator.compute_energy_cpu(sequence)
        energy_gpu = LABSEnergyCalculator.compute_energy_gpu(sequence)

        assert abs(energy_cpu - energy_gpu) < 1e-10, \
            f"GPU/CPU mismatch: CPU={energy_cpu}, GPU={energy_gpu}"

    def test_batch_energy_gpu_cpu_match(self):
        """Batch energy computation: GPU must match CPU for all sequences"""
        if not cp.cuda.is_available():
            pytest.skip("GPU not available")

        N = 15
        batch_size = 100
        np.random.seed(42)
        sequences_batch = np.random.choice([-1, 1], size=(batch_size, N))

        # CPU reference
        energies_cpu = np.array([
            LABSEnergyCalculator.compute_energy_cpu(seq) for seq in sequences_batch
        ])

        # GPU batch
        energies_gpu = LABSEnergyCalculator.compute_energy_batch_gpu(sequences_batch)

        max_error = np.max(np.abs(energies_cpu - energies_gpu))
        assert max_error < 1e-9, \
            f"Batch GPU/CPU mismatch: max error = {max_error}"

    def test_gpu_kernel_fusion_correctness(self):
        """Verify fused CUDA kernels maintain correctness"""
        if not cp.cuda.is_available():
            pytest.skip("GPU not available")

        # Test with multiple problem sizes
        for N in [10, 15, 20, 25]:
            np.random.seed(N)
            sequence = np.random.choice([-1, 1], size=N)

            energy_cpu = LABSEnergyCalculator.compute_energy_cpu(sequence)
            energy_gpu = LABSEnergyCalculator.compute_energy_gpu(sequence)

            assert abs(energy_cpu - energy_gpu) < 1e-10, \
                f"Fused kernel error at N={N}: CPU={energy_cpu}, GPU={energy_gpu}"


class TestMTSOptimizer:
    """Tests for MTS classical search"""

    def test_mts_improves_energy(self):
        """MTS must reduce energy from random initialization"""
        N = 15
        np.random.seed(42)
        initial = np.random.choice([-1, 1], size=N)
        initial_energy = LABSEnergyCalculator.compute_energy_cpu(initial)

        mts = MTSClassicalSearch(N, use_gpu=False, batch_size=100)
        result = mts.optimize(initial.copy(), max_iterations=100)

        assert result.energy <= initial_energy, \
            f"MTS failed to improve: initial={initial_energy}, final={result.energy}"

    def test_mts_respects_symmetries(self):
        """MTS solutions must satisfy physical symmetries"""
        N = 15
        np.random.seed(42)
        initial = np.random.choice([-1, 1], size=N)

        mts = MTSClassicalSearch(N, use_gpu=False, batch_size=100)
        result = mts.optimize(initial, max_iterations=100)

        # Test Z2 symmetry on result
        energy_result = result.energy
        energy_inverted = LABSEnergyCalculator.compute_energy_cpu(-result.sequence)
        assert abs(energy_result - energy_inverted) < 1e-10, \
            "MTS result violates Z2 symmetry"

        # Test reversal symmetry
        energy_reversed = LABSEnergyCalculator.compute_energy_cpu(result.sequence[::-1])
        assert abs(energy_result - energy_reversed) < 1e-10, \
            "MTS result violates reversal symmetry"

    def test_neighbor_generation_validity(self):
        """Generated neighbors must be valid ±1 sequences"""
        N = 20
        mts = MTSClassicalSearch(N, use_gpu=False, batch_size=1000)
        current = np.random.choice([-1, 1], size=N)

        neighbors = mts._generate_neighbors_batch(current, batch_size=1000)

        # All values must be ±1
        assert np.all(np.isin(neighbors, [-1, 1])), \
            "Neighbors contain invalid values (not ±1)"

        # Each neighbor should differ by exactly one flip
        hamming_distances = np.sum(neighbors != current, axis=1)
        assert np.all(hamming_distances == 1), \
            "Neighbors are not exactly 1-bit flips"


class TestGPUAcceleration:
    """Tests specifically for GPU acceleration performance"""

    @pytest.mark.skipif(not cp.cuda.is_available(), reason="GPU not available")
    def test_gpu_faster_than_cpu_for_batches(self):
        """GPU batch evaluation must be faster than CPU for large batches"""
        import time

        N = 20
        batch_size = 1000
        np.random.seed(42)
        sequences_batch = np.random.choice([-1, 1], size=(batch_size, N))

        # CPU timing
        start = time.time()
        energies_cpu = np.array([
            LABSEnergyCalculator.compute_energy_cpu(seq) for seq in sequences_batch
        ])
        cpu_time = time.time() - start

        # GPU timing
        start = time.time()
        energies_gpu = LABSEnergyCalculator.compute_energy_batch_gpu(sequences_batch)
        gpu_time = time.time() - start

        speedup = cpu_time / gpu_time if gpu_time > 0 else 0

        print(f"\nBatch evaluation speedup: {speedup:.2f}x (CPU: {cpu_time:.3f}s, GPU: {gpu_time:.3f}s)")

        # GPU should be faster for large batches (conservative check: at least 1.5x)
        assert speedup > 1.5, \
            f"GPU not faster than CPU for batches: speedup={speedup:.2f}x"

    @pytest.mark.skipif(not cp.cuda.is_available(), reason="GPU not available")
    def test_gpu_memory_efficiency(self):
        """GPU batch operations must not exceed available memory"""
        N = 30
        batch_size = 10000

        sequences_batch = np.random.choice([-1, 1], size=(batch_size, N))

        # This should not raise a memory error
        try:
            energies = LABSEnergyCalculator.compute_energy_batch_gpu(sequences_batch)
            assert len(energies) == batch_size
        except cp.cuda.memory.OutOfMemoryError:
            pytest.fail("GPU ran out of memory for reasonable batch size")


class TestAIHallucinationGuardrails:
    """Property-based tests to catch AI-generated code bugs"""

    @pytest.mark.parametrize("N", [5, 10, 15, 20, 25, 30])
    def test_energy_within_theoretical_bounds(self, N):
        """
        Hypothesis-style property test: Energy must always be in [0, N^2]
        This catches AI hallucinations that generate impossible values
        """
        for _ in range(10):  # Multiple random trials
            sequence = np.random.choice([-1, 1], size=N)
            energy = LABSEnergyCalculator.compute_energy_cpu(sequence)

            assert 0 <= energy <= N * N, \
                f"Energy {energy} violates bounds [0, {N*N}] for N={N}"

    @pytest.mark.parametrize("N", [5, 10, 15])
    def test_deterministic_reproducibility(self, N):
        """Same input must always produce same output (no random behavior in energy calc)"""
        sequence = np.random.choice([-1, 1], size=N)

        energy1 = LABSEnergyCalculator.compute_energy_cpu(sequence)
        energy2 = LABSEnergyCalculator.compute_energy_cpu(sequence)
        energy3 = LABSEnergyCalculator.compute_energy_cpu(sequence)

        assert energy1 == energy2 == energy3, \
            "Energy computation is non-deterministic"

    def test_zero_energy_only_for_perfect_sequences(self):
        """
        Energy = 0 should only occur for perfect sequences (all autocorrelations = 0)
        This is theoretically impossible for N > 4, so any E=0 for N>4 is a bug
        """
        for N in [5, 10, 15, 20]:
            for _ in range(20):
                sequence = np.random.choice([-1, 1], size=N)
                energy = LABSEnergyCalculator.compute_energy_cpu(sequence)

                if N > 4:
                    assert energy > 0, \
                        f"Impossible zero energy for N={N} (no perfect sequences exist)"


def run_coverage_report():
    """Generate test coverage report"""
    import subprocess
    try:
        result = subprocess.run(
            ['pytest', '--cov=labs_optimizer', '--cov-report=html', '--cov-report=term', 'tests.py'],
            capture_output=True,
            text=True
        )
        print(result.stdout)
        print(result.stderr)
    except Exception as e:
        print(f"Coverage report generation failed: {e}")
        print("Run manually: pytest --cov=labs_optimizer --cov-report=html tests.py")


if __name__ == "__main__":
    print("Running LABS Optimizer Test Suite")
    print("="*60)

    # Run tests with verbose output
    pytest.main(['-v', 'tests.py'])

    print("\n" + "="*60)
    print("Test suite complete. Run 'pytest --cov=labs_optimizer tests.py' for coverage.")
