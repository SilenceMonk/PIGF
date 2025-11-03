"""
Fast Evaluator with KD-tree Acceleration
Implements Algorithm 4.1 and 4.2 from the framework
"""

import numpy as np
from typing import List, Tuple, Optional
import time

from .gaussian import GaussianField, GaussianPrimitive
from .kdtree import KDTree, KDTreeAdapter


class FastEvaluator:
    """
    Accelerated evaluator for Gaussian fields using KD-tree spatial indexing

    Implements:
    - Fast single-point evaluation (Algorithm 4.1)
    - Fast gradient computation (Algorithm 4.2)
    - Batch evaluation with vectorization
    """

    def __init__(self, field: GaussianField, eps_rel: float = 1e-6,
                 use_scipy: bool = True, rebuild_threshold: float = 0.3):
        """
        Initialize fast evaluator

        Args:
            field: GaussianField to evaluate
            eps_rel: Relative tolerance for cutoff
            use_scipy: If True, use scipy's cKDTree (faster)
            rebuild_threshold: Fraction of changes to trigger rebuild
        """
        self.field = field
        self.eps_rel = eps_rel
        self.rebuild_threshold = rebuild_threshold

        # Create KD-tree
        if use_scipy:
            try:
                self.kdtree = KDTreeAdapter(leaf_size=10)
            except ImportError:
                print("Warning: scipy not available, using custom KDTree")
                self.kdtree = KDTree(leaf_size=10)
        else:
            self.kdtree = KDTree(leaf_size=10)

        self._tree_valid = False

        # Statistics
        self.stats = {
            'num_evaluations': 0,
            'num_tree_queries': 0,
            'total_active_gaussians': 0,
            'build_time': 0.0,
            'query_time': 0.0,
            'eval_time': 0.0,
        }

    def build_tree(self):
        """Build or rebuild KD-tree from current field state"""
        if self.field.num_gaussians == 0:
            self._tree_valid = False
            return

        start_time = time.time()

        # Update cutoff radii for all primitives
        self.field.update_cutoff_radii(self.eps_rel)

        # Get centers and radii
        _, mus, _ = self.field.get_parameters()
        radii = np.array([p.cutoff_radius for p in self.field.primitives])

        # Build tree
        self.kdtree.build(mus, radii)

        self._tree_valid = True
        self.stats['build_time'] += time.time() - start_time

    def invalidate_tree(self):
        """Mark tree as invalid (needs rebuild)"""
        self._tree_valid = False

    def evaluate(self, x: np.ndarray, use_acceleration: bool = True) -> float:
        """
        Evaluate field at point x using KD-tree acceleration

        Implements Algorithm 4.1 (FastEvaluate)

        Args:
            x: Query point, shape (d,)
            use_acceleration: If True, use KD-tree (otherwise naive)

        Returns:
            Field value u(x)
        """
        if not use_acceleration:
            return self.field.evaluate(x, use_acceleration=False,
                                       eps_rel=self.eps_rel)

        # Ensure tree is built
        if not self._tree_valid:
            self.build_tree()

        start_time = time.time()

        # Step 1: Range query to find candidate active Gaussians
        active_indices = self.kdtree.range_query(x, query_radius=0.0)

        self.stats['num_tree_queries'] += 1
        self.stats['query_time'] += time.time() - start_time

        # Step 2 & 3: Evaluate active Gaussians with precise Mahalanobis check
        start_time = time.time()

        result = 0.0
        threshold = np.sqrt(-2 * np.log(self.eps_rel))

        for idx in active_indices:
            primitive = self.field.primitives[idx]

            # Precise Mahalanobis distance check
            d_mahal = primitive.mahalanobis_distance(x)

            if d_mahal <= threshold:
                # Compute Gaussian value
                diff = x - primitive.mu
                exponent = -0.5 * (diff @ primitive.sigma_inv @ diff)
                gaussian_value = primitive.normalizer * np.exp(exponent)
                result += primitive.weight * gaussian_value

        self.stats['num_evaluations'] += 1
        self.stats['total_active_gaussians'] += len(active_indices)
        self.stats['eval_time'] += time.time() - start_time

        return result

    def gradient(self, x: np.ndarray, use_acceleration: bool = True) -> np.ndarray:
        """
        Compute gradient at point x using KD-tree acceleration

        Implements Algorithm 4.2 (FastGradient)

        Args:
            x: Query point, shape (d,)
            use_acceleration: If True, use KD-tree

        Returns:
            Gradient vector, shape (d,)
        """
        if not use_acceleration:
            return self.field.gradient(x, use_acceleration=False,
                                       eps_rel=self.eps_rel)

        # Ensure tree is built
        if not self._tree_valid:
            self.build_tree()

        # Range query
        active_indices = self.kdtree.range_query(x, query_radius=0.0)

        # Compute gradient
        grad = np.zeros(self.field.dim)
        threshold = np.sqrt(-2 * np.log(self.eps_rel))

        for idx in active_indices:
            primitive = self.field.primitives[idx]

            # Check Mahalanobis distance
            diff = x - primitive.mu
            d_mahal = primitive.mahalanobis_distance(x)

            if d_mahal <= threshold:
                # Compute Gaussian value
                exponent = -0.5 * (diff @ primitive.sigma_inv @ diff)
                gaussian_value = primitive.normalizer * np.exp(exponent)

                # Gradient: ∇G = -Σ⁻¹(x-μ) · G
                grad += -primitive.weight * (primitive.sigma_inv @ diff) * gaussian_value

        return grad

    def laplacian(self, x: np.ndarray, use_acceleration: bool = True) -> float:
        """
        Compute Laplacian at point x using KD-tree acceleration

        Args:
            x: Query point, shape (d,)
            use_acceleration: If True, use KD-tree

        Returns:
            Laplacian value
        """
        if not use_acceleration:
            return self.field.laplacian(x, use_acceleration=False,
                                        eps_rel=self.eps_rel)

        if not self._tree_valid:
            self.build_tree()

        active_indices = self.kdtree.range_query(x, query_radius=0.0)

        result = 0.0
        threshold = np.sqrt(-2 * np.log(self.eps_rel))

        for idx in active_indices:
            primitive = self.field.primitives[idx]

            diff = x - primitive.mu
            d_mahal = primitive.mahalanobis_distance(x)

            if d_mahal <= threshold:
                # Gaussian value
                exponent = -0.5 * (diff @ primitive.sigma_inv @ diff)
                gaussian_value = primitive.normalizer * np.exp(exponent)

                # Laplacian: ΔG = [tr(Σ⁻¹) - diffᵀ Σ⁻¹ Σ⁻¹ diff] · G
                trace_term = np.trace(primitive.sigma_inv)
                quadratic_term = diff @ primitive.sigma_inv @ primitive.sigma_inv @ diff

                result += primitive.weight * (trace_term - quadratic_term) * gaussian_value

        return result

    def batch_evaluate(self, X: np.ndarray, use_acceleration: bool = True,
                       parallel: bool = False) -> np.ndarray:
        """
        Evaluate field at multiple points

        Args:
            X: Query points, shape (M, d)
            use_acceleration: If True, use KD-tree
            parallel: If True, use parallel evaluation (not implemented yet)

        Returns:
            Field values, shape (M,)
        """
        M = len(X)
        results = np.zeros(M)

        if use_acceleration and not self._tree_valid:
            self.build_tree()

        for i in range(M):
            results[i] = self.evaluate(X[i], use_acceleration=use_acceleration)

        return results

    def batch_gradient(self, X: np.ndarray, use_acceleration: bool = True) -> np.ndarray:
        """
        Compute gradients at multiple points

        Args:
            X: Query points, shape (M, d)
            use_acceleration: If True, use KD-tree

        Returns:
            Gradients, shape (M, d)
        """
        M = len(X)
        d = self.field.dim
        results = np.zeros((M, d))

        if use_acceleration and not self._tree_valid:
            self.build_tree()

        for i in range(M):
            results[i] = self.gradient(X[i], use_acceleration=use_acceleration)

        return results

    def batch_laplacian(self, X: np.ndarray, use_acceleration: bool = True) -> np.ndarray:
        """
        Compute Laplacians at multiple points

        Args:
            X: Query points, shape (M, d)
            use_acceleration: If True, use KD-tree

        Returns:
            Laplacians, shape (M,)
        """
        M = len(X)
        results = np.zeros(M)

        if use_acceleration and not self._tree_valid:
            self.build_tree()

        for i in range(M):
            results[i] = self.laplacian(X[i], use_acceleration=use_acceleration)

        return results

    def get_average_active_gaussians(self) -> float:
        """Get average number of active Gaussians per query"""
        if self.stats['num_tree_queries'] == 0:
            return 0.0
        return self.stats['total_active_gaussians'] / self.stats['num_tree_queries']

    def get_statistics(self) -> dict:
        """Get performance statistics"""
        avg_active = self.get_average_active_gaussians()

        return {
            **self.stats,
            'avg_active_gaussians': avg_active,
            'total_gaussians': self.field.num_gaussians,
            'speedup_estimate': (
                self.field.num_gaussians / avg_active if avg_active > 0 else 1.0
            ),
            'tree_valid': self._tree_valid,
        }

    def reset_statistics(self):
        """Reset performance counters"""
        self.stats = {
            'num_evaluations': 0,
            'num_tree_queries': 0,
            'total_active_gaussians': 0,
            'build_time': 0.0,
            'query_time': 0.0,
            'eval_time': 0.0,
        }

    def benchmark(self, X: np.ndarray, num_runs: int = 10) -> dict:
        """
        Benchmark acceleration vs naive evaluation

        Args:
            X: Test points, shape (M, d)
            num_runs: Number of runs for averaging

        Returns:
            Dictionary with timing results
        """
        print(f"Benchmarking on {len(X)} points, {num_runs} runs...")

        # Warm up and build tree
        self.build_tree()
        _ = self.evaluate(X[0], use_acceleration=True)

        # Accelerated evaluation
        accel_times = []
        for _ in range(num_runs):
            start = time.time()
            for x in X:
                _ = self.evaluate(x, use_acceleration=True)
            accel_times.append(time.time() - start)

        accel_time = np.mean(accel_times)

        # Naive evaluation
        naive_times = []
        for _ in range(num_runs):
            start = time.time()
            for x in X:
                _ = self.evaluate(x, use_acceleration=False)
            naive_times.append(time.time() - start)

        naive_time = np.mean(naive_times)

        speedup = naive_time / accel_time if accel_time > 0 else float('inf')

        results = {
            'num_points': len(X),
            'num_gaussians': self.field.num_gaussians,
            'accelerated_time': accel_time,
            'naive_time': naive_time,
            'speedup': speedup,
            'avg_active_gaussians': self.get_average_active_gaussians(),
        }

        print(f"Results: {speedup:.2f}x speedup "
              f"({naive_time:.4f}s → {accel_time:.4f}s)")
        print(f"Average active Gaussians: {results['avg_active_gaussians']:.1f} / "
              f"{self.field.num_gaussians}")

        return results
