"""
GPU Acceleration Benchmark
Tests Uniform Grid + GPU acceleration vs KD-tree + CPU
"""

import sys
sys.path.append('../src')

import numpy as np
import torch
import time
import matplotlib.pyplot as plt

from ar_amr_pigf.gaussian import GaussianField, GaussianPrimitive
from ar_amr_pigf.evaluator import FastEvaluator
from ar_amr_pigf.gpu_evaluator import GPUEvaluator, GridConfig


def create_test_field(num_gaussians: int, dim: int = 2) -> GaussianField:
    """Create random Gaussian field for testing"""
    field = GaussianField()

    for i in range(num_gaussians):
        mu = np.random.uniform(0, 1, size=dim)
        scale = 0.03
        sigma = scale**2 * np.eye(dim)
        weight = np.random.randn()

        primitive = GaussianPrimitive(weight=weight, mu=mu, sigma=sigma)
        field.add_primitive(primitive)

    return field


def benchmark_comparison(num_gaussians_list: list,
                        num_queries: int = 1000,
                        dim: int = 2):
    """
    Compare CPU (KD-tree) vs GPU (Uniform Grid) performance

    Args:
        num_gaussians_list: List of Gaussian counts to test
        num_queries: Number of query points
        dim: Spatial dimension
    """
    print("="*70)
    print("GPU ACCELERATION BENCHMARK")
    print("="*70)
    print(f"Query points: {num_queries}")
    print(f"Dimension: {dim}D")
    print(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    print("="*70)

    results = {
        'num_gaussians': [],
        'cpu_kdtree_time': [],
        'gpu_nogrid_time': [],
        'gpu_grid_time': [],
        'cpu_speedup': [],
        'grid_speedup': [],
    }

    for num_gaussians in num_gaussians_list:
        print(f"\n{'='*70}")
        print(f"Testing N = {num_gaussians} Gaussians")
        print(f"{'='*70}")

        # Create field
        field = create_test_field(num_gaussians, dim=dim)

        # Update cutoff radii
        field.update_cutoff_radii(eps_rel=1e-6)

        # Query points
        query_points = np.random.uniform(0, 1, size=(num_queries, dim))

        # ============================================================
        # CPU with KD-tree
        # ============================================================
        print("\n[1/3] CPU with KD-tree...")
        cpu_evaluator = FastEvaluator(field, eps_rel=1e-6, use_scipy=True)
        cpu_evaluator.build_tree()

        cpu_times = []
        for _ in range(5):
            start = time.time()
            cpu_result = cpu_evaluator.batch_evaluate(
                query_points, use_acceleration=True
            )
            cpu_times.append(time.time() - start)

        cpu_time = np.mean(cpu_times)
        print(f"  Time: {cpu_time:.4f}s")

        # ============================================================
        # GPU without grid
        # ============================================================
        print("\n[2/3] GPU without grid...")
        gpu_evaluator = GPUEvaluator(
            field,
            eps_rel=1e-6,
            grid_config=GridConfig(cells_per_dim=20)
        )

        gpu_times_nogrid = []
        for _ in range(5):
            start = time.time()
            gpu_result_nogrid = gpu_evaluator.evaluate_batch(
                query_points, use_grid=False
            )
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            gpu_times_nogrid.append(time.time() - start)

        gpu_time_nogrid = np.mean(gpu_times_nogrid)
        print(f"  Time: {gpu_time_nogrid:.4f}s")

        # ============================================================
        # GPU with grid
        # ============================================================
        print("\n[3/3] GPU with uniform grid...")
        domain_min = np.zeros(dim)
        domain_max = np.ones(dim)
        gpu_evaluator.build_grid(domain_min, domain_max)

        grid_stats = gpu_evaluator.grid.get_statistics()
        print(f"  Grid: {grid_stats['num_cells']} cells")
        print(f"  Avg Gaussians/cell: {grid_stats['avg_gaussians_per_cell']:.1f}")

        gpu_times_grid = []
        for _ in range(5):
            start = time.time()
            gpu_result_grid = gpu_evaluator.evaluate_batch(
                query_points, use_grid=True
            )
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            gpu_times_grid.append(time.time() - start)

        gpu_time_grid = np.mean(gpu_times_grid)
        print(f"  Time: {gpu_time_grid:.4f}s")

        # ============================================================
        # Results
        # ============================================================
        cpu_speedup = cpu_time / gpu_time_grid
        grid_speedup = gpu_time_nogrid / gpu_time_grid

        print(f"\n{'='*70}")
        print("RESULTS:")
        print(f"  CPU (KD-tree):     {cpu_time:.4f}s")
        print(f"  GPU (no grid):     {gpu_time_nogrid:.4f}s")
        print(f"  GPU (with grid):   {gpu_time_grid:.4f}s")
        print(f"  GPU vs CPU:        {cpu_speedup:.2f}x {'faster' if cpu_speedup > 1 else 'slower'}")
        print(f"  Grid acceleration: {grid_speedup:.2f}x")
        print(f"{'='*70}")

        # Verify correctness (rough check)
        error = np.mean(np.abs(cpu_result - gpu_result_grid))
        max_error = np.max(np.abs(cpu_result - gpu_result_grid))
        print(f"\nAccuracy check:")
        print(f"  Mean error: {error:.6e}")
        print(f"  Max error:  {max_error:.6e}")

        # Store results
        results['num_gaussians'].append(num_gaussians)
        results['cpu_kdtree_time'].append(cpu_time)
        results['gpu_nogrid_time'].append(gpu_time_nogrid)
        results['gpu_grid_time'].append(gpu_time_grid)
        results['cpu_speedup'].append(cpu_speedup)
        results['grid_speedup'].append(grid_speedup)

    return results


def plot_results(results: dict):
    """Plot benchmark results"""
    fig = plt.figure(figsize=(15, 5))

    # 1. Time comparison
    ax1 = plt.subplot(1, 3, 1)
    ax1.loglog(results['num_gaussians'], results['cpu_kdtree_time'],
               'bo-', label='CPU (KD-tree)', linewidth=2, markersize=8)
    ax1.loglog(results['num_gaussians'], results['gpu_nogrid_time'],
               'go-', label='GPU (no grid)', linewidth=2, markersize=8)
    ax1.loglog(results['num_gaussians'], results['gpu_grid_time'],
               'ro-', label='GPU (with grid)', linewidth=2, markersize=8)
    ax1.set_xlabel('Number of Gaussians')
    ax1.set_ylabel('Time (seconds)')
    ax1.set_title('Evaluation Time Comparison')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Speedup vs CPU
    ax2 = plt.subplot(1, 3, 2)
    ax2.semilogx(results['num_gaussians'], results['cpu_speedup'],
                 'mo-', linewidth=2, markersize=8)
    ax2.axhline(y=1.0, color='k', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Number of Gaussians')
    ax2.set_ylabel('Speedup Factor')
    ax2.set_title('GPU (with grid) vs CPU (KD-tree)')
    ax2.grid(True, alpha=0.3)

    # 3. Grid acceleration
    ax3 = plt.subplot(1, 3, 3)
    ax3.semilogx(results['num_gaussians'], results['grid_speedup'],
                 'co-', linewidth=2, markersize=8)
    ax3.axhline(y=1.0, color='k', linestyle='--', alpha=0.5)
    ax3.set_xlabel('Number of Gaussians')
    ax3.set_ylabel('Speedup Factor')
    ax3.set_title('Grid Acceleration (GPU)')
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('../results/gpu_benchmark_results.png', dpi=150, bbox_inches='tight')
    print("\n✓ Plot saved to ../results/gpu_benchmark_results.png")
    plt.show()


def test_different_dimensions():
    """Test GPU acceleration in different dimensions"""
    print("\n" + "="*70)
    print("DIMENSION SCALING TEST")
    print("="*70)

    num_gaussians = 1000
    num_queries = 500

    for dim in [1, 2, 3]:
        print(f"\n{'='*70}")
        print(f"Testing {dim}D")
        print(f"{'='*70}")

        field = create_test_field(num_gaussians, dim=dim)
        field.update_cutoff_radii(eps_rel=1e-6)

        query_points = np.random.uniform(0, 1, size=(num_queries, dim))

        # GPU evaluation
        gpu_evaluator = GPUEvaluator(field, eps_rel=1e-6)
        domain_min = np.zeros(dim)
        domain_max = np.ones(dim)
        gpu_evaluator.build_grid(domain_min, domain_max)

        # Benchmark
        start = time.time()
        result = gpu_evaluator.evaluate_batch(query_points, use_grid=True)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        gpu_time = time.time() - start

        print(f"GPU time ({dim}D): {gpu_time:.4f}s")

        grid_stats = gpu_evaluator.grid.get_statistics()
        print(f"Grid cells: {grid_stats['num_cells']}")
        print(f"Avg Gaussians/cell: {grid_stats['avg_gaussians_per_cell']:.1f}")


def run_all_benchmarks():
    """Run comprehensive GPU benchmarks"""
    import os
    os.makedirs('../results', exist_ok=True)

    print("\n" + "="*70)
    print("COMPREHENSIVE GPU ACCELERATION BENCHMARKS")
    print("="*70)
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
        print(f"CUDA version: {torch.version.cuda}")

    # Main benchmark
    print("\n" + "="*70)
    print("MAIN BENCHMARK: Scaling with number of Gaussians")
    print("="*70)

    num_gaussians_list = [100, 500, 1000, 2000, 5000, 10000]
    results = benchmark_comparison(
        num_gaussians_list,
        num_queries=1000,
        dim=2
    )

    # Plot results
    plot_results(results)

    # Dimension test
    test_different_dimensions()

    print("\n" + "="*70)
    print("✓ ALL BENCHMARKS COMPLETE")
    print("="*70)


if __name__ == '__main__':
    run_all_benchmarks()
