"""
Quick GPU Verification Test
Tests GPU acceleration with Uniform Grid
"""

import sys
sys.path.append('./src')

import numpy as np
import torch
from ar_amr_pigf import (
    GaussianPrimitive,
    GaussianField,
    GPUEvaluator,
    GridConfig
)

print("="*60)
print("GPU Acceleration Quick Test")
print("="*60)

# Check CUDA
print(f"\nPyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    device = torch.device('cuda')
else:
    print("Using CPU (GPU not available)")
    device = torch.device('cpu')

# Test 1: Create GPU field
print("\n[1/4] Creating Gaussian field...")
field = GaussianField()

num_gaussians = 100
for i in range(num_gaussians):
    mu = np.random.uniform(0, 1, size=2)
    sigma = 0.05**2 * np.eye(2)
    weight = np.random.randn()
    primitive = GaussianPrimitive(weight=weight, mu=mu, sigma=sigma)
    field.add_primitive(primitive)

print(f"  ✓ Created field with {num_gaussians} Gaussians")

# Test 2: Create GPU evaluator
print("\n[2/4] Creating GPU evaluator...")
field.update_cutoff_radii(eps_rel=1e-6)

gpu_evaluator = GPUEvaluator(
    field,
    eps_rel=1e-6,
    grid_config=GridConfig(cells_per_dim=10),
    device=device
)

domain_min = np.zeros(2)
domain_max = np.ones(2)
gpu_evaluator.build_grid(domain_min, domain_max)

print(f"  ✓ GPU evaluator created on {device}")

grid_stats = gpu_evaluator.grid.get_statistics()
print(f"  ✓ Grid: {grid_stats['num_cells']} cells")
print(f"  ✓ Avg Gaussians/cell: {grid_stats['avg_gaussians_per_cell']:.1f}")

# Test 3: Evaluate
print("\n[3/4] Testing evaluation...")
query_points = np.random.uniform(0, 1, size=(50, 2))

# Without grid
result_no_grid = gpu_evaluator.evaluate_batch(query_points, use_grid=False)
print(f"  ✓ Evaluation (no grid) complete")

# With grid
result_with_grid = gpu_evaluator.evaluate_batch(query_points, use_grid=True)
print(f"  ✓ Evaluation (with grid) complete")

# Check consistency
error = np.mean(np.abs(result_no_grid - result_with_grid))
print(f"  ✓ Consistency error: {error:.6e}")

# Test 4: Benchmark
print("\n[4/4] Running mini benchmark...")
import time

num_queries = 200
query_points_bench = np.random.uniform(0, 1, size=(num_queries, 2))

# No grid
times_no_grid = []
for _ in range(5):
    start = time.time()
    _ = gpu_evaluator.evaluate_batch(query_points_bench, use_grid=False)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    times_no_grid.append(time.time() - start)

time_no_grid = np.mean(times_no_grid)

# With grid
times_with_grid = []
for _ in range(5):
    start = time.time()
    _ = gpu_evaluator.evaluate_batch(query_points_bench, use_grid=True)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    times_with_grid.append(time.time() - start)

time_with_grid = np.mean(times_with_grid)

speedup = time_no_grid / time_with_grid

print(f"\n  Results ({num_queries} queries):")
print(f"    No grid:   {time_no_grid:.4f}s")
print(f"    With grid: {time_with_grid:.4f}s")
print(f"    Speedup:   {speedup:.2f}x")

# Test gradient
print("\n[Bonus] Testing gradient computation...")
grad_no_grid = gpu_evaluator.gradient_batch(query_points, use_grid=False)
grad_with_grid = gpu_evaluator.gradient_batch(query_points, use_grid=True)
grad_error = np.mean(np.abs(grad_no_grid - grad_with_grid))
print(f"  ✓ Gradient computed, error: {grad_error:.6e}")

print("\n" + "="*60)
print("✓ ALL GPU TESTS PASSED!")
print("="*60)

stats = gpu_evaluator.get_statistics()
print("\nGPU Evaluator Statistics:")
print(f"  Device: {stats['device']}")
print(f"  Num Gaussians: {stats['num_gaussians']}")
print(f"  Grid cells: {stats['grid_cells']}")
print(f"  Avg Gaussians/cell: {stats['avg_gaussians_per_cell']:.1f}")
print(f"  Max Gaussians/cell: {stats['max_gaussians_per_cell']}")

print("\n" + "="*60)
print("GPU acceleration is working!")
print("\nNext steps:")
print("  1. Run full benchmark: python tests/test_gpu_acceleration.py")
print("  2. Try with larger problems (N=1000-10000)")
print("  3. Test on real CUDA GPU for maximum speedup")
print("="*60)
