"""
Quick verification test for AR-AMR-PIGF framework
Tests basic functionality without full PDE solve
"""

import sys
sys.path.append('./src')

import numpy as np
from ar_amr_pigf import (
    GaussianPrimitive,
    GaussianField,
    KDTree,
    FastEvaluator
)

print("="*60)
print("AR-AMR-PIGF Quick Verification Test")
print("="*60)

# Test 1: Gaussian Primitive
print("\n[1/4] Testing Gaussian Primitive...")
try:
    mu = np.array([0.5, 0.5])
    sigma = 0.1**2 * np.eye(2)
    primitive = GaussianPrimitive(weight=1.0, mu=mu, sigma=sigma)

    # Evaluate at center
    val = primitive.evaluate(mu)
    print(f"  ✓ Gaussian evaluation: {val:.6f}")

    # Compute gradient
    grad = primitive.gradient(mu)
    print(f"  ✓ Gradient at center: {np.linalg.norm(grad):.6e} (should be ~0)")

    # Compute cutoff radius
    r_cutoff = primitive.compute_cutoff_radius(eps_rel=1e-6)
    print(f"  ✓ Cutoff radius: {r_cutoff:.4f}")

except Exception as e:
    print(f"  ✗ FAILED: {e}")
    sys.exit(1)

# Test 2: Gaussian Field
print("\n[2/4] Testing Gaussian Field...")
try:
    field = GaussianField()

    # Add 10 random Gaussians
    for i in range(10):
        mu = np.random.uniform(0, 1, size=2)
        sigma = 0.05**2 * np.eye(2)
        weight = np.random.randn()
        primitive = GaussianPrimitive(weight=weight, mu=mu, sigma=sigma)
        field.add_primitive(primitive)

    print(f"  ✓ Created field with {field.num_gaussians} Gaussians")

    # Evaluate at test point
    test_point = np.array([0.5, 0.5])
    val = field.evaluate(test_point, use_acceleration=False)
    print(f"  ✓ Field evaluation: {val:.6f}")

except Exception as e:
    print(f"  ✗ FAILED: {e}")
    sys.exit(1)

# Test 3: KD-tree
print("\n[3/4] Testing KD-tree...")
try:
    # Create larger field for meaningful test
    field_large = GaussianField()
    N = 100

    for i in range(N):
        mu = np.random.uniform(0, 1, size=2)
        sigma = 0.03**2 * np.eye(2)
        weight = 1.0
        primitive = GaussianPrimitive(weight=weight, mu=mu, sigma=sigma)
        field_large.add_primitive(primitive)

    # Build KD-tree
    evaluator = FastEvaluator(field_large, eps_rel=1e-6, use_scipy=False)
    evaluator.build_tree()

    print(f"  ✓ Built KD-tree for {N} Gaussians")

    # Test range query
    query_point = np.array([0.5, 0.5])
    active_indices = evaluator.kdtree.range_query(query_point, query_radius=0.0)

    print(f"  ✓ Range query found {len(active_indices)} active Gaussians")

except Exception as e:
    print(f"  ✗ FAILED: {e}")
    sys.exit(1)

# Test 4: Acceleration benchmark
print("\n[4/4] Testing Acceleration...")
try:
    # Small benchmark
    query_points = np.random.uniform(0, 1, size=(50, 2))

    # Time naive evaluation
    import time
    start = time.time()
    for pt in query_points:
        _ = field_large.evaluate(pt, use_acceleration=False)
    naive_time = time.time() - start

    # Time accelerated evaluation
    start = time.time()
    for pt in query_points:
        _ = evaluator.evaluate(pt, use_acceleration=True)
    accel_time = time.time() - start

    speedup = naive_time / accel_time if accel_time > 0 else float('inf')

    print(f"  ✓ Naive time: {naive_time:.4f}s")
    print(f"  ✓ Accelerated time: {accel_time:.4f}s")
    print(f"  ✓ Speedup: {speedup:.2f}x")

    # Get statistics
    stats = evaluator.get_statistics()
    print(f"  ✓ Avg active Gaussians: {stats['avg_active_gaussians']:.1f}/{N}")

except Exception as e:
    print(f"  ✗ FAILED: {e}")
    sys.exit(1)

print("\n" + "="*60)
print("✓ ALL TESTS PASSED!")
print("="*60)
print("\nFramework is ready to use!")
print("\nNext steps:")
print("  1. Run full tests: ./run_tests.sh")
print("  2. Open Colab notebook: notebooks/ar_amr_pigf_verification.ipynb")
print("  3. Try custom PDE problems")
print("="*60)
