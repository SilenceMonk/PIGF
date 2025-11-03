#!/bin/bash

echo "=================================="
echo "AR-AMR-PIGF Test Suite"
echo "=================================="

# Create results directory
mkdir -p results

echo ""
echo "1. Running 1D Burgers equation test..."
cd tests
python test_burgers_1d.py

echo ""
echo "2. Running 2D diffusion equation test..."
python test_diffusion_2d.py

echo ""
echo "=================================="
echo "All tests complete!"
echo "Results saved in results/"
echo "=================================="
