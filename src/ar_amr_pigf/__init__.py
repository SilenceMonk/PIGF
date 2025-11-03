"""
AR-AMR-PIGF with KD-tree Acceleration and GPU Support
Adaptive Refinement - Adaptive Mesh Refinement - Physics-Informed Gaussian Fields
"""

__version__ = "0.2.0"

from .gaussian import GaussianPrimitive, GaussianField
from .kdtree import KDTree
from .evaluator import FastEvaluator
from .amr import AMRRefiner
from .solver import ARSolver

# GPU acceleration modules
from .uniform_grid import UniformGrid, GridConfig
from .gpu_evaluator import GPUEvaluator, GPUGaussianField

__all__ = [
    "GaussianPrimitive",
    "GaussianField",
    "KDTree",
    "FastEvaluator",
    "AMRRefiner",
    "ARSolver",
    # GPU modules
    "UniformGrid",
    "GridConfig",
    "GPUEvaluator",
    "GPUGaussianField",
]
