"""
AR-AMR-PIGF with KD-tree Acceleration
Adaptive Refinement - Adaptive Mesh Refinement - Physics-Informed Gaussian Fields
"""

__version__ = "0.1.0"

from .gaussian import GaussianPrimitive, GaussianField
from .kdtree import KDTree
from .evaluator import FastEvaluator
from .amr import AMRRefiner
from .solver import ARSolver

__all__ = [
    "GaussianPrimitive",
    "GaussianField",
    "KDTree",
    "FastEvaluator",
    "AMRRefiner",
    "ARSolver",
]
