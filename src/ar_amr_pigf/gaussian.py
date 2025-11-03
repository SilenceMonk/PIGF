"""
Gaussian Primitives and Field Representation
Implements the core Gaussian basis functions and field representation
"""

import numpy as np
import torch
from typing import Optional, Tuple, Union
from dataclasses import dataclass


@dataclass
class GaussianPrimitive:
    """
    Single Gaussian primitive with parameters (w, μ, Σ)

    Represents: w * G(x; μ, Σ)
    where G(x; μ, Σ) = (2π)^(-d/2) |Σ|^(-1/2) exp(-0.5 (x-μ)ᵀ Σ⁻¹ (x-μ))
    """
    weight: float  # w_i
    mu: np.ndarray  # μ_i, shape (d,)
    sigma: np.ndarray  # Σ_i, shape (d, d)

    # Cached quantities for efficiency
    sigma_inv: Optional[np.ndarray] = None  # Σ⁻¹
    sigma_chol: Optional[np.ndarray] = None  # L where Σ = L Lᵀ
    det_sigma: Optional[float] = None  # |Σ|
    normalizer: Optional[float] = None  # (2π)^(-d/2) |Σ|^(-1/2)
    cutoff_radius: Optional[float] = None  # r_i for truncation

    def __post_init__(self):
        """Precompute cached quantities"""
        self.dim = len(self.mu)
        self._update_cache()

    def _update_cache(self):
        """Update all cached quantities after parameter changes"""
        # Cholesky decomposition: Σ = L Lᵀ
        self.sigma_chol = np.linalg.cholesky(self.sigma)

        # Determinant from Cholesky factors
        self.det_sigma = np.prod(np.diag(self.sigma_chol)) ** 2

        # Inverse from Cholesky
        L_inv = np.linalg.inv(self.sigma_chol)
        self.sigma_inv = L_inv.T @ L_inv

        # Normalization constant
        self.normalizer = 1.0 / np.sqrt((2 * np.pi) ** self.dim * self.det_sigma)

    def compute_cutoff_radius(self, eps_rel: float = 1e-6) -> float:
        """
        Compute effective support radius based on relative tolerance

        r_i = sqrt(-2 ln(eps_rel)) * sqrt(λ_max(Σ))

        Args:
            eps_rel: Relative tolerance for truncation

        Returns:
            Cutoff radius r_i
        """
        # Maximum eigenvalue of Σ
        eigenvalues = np.linalg.eigvalsh(self.sigma)
        lambda_max = np.max(eigenvalues)

        # Cutoff radius formula
        self.cutoff_radius = np.sqrt(-2 * np.log(eps_rel)) * np.sqrt(lambda_max)
        return self.cutoff_radius

    def mahalanobis_distance(self, x: np.ndarray) -> float:
        """
        Compute Mahalanobis distance: d_Σ(x, μ) = sqrt((x-μ)ᵀ Σ⁻¹ (x-μ))

        Args:
            x: Query point, shape (d,)

        Returns:
            Mahalanobis distance
        """
        diff = x - self.mu
        return np.sqrt(diff @ self.sigma_inv @ diff)

    def evaluate(self, x: np.ndarray, check_cutoff: bool = False,
                 eps_rel: float = 1e-6) -> float:
        """
        Evaluate Gaussian at point x: w * G(x; μ, Σ)

        Args:
            x: Query point, shape (d,)
            check_cutoff: If True, return 0 outside cutoff radius
            eps_rel: Relative tolerance for cutoff

        Returns:
            Weighted Gaussian value
        """
        # Check cutoff if requested
        if check_cutoff:
            if self.cutoff_radius is None:
                self.compute_cutoff_radius(eps_rel)

            # Euclidean distance check (conservative)
            if np.linalg.norm(x - self.mu) > self.cutoff_radius:
                return 0.0

            # Precise Mahalanobis distance check
            d_mahal = self.mahalanobis_distance(x)
            threshold = np.sqrt(-2 * np.log(eps_rel))
            if d_mahal > threshold:
                return 0.0

        # Compute Gaussian value
        diff = x - self.mu
        exponent = -0.5 * (diff @ self.sigma_inv @ diff)
        gaussian_value = self.normalizer * np.exp(exponent)

        return self.weight * gaussian_value

    def gradient(self, x: np.ndarray, check_cutoff: bool = False,
                 eps_rel: float = 1e-6) -> np.ndarray:
        """
        Compute gradient at point x: ∇_x [w * G(x; μ, Σ)]

        Formula: ∇G = -Σ⁻¹(x-μ) · G(x; μ, Σ)

        Args:
            x: Query point, shape (d,)
            check_cutoff: If True, return 0 outside cutoff radius
            eps_rel: Relative tolerance for cutoff

        Returns:
            Gradient vector, shape (d,)
        """
        # Check cutoff
        if check_cutoff:
            if self.cutoff_radius is None:
                self.compute_cutoff_radius(eps_rel)

            if np.linalg.norm(x - self.mu) > self.cutoff_radius:
                return np.zeros_like(x)

            d_mahal = self.mahalanobis_distance(x)
            threshold = np.sqrt(-2 * np.log(eps_rel))
            if d_mahal > threshold:
                return np.zeros_like(x)

        # Compute Gaussian and gradient
        diff = x - self.mu
        gaussian_value = self.evaluate(x, check_cutoff=False)

        # ∇G = -Σ⁻¹(x-μ) · G
        return -self.weight * (self.sigma_inv @ diff) * gaussian_value / self.weight

    def laplacian(self, x: np.ndarray, check_cutoff: bool = False,
                  eps_rel: float = 1e-6) -> float:
        """
        Compute Laplacian at point x: Δ[w * G(x; μ, Σ)]

        Formula: ΔG = [tr(Σ⁻¹) - (x-μ)ᵀ Σ⁻¹ Σ⁻¹ (x-μ)] · G

        Args:
            x: Query point, shape (d,)
            check_cutoff: If True, return 0 outside cutoff radius
            eps_rel: Relative tolerance for cutoff

        Returns:
            Laplacian value
        """
        if check_cutoff:
            if self.cutoff_radius is None:
                self.compute_cutoff_radius(eps_rel)

            if np.linalg.norm(x - self.mu) > self.cutoff_radius:
                return 0.0

            d_mahal = self.mahalanobis_distance(x)
            threshold = np.sqrt(-2 * np.log(eps_rel))
            if d_mahal > threshold:
                return 0.0

        diff = x - self.mu
        gaussian_value = self.evaluate(x, check_cutoff=False)

        # ΔG = [tr(Σ⁻¹) - diffᵀ Σ⁻¹ Σ⁻¹ diff] · G
        trace_term = np.trace(self.sigma_inv)
        quadratic_term = diff @ self.sigma_inv @ self.sigma_inv @ diff

        return self.weight * (trace_term - quadratic_term) * gaussian_value / self.weight


class GaussianField:
    """
    Field representation as sum of Gaussian primitives

    u(x) = Σᵢ wᵢ G(x; μᵢ, Σᵢ)
    """

    def __init__(self, primitives: Optional[list] = None):
        """
        Initialize Gaussian field

        Args:
            primitives: List of GaussianPrimitive objects
        """
        self.primitives = primitives if primitives is not None else []
        self._kdtree = None
        self._kdtree_valid = False

    @property
    def num_gaussians(self) -> int:
        """Number of Gaussian primitives"""
        return len(self.primitives)

    @property
    def dim(self) -> int:
        """Spatial dimension"""
        if len(self.primitives) == 0:
            return 0
        return self.primitives[0].dim

    def add_primitive(self, primitive: GaussianPrimitive):
        """Add a Gaussian primitive to the field"""
        self.primitives.append(primitive)
        self._kdtree_valid = False

    def remove_primitive(self, index: int):
        """Remove a Gaussian primitive by index"""
        del self.primitives[index]
        self._kdtree_valid = False

    def get_parameters(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get all parameters as arrays

        Returns:
            weights: shape (N,)
            mus: shape (N, d)
            sigmas: shape (N, d, d)
        """
        N = self.num_gaussians
        d = self.dim

        weights = np.array([p.weight for p in self.primitives])
        mus = np.array([p.mu for p in self.primitives])
        sigmas = np.array([p.sigma for p in self.primitives])

        return weights, mus, sigmas

    def set_parameters(self, weights: np.ndarray, mus: np.ndarray,
                       sigmas: np.ndarray):
        """
        Set parameters from arrays

        Args:
            weights: shape (N,)
            mus: shape (N, d)
            sigmas: shape (N, d, d)
        """
        N = len(weights)
        self.primitives = []

        for i in range(N):
            primitive = GaussianPrimitive(
                weight=weights[i],
                mu=mus[i],
                sigma=sigmas[i]
            )
            self.primitives.append(primitive)

        self._kdtree_valid = False

    def evaluate(self, x: np.ndarray, use_acceleration: bool = False,
                 eps_rel: float = 1e-6) -> float:
        """
        Evaluate field at point x

        Args:
            x: Query point, shape (d,)
            use_acceleration: If True, use spatial acceleration
            eps_rel: Relative tolerance for cutoff

        Returns:
            Field value u(x)
        """
        if use_acceleration:
            # Will be implemented with KDTree
            raise NotImplementedError("Acceleration requires KDTree integration")

        # Naive evaluation: sum over all primitives
        result = 0.0
        for primitive in self.primitives:
            result += primitive.evaluate(x, check_cutoff=True, eps_rel=eps_rel)

        return result

    def gradient(self, x: np.ndarray, use_acceleration: bool = False,
                 eps_rel: float = 1e-6) -> np.ndarray:
        """
        Compute gradient at point x

        Args:
            x: Query point, shape (d,)
            use_acceleration: If True, use spatial acceleration
            eps_rel: Relative tolerance for cutoff

        Returns:
            Gradient ∇u(x), shape (d,)
        """
        if use_acceleration:
            raise NotImplementedError("Acceleration requires KDTree integration")

        # Naive evaluation
        result = np.zeros(self.dim)
        for primitive in self.primitives:
            result += primitive.gradient(x, check_cutoff=True, eps_rel=eps_rel)

        return result

    def laplacian(self, x: np.ndarray, use_acceleration: bool = False,
                  eps_rel: float = 1e-6) -> float:
        """
        Compute Laplacian at point x

        Args:
            x: Query point, shape (d,)
            use_acceleration: If True, use spatial acceleration
            eps_rel: Relative tolerance for cutoff

        Returns:
            Laplacian Δu(x)
        """
        if use_acceleration:
            raise NotImplementedError("Acceleration requires KDTree integration")

        # Naive evaluation
        result = 0.0
        for primitive in self.primitives:
            result += primitive.laplacian(x, check_cutoff=True, eps_rel=eps_rel)

        return result

    def compute_total_weight(self) -> float:
        """Compute W_total = Σ|wᵢ|"""
        return sum(abs(p.weight) for p in self.primitives)

    def update_cutoff_radii(self, eps_rel: float = 1e-6):
        """Update cutoff radii for all primitives"""
        for primitive in self.primitives:
            primitive.compute_cutoff_radius(eps_rel)
