"""
Adaptive Mesh Refinement (AMR) for Gaussian Fields
Implements error estimation, Dörfler marking, and refinement/coarsening operations
"""

import numpy as np
from typing import List, Tuple, Set
from .gaussian import GaussianField, GaussianPrimitive
from .evaluator import FastEvaluator


class AMRRefiner:
    """
    Adaptive mesh refinement for Gaussian fields

    Implements:
    - Local error estimation
    - Dörfler marking strategy
    - Gaussian splitting (refinement)
    - Gaussian merging (coarsening)
    """

    def __init__(self, field: GaussianField, evaluator: FastEvaluator):
        """
        Initialize AMR refiner

        Args:
            field: GaussianField to refine
            evaluator: FastEvaluator for efficient evaluation
        """
        self.field = field
        self.evaluator = evaluator

    def estimate_errors(self, metric: str = 'gradient') -> np.ndarray:
        """
        Estimate local error indicators for each Gaussian

        Formula: η_i = Volume(Σ_i) · ||∇u(μ_i)||

        Args:
            metric: Error metric ('gradient', 'laplacian', or 'residual')

        Returns:
            Error indicators, shape (N,)
        """
        N = self.field.num_gaussians
        errors = np.zeros(N)

        for i, primitive in enumerate(self.field.primitives):
            mu_i = primitive.mu

            if metric == 'gradient':
                # η_i = Volume(Σ_i) · ||∇u(μ_i)||
                grad = self.evaluator.gradient(mu_i, use_acceleration=True)
                grad_norm = np.linalg.norm(grad)

                # Volume ≈ sqrt(det(Σ))
                volume = np.sqrt(primitive.det_sigma)

                errors[i] = volume * grad_norm

            elif metric == 'laplacian':
                # η_i = Volume(Σ_i) · |Δu(μ_i)|
                lapl = self.evaluator.laplacian(mu_i, use_acceleration=True)
                volume = np.sqrt(primitive.det_sigma)

                errors[i] = volume * abs(lapl)

            elif metric == 'residual':
                # Placeholder: would need PDE residual
                errors[i] = np.sqrt(primitive.det_sigma)

            else:
                raise ValueError(f"Unknown metric: {metric}")

        return errors

    def compute_global_error(self, local_errors: np.ndarray) -> float:
        """
        Compute global error indicator

        η_global = sqrt(Σ η_i²)

        Args:
            local_errors: Local error indicators, shape (N,)

        Returns:
            Global error indicator
        """
        return np.sqrt(np.sum(local_errors ** 2))

    def dorfler_marking(self, local_errors: np.ndarray,
                        theta_refine: float = 0.5,
                        theta_coarsen: float = 0.1) -> Tuple[Set[int], Set[int]]:
        """
        Dörfler marking strategy for adaptive refinement

        Refinement: Mark smallest set M such that
            Σ_{i ∈ M} η_i² ≥ θ · Σ_i η_i²

        Coarsening: Mark elements with η_i < θ_c · max(η)

        Args:
            local_errors: Local error indicators, shape (N,)
            theta_refine: Dörfler parameter for refinement (0 < θ ≤ 1)
            theta_coarsen: Threshold for coarsening (0 < θ_c < 1)

        Returns:
            refine_set: Indices to refine
            coarsen_set: Indices to coarsen
        """
        N = len(local_errors)

        if N == 0:
            return set(), set()

        # Sort by error (descending)
        sorted_indices = np.argsort(local_errors)[::-1]
        sorted_errors = local_errors[sorted_indices]

        # Refinement marking (Dörfler strategy)
        total_error_sq = np.sum(local_errors ** 2)
        target_error_sq = theta_refine * total_error_sq

        cumulative_error_sq = 0.0
        refine_set = set()

        for idx in sorted_indices:
            refine_set.add(idx)
            cumulative_error_sq += local_errors[idx] ** 2

            if cumulative_error_sq >= target_error_sq:
                break

        # Coarsening marking (small error elements)
        max_error = np.max(local_errors)
        coarsen_threshold = theta_coarsen * max_error

        coarsen_set = set()
        for i in range(N):
            if local_errors[i] < coarsen_threshold and i not in refine_set:
                coarsen_set.add(i)

        return refine_set, coarsen_set

    def split_gaussian(self, index: int, num_children: int = 2) -> List[GaussianPrimitive]:
        """
        Split a Gaussian into multiple children

        Strategy: Split along largest principal axis

        Args:
            index: Index of Gaussian to split
            num_children: Number of children (default 2)

        Returns:
            List of child Gaussian primitives
        """
        parent = self.field.primitives[index]

        # Find largest principal axis
        eigenvalues, eigenvectors = np.linalg.eigh(parent.sigma)
        max_eig_idx = np.argmax(eigenvalues)
        max_eigenvalue = eigenvalues[max_eig_idx]
        max_eigenvector = eigenvectors[:, max_eig_idx]

        # Create children along principal axis
        children = []

        # Reduce covariance in split direction
        new_eigenvalues = eigenvalues.copy()
        new_eigenvalues[max_eig_idx] = max_eigenvalue / (num_children ** 2)

        # Reconstruct covariance
        new_sigma = eigenvectors @ np.diag(new_eigenvalues) @ eigenvectors.T

        # Position children along principal axis
        offset = np.sqrt(max_eigenvalue) / num_children

        for i in range(num_children):
            # Offset from parent center
            t = (i - (num_children - 1) / 2) * offset
            new_mu = parent.mu + t * max_eigenvector

            # Equal weight distribution
            new_weight = parent.weight / num_children

            child = GaussianPrimitive(
                weight=new_weight,
                mu=new_mu,
                sigma=new_sigma
            )

            children.append(child)

        return children

    def merge_gaussians(self, indices: List[int]) -> GaussianPrimitive:
        """
        Merge multiple Gaussians into one

        Strategy: Weighted average of parameters

        Args:
            indices: Indices of Gaussians to merge

        Returns:
            Merged Gaussian primitive
        """
        if len(indices) == 0:
            raise ValueError("Cannot merge empty list")

        if len(indices) == 1:
            return self.field.primitives[indices[0]]

        # Collect weights and parameters
        weights = []
        mus = []
        sigmas = []

        for idx in indices:
            p = self.field.primitives[idx]
            weights.append(abs(p.weight))
            mus.append(p.mu)
            sigmas.append(p.sigma)

        weights = np.array(weights)
        total_weight = np.sum(weights)

        # Weighted average of centers
        merged_mu = np.average(mus, axis=0, weights=weights)

        # Weighted average of covariances + variance of centers
        merged_sigma = np.average(sigmas, axis=0, weights=weights)

        # Add covariance from center dispersion
        for i, mu in enumerate(mus):
            diff = mu - merged_mu
            merged_sigma += weights[i] / total_weight * np.outer(diff, diff)

        # Sum of weights (with signs)
        merged_weight = sum(self.field.primitives[idx].weight for idx in indices)

        return GaussianPrimitive(
            weight=merged_weight,
            mu=merged_mu,
            sigma=merged_sigma
        )

    def refine(self, refine_indices: Set[int], num_children: int = 2):
        """
        Refine marked Gaussians

        Args:
            refine_indices: Set of indices to refine
            num_children: Number of children per split
        """
        # Sort indices in descending order to avoid index shifts
        sorted_indices = sorted(refine_indices, reverse=True)

        for idx in sorted_indices:
            # Split Gaussian
            children = self.split_gaussian(idx, num_children=num_children)

            # Remove parent
            self.field.remove_primitive(idx)

            # Add children
            for child in children:
                self.field.add_primitive(child)

        # Invalidate evaluator tree
        self.evaluator.invalidate_tree()

    def coarsen(self, coarsen_indices: Set[int], merge_radius: float = None):
        """
        Coarsen marked Gaussians by merging nearby ones

        Args:
            coarsen_indices: Set of indices to coarsen
            merge_radius: Radius for grouping nearby Gaussians (auto if None)
        """
        if len(coarsen_indices) == 0:
            return

        # Convert to list
        indices_list = list(coarsen_indices)

        # Auto-determine merge radius
        if merge_radius is None:
            # Use average cutoff radius
            radii = [self.field.primitives[i].cutoff_radius
                     for i in indices_list
                     if self.field.primitives[i].cutoff_radius is not None]
            if len(radii) > 0:
                merge_radius = np.mean(radii) * 2
            else:
                merge_radius = 0.1  # Default

        # Group nearby Gaussians
        groups = self._group_nearby_gaussians(indices_list, merge_radius)

        # Merge each group
        # Process in reverse order to avoid index issues
        all_to_remove = set()

        for group in groups:
            if len(group) < 2:
                continue

            # Merge group
            merged = self.merge_gaussians(list(group))

            # Mark for removal
            all_to_remove.update(group)

            # Add merged Gaussian
            self.field.add_primitive(merged)

        # Remove old Gaussians (in descending order)
        for idx in sorted(all_to_remove, reverse=True):
            self.field.remove_primitive(idx)

        # Invalidate tree
        self.evaluator.invalidate_tree()

    def _group_nearby_gaussians(self, indices: List[int],
                                 radius: float) -> List[Set[int]]:
        """
        Group Gaussians that are within radius of each other

        Uses simple greedy clustering

        Args:
            indices: Gaussian indices
            radius: Grouping radius

        Returns:
            List of groups (each group is a set of indices)
        """
        unassigned = set(indices)
        groups = []

        while unassigned:
            # Start new group
            seed = unassigned.pop()
            group = {seed}

            # Find nearby Gaussians
            seed_mu = self.field.primitives[seed].mu

            to_check = list(unassigned)
            for idx in to_check:
                mu = self.field.primitives[idx].mu
                if np.linalg.norm(mu - seed_mu) <= radius:
                    group.add(idx)
                    unassigned.remove(idx)

            groups.append(group)

        return groups

    def adapt_step(self, tol: float, theta_refine: float = 0.5,
                   theta_coarsen: float = 0.1,
                   metric: str = 'gradient') -> Tuple[float, bool]:
        """
        Perform one adaptation step

        Returns:
            global_error: Current global error
            converged: True if error below tolerance
        """
        # Estimate errors
        local_errors = self.estimate_errors(metric=metric)
        global_error = self.compute_global_error(local_errors)

        # Check convergence
        if global_error < tol:
            return global_error, True

        # Mark for refinement/coarsening
        refine_set, coarsen_set = self.dorfler_marking(
            local_errors, theta_refine, theta_coarsen
        )

        # Refine
        if len(refine_set) > 0:
            self.refine(refine_set, num_children=2)

        # Coarsen
        if len(coarsen_set) > 0:
            self.coarsen(coarsen_set)

        return global_error, False
