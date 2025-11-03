"""
GPU-Accelerated Evaluator using Uniform Grid and PyTorch
Replaces KD-tree with GPU-friendly operations
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Optional, Tuple
import time

from .gaussian import GaussianField
from .uniform_grid import UniformGrid, GridConfig


class GPUGaussianField(nn.Module):
    """
    GPU-accelerated Gaussian field using PyTorch

    Stores all parameters as CUDA tensors for fast batch evaluation
    """

    def __init__(self, field: GaussianField = None,
                 device: torch.device = None):
        """
        Initialize GPU field

        Args:
            field: CPU GaussianField to convert (optional)
            device: CUDA device (auto if None)
        """
        super().__init__()

        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device

        if field is not None:
            self.from_cpu_field(field)
        else:
            # Empty field
            self.weights = torch.empty(0, device=self.device, dtype=torch.float32)
            self.mus = torch.empty(0, 0, device=self.device, dtype=torch.float32)
            self.sigmas = torch.empty(0, 0, 0, device=self.device, dtype=torch.float32)
            self.sigma_invs = torch.empty(0, 0, 0, device=self.device, dtype=torch.float32)
            self.normalizers = torch.empty(0, device=self.device, dtype=torch.float32)
            self.cutoff_radii = torch.empty(0, device=self.device, dtype=torch.float32)

    def from_cpu_field(self, field: GaussianField):
        """
        Convert CPU GaussianField to GPU tensors

        Args:
            field: CPU GaussianField
        """
        N = field.num_gaussians
        if N == 0:
            return

        d = field.dim

        # Collect parameters
        weights, mus, sigmas = field.get_parameters()

        # Convert to tensors
        self.weights = torch.tensor(weights, device=self.device, dtype=torch.float32)
        self.mus = torch.tensor(mus, device=self.device, dtype=torch.float32)
        self.sigmas = torch.tensor(sigmas, device=self.device, dtype=torch.float32)

        # Precompute inverses and normalizers
        self.sigma_invs = torch.zeros_like(self.sigmas)
        self.normalizers = torch.zeros(N, device=self.device, dtype=torch.float32)

        for i in range(N):
            # Inverse
            self.sigma_invs[i] = torch.inverse(self.sigmas[i])

            # Normalizer: (2π)^(-d/2) |Σ|^(-1/2)
            det = torch.det(self.sigmas[i])
            self.normalizers[i] = 1.0 / torch.sqrt((2 * np.pi) ** d * det)

        # Cutoff radii
        cutoff_radii = [p.cutoff_radius if p.cutoff_radius is not None else 0.1
                       for p in field.primitives]
        self.cutoff_radii = torch.tensor(cutoff_radii, device=self.device, dtype=torch.float32)

    def to_cpu_field(self) -> GaussianField:
        """
        Convert GPU field back to CPU GaussianField

        Returns:
            CPU GaussianField
        """
        from .gaussian import GaussianPrimitive, GaussianField

        field = GaussianField()

        N = len(self.weights)
        for i in range(N):
            primitive = GaussianPrimitive(
                weight=self.weights[i].item(),
                mu=self.mus[i].cpu().numpy(),
                sigma=self.sigmas[i].cpu().numpy()
            )
            field.add_primitive(primitive)

        return field

    @property
    def num_gaussians(self) -> int:
        return len(self.weights)

    @property
    def dim(self) -> int:
        return self.mus.shape[1] if len(self.mus) > 0 else 0

    def evaluate_batch(self, X: torch.Tensor,
                      active_mask: Optional[torch.Tensor] = None,
                      eps_rel: float = 1e-6) -> torch.Tensor:
        """
        Evaluate field at batch of points (GPU-accelerated)

        Args:
            X: Query points, shape (M, d)
            active_mask: Binary mask of active Gaussians per point, shape (M, N)
                        If None, evaluate all Gaussians
            eps_rel: Relative tolerance for cutoff

        Returns:
            Field values, shape (M,)
        """
        M = X.shape[0]
        N = self.num_gaussians
        d = self.dim

        if N == 0:
            return torch.zeros(M, device=self.device)

        # Expand dimensions for broadcasting
        # X: (M, d) -> (M, 1, d)
        # mus: (N, d) -> (1, N, d)
        X_expanded = X.unsqueeze(1)  # (M, 1, d)
        mus_expanded = self.mus.unsqueeze(0)  # (1, N, d)

        # Differences: (M, N, d)
        diffs = X_expanded - mus_expanded

        # Mahalanobis distances: (M, N)
        # For each (m, n): diff^T Sigma_inv diff
        sigma_invs_expanded = self.sigma_invs.unsqueeze(0)  # (1, N, d, d)

        # Compute: diff @ sigma_inv @ diff^T
        # diffs: (M, N, d)
        # sigma_invs: (1, N, d, d)
        temp = torch.einsum('mni,nij->mnj', diffs, self.sigma_invs)  # (M, N, d)
        mahal_sq = torch.einsum('mni,mni->mn', temp, diffs)  # (M, N)

        # Cutoff check
        threshold_sq = -2 * np.log(eps_rel)
        cutoff_mask = mahal_sq <= threshold_sq  # (M, N)

        # Apply active mask if provided
        if active_mask is not None:
            cutoff_mask = cutoff_mask & active_mask

        # Gaussian values: (M, N)
        exponent = -0.5 * mahal_sq
        gaussian_values = self.normalizers.unsqueeze(0) * torch.exp(exponent)

        # Apply mask
        gaussian_values = gaussian_values * cutoff_mask.float()

        # Weighted sum: (M,)
        result = torch.sum(self.weights.unsqueeze(0) * gaussian_values, dim=1)

        return result

    def gradient_batch(self, X: torch.Tensor,
                      active_mask: Optional[torch.Tensor] = None,
                      eps_rel: float = 1e-6) -> torch.Tensor:
        """
        Compute gradients at batch of points (GPU-accelerated)

        Args:
            X: Query points, shape (M, d)
            active_mask: Binary mask, shape (M, N)
            eps_rel: Relative tolerance

        Returns:
            Gradients, shape (M, d)
        """
        M = X.shape[0]
        N = self.num_gaussians
        d = self.dim

        if N == 0:
            return torch.zeros(M, d, device=self.device)

        # Expand dimensions
        X_expanded = X.unsqueeze(1)  # (M, 1, d)
        mus_expanded = self.mus.unsqueeze(0)  # (1, N, d)
        diffs = X_expanded - mus_expanded  # (M, N, d)

        # Mahalanobis distances
        temp = torch.einsum('mni,nij->mnj', diffs, self.sigma_invs)  # (M, N, d)
        mahal_sq = torch.einsum('mni,mni->mn', temp, diffs)  # (M, N)

        # Cutoff
        threshold_sq = -2 * np.log(eps_rel)
        cutoff_mask = mahal_sq <= threshold_sq

        if active_mask is not None:
            cutoff_mask = cutoff_mask & active_mask

        # Gaussian values
        exponent = -0.5 * mahal_sq
        gaussian_values = self.normalizers.unsqueeze(0) * torch.exp(exponent)  # (M, N)
        gaussian_values = gaussian_values * cutoff_mask.float()

        # Gradient: -Σ^{-1}(x-μ) · G(x)
        # temp: (M, N, d) already contains Σ^{-1}(x-μ)
        # gradient_per_gaussian: (M, N, d)
        weights_expanded = self.weights.unsqueeze(0).unsqueeze(2)  # (1, N, 1)
        gaussian_values_expanded = gaussian_values.unsqueeze(2)  # (M, N, 1)

        gradient_per_gaussian = -weights_expanded * temp * gaussian_values_expanded

        # Sum over Gaussians: (M, d)
        gradient = torch.sum(gradient_per_gaussian, dim=1)

        return gradient

    def laplacian_batch(self, X: torch.Tensor,
                       active_mask: Optional[torch.Tensor] = None,
                       eps_rel: float = 1e-6) -> torch.Tensor:
        """
        Compute Laplacians at batch of points (GPU-accelerated)

        Args:
            X: Query points, shape (M, d)
            active_mask: Binary mask, shape (M, N)
            eps_rel: Relative tolerance

        Returns:
            Laplacians, shape (M,)
        """
        M = X.shape[0]
        N = self.num_gaussians

        if N == 0:
            return torch.zeros(M, device=self.device)

        # Similar computation as above
        X_expanded = X.unsqueeze(1)
        mus_expanded = self.mus.unsqueeze(0)
        diffs = X_expanded - mus_expanded

        # Mahalanobis distances
        temp = torch.einsum('mni,nij->mnj', diffs, self.sigma_invs)
        mahal_sq = torch.einsum('mni,mni->mn', temp, diffs)

        # Cutoff
        threshold_sq = -2 * np.log(eps_rel)
        cutoff_mask = mahal_sq <= threshold_sq

        if active_mask is not None:
            cutoff_mask = cutoff_mask & active_mask

        # Gaussian values
        exponent = -0.5 * mahal_sq
        gaussian_values = self.normalizers.unsqueeze(0) * torch.exp(exponent)
        gaussian_values = gaussian_values * cutoff_mask.float()

        # Laplacian: [tr(Σ^{-1}) - diff^T Σ^{-1} Σ^{-1} diff] · G
        # Trace term: (N,)
        trace_term = torch.diagonal(self.sigma_invs, dim1=1, dim2=2).sum(dim=1)  # (N,)

        # Quadratic term: diff^T Σ^{-1} Σ^{-1} diff
        # temp: (M, N, d) = Σ^{-1} diff
        # temp2: (M, N, d) = Σ^{-1} temp
        temp2 = torch.einsum('mni,nij->mnj', temp, self.sigma_invs)  # (M, N, d)
        quadratic_term = torch.einsum('mni,mni->mn', temp2, diffs)  # (M, N)

        # Laplacian per Gaussian: (M, N)
        lapl_per_gaussian = self.weights.unsqueeze(0) * \
                           (trace_term.unsqueeze(0) - quadratic_term) * \
                           gaussian_values

        # Sum: (M,)
        laplacian = torch.sum(lapl_per_gaussian, dim=1)

        return laplacian


class GPUEvaluator:
    """
    GPU-accelerated evaluator using Uniform Grid

    Much faster than CPU KD-tree for large batches
    """

    def __init__(self, field: GaussianField,
                 grid_config: GridConfig = None,
                 eps_rel: float = 1e-6,
                 device: torch.device = None):
        """
        Initialize GPU evaluator

        Args:
            field: CPU GaussianField
            grid_config: Grid configuration
            eps_rel: Relative tolerance
            device: CUDA device
        """
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device

        self.eps_rel = eps_rel
        self.grid_config = grid_config if grid_config is not None else GridConfig()

        # Convert field to GPU
        self.gpu_field = GPUGaussianField(field, device=self.device)

        # Build uniform grid (CPU for now, indices only)
        self.grid = None
        self.domain_min = None
        self.domain_max = None

        # Statistics
        self.stats = {
            'num_evaluations': 0,
            'total_time': 0.0,
            'build_time': 0.0,
        }

    def build_grid(self, domain_min: np.ndarray, domain_max: np.ndarray):
        """
        Build uniform grid

        Args:
            domain_min: Domain minimum corner
            domain_max: Domain maximum corner
        """
        start_time = time.time()

        self.domain_min = domain_min
        self.domain_max = domain_max

        # Auto-adjust grid resolution
        if self.grid_config.auto_adjust and self.gpu_field.num_gaussians > 0:
            avg_radius = torch.mean(self.gpu_field.cutoff_radii).item()
            cells_per_dim = self.grid_config.compute_optimal_cells(
                self.gpu_field.num_gaussians,
                avg_radius,
                domain_max - domain_min
            )
        else:
            cells_per_dim = self.grid_config.cells_per_dim

        # Create grid
        self.grid = UniformGrid(
            domain_min, domain_max,
            cells_per_dim=cells_per_dim
        )

        # Build grid with Gaussian centers and radii
        if self.gpu_field.num_gaussians > 0:
            mus = self.gpu_field.mus.cpu().numpy()
            radii = self.gpu_field.cutoff_radii.cpu().numpy()
            self.grid.build(mus, radii)

        self.stats['build_time'] = time.time() - start_time

    def evaluate_batch(self, X: np.ndarray,
                      use_grid: bool = True) -> np.ndarray:
        """
        Evaluate field at batch of points

        Args:
            X: Query points, shape (M, d), numpy array
            use_grid: If True, use grid acceleration

        Returns:
            Field values, shape (M,), numpy array
        """
        start_time = time.time()

        # Convert to torch
        X_torch = torch.tensor(X, device=self.device, dtype=torch.float32)

        if use_grid and self.grid is not None:
            # Use grid acceleration
            M = len(X)
            N = self.gpu_field.num_gaussians

            # Create active mask: (M, N)
            active_mask = torch.zeros(M, N, device=self.device, dtype=torch.bool)

            # Query grid for each point (CPU for now)
            for m in range(M):
                active_indices = self.grid.range_query(X[m])
                if len(active_indices) > 0:
                    active_mask[m, active_indices] = True

            # Evaluate with mask
            result = self.gpu_field.evaluate_batch(X_torch, active_mask, self.eps_rel)
        else:
            # Evaluate all Gaussians
            result = self.gpu_field.evaluate_batch(X_torch, None, self.eps_rel)

        self.stats['num_evaluations'] += len(X)
        self.stats['total_time'] += time.time() - start_time

        return result.cpu().numpy()

    def gradient_batch(self, X: np.ndarray,
                      use_grid: bool = True) -> np.ndarray:
        """
        Compute gradients at batch of points

        Args:
            X: Query points, shape (M, d)
            use_grid: Use grid acceleration

        Returns:
            Gradients, shape (M, d)
        """
        X_torch = torch.tensor(X, device=self.device, dtype=torch.float32)

        if use_grid and self.grid is not None:
            M = len(X)
            N = self.gpu_field.num_gaussians

            active_mask = torch.zeros(M, N, device=self.device, dtype=torch.bool)
            for m in range(M):
                active_indices = self.grid.range_query(X[m])
                if len(active_indices) > 0:
                    active_mask[m, active_indices] = True

            result = self.gpu_field.gradient_batch(X_torch, active_mask, self.eps_rel)
        else:
            result = self.gpu_field.gradient_batch(X_torch, None, self.eps_rel)

        return result.cpu().numpy()

    def laplacian_batch(self, X: np.ndarray,
                       use_grid: bool = True) -> np.ndarray:
        """
        Compute Laplacians at batch of points

        Args:
            X: Query points, shape (M, d)
            use_grid: Use grid acceleration

        Returns:
            Laplacians, shape (M,)
        """
        X_torch = torch.tensor(X, device=self.device, dtype=torch.float32)

        if use_grid and self.grid is not None:
            M = len(X)
            N = self.gpu_field.num_gaussians

            active_mask = torch.zeros(M, N, device=self.device, dtype=torch.bool)
            for m in range(M):
                active_indices = self.grid.range_query(X[m])
                if len(active_indices) > 0:
                    active_mask[m, active_indices] = True

            result = self.gpu_field.laplacian_batch(X_torch, active_mask, self.eps_rel)
        else:
            result = self.gpu_field.laplacian_batch(X_torch, None, self.eps_rel)

        return result.cpu().numpy()

    def get_statistics(self) -> dict:
        """Get performance statistics"""
        stats = self.stats.copy()

        if self.grid is not None:
            grid_stats = self.grid.get_statistics()
            stats.update({
                'grid_cells': grid_stats['num_cells'],
                'avg_gaussians_per_cell': grid_stats['avg_gaussians_per_cell'],
                'max_gaussians_per_cell': grid_stats['max_gaussians_per_cell'],
            })

        stats['device'] = str(self.device)
        stats['num_gaussians'] = self.gpu_field.num_gaussians

        return stats

    def benchmark(self, X: np.ndarray, num_runs: int = 10) -> dict:
        """
        Benchmark GPU vs CPU evaluation

        Args:
            X: Test points
            num_runs: Number of runs

        Returns:
            Benchmark results
        """
        print(f"\nGPU Benchmark: {len(X)} points, {self.gpu_field.num_gaussians} Gaussians")
        print(f"Device: {self.device}")

        # Warm up
        _ = self.evaluate_batch(X[:10], use_grid=True)
        if self.device.type == 'cuda':
            torch.cuda.synchronize()

        # GPU with grid
        times_gpu_grid = []
        for _ in range(num_runs):
            start = time.time()
            _ = self.evaluate_batch(X, use_grid=True)
            if self.device.type == 'cuda':
                torch.cuda.synchronize()
            times_gpu_grid.append(time.time() - start)

        gpu_grid_time = np.mean(times_gpu_grid)

        # GPU without grid
        times_gpu_no_grid = []
        for _ in range(num_runs):
            start = time.time()
            _ = self.evaluate_batch(X, use_grid=False)
            if self.device.type == 'cuda':
                torch.cuda.synchronize()
            times_gpu_no_grid.append(time.time() - start)

        gpu_no_grid_time = np.mean(times_gpu_no_grid)

        speedup = gpu_no_grid_time / gpu_grid_time

        results = {
            'num_points': len(X),
            'num_gaussians': self.gpu_field.num_gaussians,
            'gpu_grid_time': gpu_grid_time,
            'gpu_no_grid_time': gpu_no_grid_time,
            'speedup': speedup,
            'device': str(self.device),
        }

        print(f"\nResults:")
        print(f"  GPU (with grid): {gpu_grid_time:.4f}s")
        print(f"  GPU (no grid):   {gpu_no_grid_time:.4f}s")
        print(f"  Grid speedup:    {speedup:.2f}x")

        return results
