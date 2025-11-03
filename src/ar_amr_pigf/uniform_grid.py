"""
Uniform Grid Spatial Index for GPU Acceleration
Replaces KD-tree with simple grid structure that's GPU-friendly
"""

import numpy as np
import torch
from typing import List, Tuple, Optional
from dataclasses import dataclass


class UniformGrid:
    """
    Uniform grid spatial index for Gaussian primitives

    Much simpler than KD-tree and better suited for GPU parallelization:
    - O(1) cell lookup
    - Easy to vectorize
    - GPU-friendly data structure

    Grid divides domain into cells. Each cell stores indices of Gaussians
    whose support intersects that cell.
    """

    def __init__(self, domain_min: np.ndarray, domain_max: np.ndarray,
                 cell_size: float = None, cells_per_dim: int = None):
        """
        Initialize uniform grid

        Args:
            domain_min: Minimum corner of domain, shape (d,)
            domain_max: Maximum corner of domain, shape (d,)
            cell_size: Size of each grid cell (auto if None)
            cells_per_dim: Number of cells per dimension (auto if None)
        """
        self.domain_min = domain_min
        self.domain_max = domain_max
        self.dim = len(domain_min)

        self.domain_size = domain_max - domain_min

        # Determine grid resolution
        if cell_size is not None:
            self.cell_size = cell_size
            self.cells_per_dim = tuple(
                int(np.ceil(self.domain_size[i] / cell_size))
                for i in range(self.dim)
            )
        elif cells_per_dim is not None:
            if isinstance(cells_per_dim, int):
                self.cells_per_dim = tuple([cells_per_dim] * self.dim)
            else:
                self.cells_per_dim = tuple(cells_per_dim)
            self.cell_size = np.min(self.domain_size / np.array(self.cells_per_dim))
        else:
            # Auto: aim for ~10-20 cells per dimension
            self.cells_per_dim = tuple([20] * self.dim)
            self.cell_size = np.min(self.domain_size / np.array(self.cells_per_dim))

        # Cell size vector
        self.cell_sizes = self.domain_size / np.array(self.cells_per_dim)

        # Grid storage: dict mapping cell index to list of Gaussian indices
        self.grid = {}
        self.num_gaussians = 0

        # Statistics
        self.build_time = 0.0

    def _point_to_cell(self, point: np.ndarray) -> Tuple[int, ...]:
        """
        Convert point to cell indices

        Args:
            point: Point in space, shape (d,)

        Returns:
            Cell indices, tuple of ints
        """
        # Clamp to domain
        point = np.clip(point, self.domain_min, self.domain_max)

        # Compute cell indices
        indices = ((point - self.domain_min) / self.cell_sizes).astype(int)

        # Clamp to valid range
        indices = np.minimum(indices, np.array(self.cells_per_dim) - 1)

        return tuple(indices)

    def _get_cells_in_sphere(self, center: np.ndarray, radius: float) -> List[Tuple[int, ...]]:
        """
        Get all grid cells that intersect with sphere

        Args:
            center: Sphere center, shape (d,)
            radius: Sphere radius

        Returns:
            List of cell indices
        """
        # Bounding box of sphere
        bbox_min = np.maximum(center - radius, self.domain_min)
        bbox_max = np.minimum(center + radius, self.domain_max)

        # Cell range
        cell_min = self._point_to_cell(bbox_min)
        cell_max = self._point_to_cell(bbox_max)

        # Generate all cells in range
        cells = []

        if self.dim == 1:
            for i in range(cell_min[0], cell_max[0] + 1):
                cells.append((i,))
        elif self.dim == 2:
            for i in range(cell_min[0], cell_max[0] + 1):
                for j in range(cell_min[1], cell_max[1] + 1):
                    cells.append((i, j))
        elif self.dim == 3:
            for i in range(cell_min[0], cell_max[0] + 1):
                for j in range(cell_min[1], cell_max[1] + 1):
                    for k in range(cell_min[2], cell_max[2] + 1):
                        cells.append((i, j, k))
        else:
            # General case (slower)
            import itertools
            ranges = [range(cell_min[i], cell_max[i] + 1) for i in range(self.dim)]
            cells = list(itertools.product(*ranges))

        return cells

    def build(self, mus: np.ndarray, radii: np.ndarray):
        """
        Build grid from Gaussian centers and radii

        Args:
            mus: Gaussian centers, shape (N, d)
            radii: Cutoff radii, shape (N,)
        """
        import time
        start_time = time.time()

        self.num_gaussians = len(mus)
        self.grid = {}

        # Insert each Gaussian into grid
        for idx in range(self.num_gaussians):
            mu = mus[idx]
            radius = radii[idx]

            # Find all cells this Gaussian intersects
            cells = self._get_cells_in_sphere(mu, radius)

            # Add Gaussian to each cell
            for cell in cells:
                if cell not in self.grid:
                    self.grid[cell] = []
                self.grid[cell].append(idx)

        self.build_time = time.time() - start_time

    def range_query(self, query_point: np.ndarray,
                    query_radius: float = 0.0) -> List[int]:
        """
        Find all Gaussians whose support intersects with query point

        Args:
            query_point: Query location, shape (d,)
            query_radius: Additional search radius (default 0)

        Returns:
            List of Gaussian indices
        """
        # Find cell containing query point
        cell = self._point_to_cell(query_point)

        # If query_radius > 0, need to check neighboring cells
        if query_radius > 0:
            cells = self._get_cells_in_sphere(query_point, query_radius)
        else:
            cells = [cell]

        # Collect all Gaussians from these cells
        candidates = set()
        for c in cells:
            if c in self.grid:
                candidates.update(self.grid[c])

        return list(candidates)

    def batch_range_query(self, query_points: np.ndarray,
                          query_radius: float = 0.0) -> List[List[int]]:
        """
        Perform range queries for multiple points

        Args:
            query_points: Query locations, shape (M, d)
            query_radius: Search radius

        Returns:
            List of active sets for each query point
        """
        results = []
        for query_point in query_points:
            active_set = self.range_query(query_point, query_radius)
            results.append(active_set)

        return results

    def get_statistics(self) -> dict:
        """
        Get grid statistics

        Returns:
            Dictionary with statistics
        """
        if len(self.grid) == 0:
            return {
                'num_gaussians': self.num_gaussians,
                'num_cells': 0,
                'avg_gaussians_per_cell': 0,
                'max_gaussians_per_cell': 0,
                'cells_per_dim': self.cells_per_dim,
                'cell_size': self.cell_size,
            }

        cell_sizes = [len(self.grid[cell]) for cell in self.grid]

        return {
            'num_gaussians': self.num_gaussians,
            'num_cells': len(self.grid),
            'avg_gaussians_per_cell': np.mean(cell_sizes),
            'max_gaussians_per_cell': np.max(cell_sizes),
            'cells_per_dim': self.cells_per_dim,
            'cell_size': self.cell_size,
            'build_time': self.build_time,
        }

    def to_dense_tensor(self, device: torch.device = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Convert grid to dense tensors for GPU processing

        Returns:
            cell_indices: Tensor of Gaussian indices per cell, shape (num_cells, max_per_cell)
                         Padded with -1 for empty slots
            cell_counts: Number of Gaussians per cell, shape (num_cells,)
        """
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if len(self.grid) == 0:
            return torch.empty(0, 0, dtype=torch.long, device=device), \
                   torch.empty(0, dtype=torch.long, device=device)

        # Find max Gaussians per cell
        max_per_cell = max(len(self.grid[cell]) for cell in self.grid)
        num_cells = len(self.grid)

        # Create dense arrays
        cell_indices_list = []
        cell_counts = []

        for cell in sorted(self.grid.keys()):
            indices = self.grid[cell]
            # Pad with -1
            padded = indices + [-1] * (max_per_cell - len(indices))
            cell_indices_list.append(padded)
            cell_counts.append(len(indices))

        cell_indices = torch.tensor(cell_indices_list, dtype=torch.long, device=device)
        cell_counts = torch.tensor(cell_counts, dtype=torch.long, device=device)

        return cell_indices, cell_counts

    def get_cell_for_points_batch(self, points: torch.Tensor) -> torch.Tensor:
        """
        Get cell indices for batch of points (GPU-friendly)

        Args:
            points: Points, shape (M, d)

        Returns:
            Cell indices, shape (M, d)
        """
        # Clamp to domain
        domain_min = torch.tensor(self.domain_min, device=points.device, dtype=points.dtype)
        domain_max = torch.tensor(self.domain_max, device=points.device, dtype=points.dtype)
        cell_sizes = torch.tensor(self.cell_sizes, device=points.device, dtype=points.dtype)
        cells_per_dim = torch.tensor(self.cells_per_dim, device=points.device, dtype=torch.long)

        points_clamped = torch.clamp(points, domain_min, domain_max)

        # Compute cell indices
        indices = ((points_clamped - domain_min) / cell_sizes).long()

        # Clamp to valid range
        indices = torch.minimum(indices, cells_per_dim - 1)

        return indices


class GridConfig:
    """Configuration for uniform grid"""

    def __init__(self,
                 cells_per_dim: int = 20,
                 auto_adjust: bool = True,
                 target_gaussians_per_cell: int = 10):
        """
        Args:
            cells_per_dim: Initial cells per dimension
            auto_adjust: Automatically adjust based on Gaussian density
            target_gaussians_per_cell: Target average Gaussians per cell
        """
        self.cells_per_dim = cells_per_dim
        self.auto_adjust = auto_adjust
        self.target_gaussians_per_cell = target_gaussians_per_cell

    def compute_optimal_cells(self, num_gaussians: int,
                             avg_radius: float,
                             domain_size: np.ndarray) -> int:
        """
        Compute optimal number of cells per dimension

        Based on:
        - Number of Gaussians
        - Average Gaussian radius
        - Domain size

        Goal: Each cell contains ~target_gaussians_per_cell Gaussians
        """
        dim = len(domain_size)
        domain_volume = np.prod(domain_size)

        # Estimate average Gaussian volume
        gaussian_volume = (np.pi ** (dim / 2)) * (avg_radius ** dim)

        # Total "coverage volume"
        total_coverage = num_gaussians * gaussian_volume

        # Density
        density = total_coverage / domain_volume

        # Cell size to achieve target
        # cell_volume * density ≈ target_gaussians_per_cell
        target_cell_volume = self.target_gaussians_per_cell / max(density, 1e-6)

        # Cell size
        target_cell_size = target_cell_volume ** (1.0 / dim)

        # Cells per dimension
        cells = max(int(np.min(domain_size) / target_cell_size), 10)
        cells = min(cells, 100)  # Upper limit

        return cells
