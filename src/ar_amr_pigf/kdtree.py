"""
KD-Tree Spatial Index for Gaussian Primitives
Implements efficient range queries for finding active Gaussians
"""

import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class KDNode:
    """Node in KD-tree"""
    # Leaf node data
    index: Optional[int] = None  # Gaussian index
    mu: Optional[np.ndarray] = None  # Gaussian center
    radius: Optional[float] = None  # Cutoff radius

    # Internal node data
    split_dim: Optional[int] = None
    split_value: Optional[float] = None
    left: Optional['KDNode'] = None
    right: Optional['KDNode'] = None

    # Bounding box (for pruning)
    bbox_min: Optional[np.ndarray] = None
    bbox_max: Optional[np.ndarray] = None

    def is_leaf(self) -> bool:
        """Check if node is a leaf"""
        return self.index is not None


class KDTree:
    """
    KD-Tree for efficient spatial queries on Gaussian primitives

    Supports O(log N + k) range queries where k is the number of results
    """

    def __init__(self, leaf_size: int = 10):
        """
        Initialize KD-tree

        Args:
            leaf_size: Maximum number of points in a leaf node
        """
        self.root = None
        self.leaf_size = leaf_size
        self.dim = None
        self.num_points = 0

    def build(self, mus: np.ndarray, radii: np.ndarray):
        """
        Build KD-tree from Gaussian centers and radii

        Args:
            mus: Gaussian centers, shape (N, d)
            radii: Cutoff radii, shape (N,)
        """
        self.num_points = len(mus)
        self.dim = mus.shape[1]

        # Create list of (mu, radius, index) tuples
        indices = np.arange(self.num_points)
        data = list(zip(mus, radii, indices))

        # Build tree recursively
        self.root = self._build_recursive(data, depth=0)

    def _build_recursive(self, data: List[Tuple], depth: int) -> KDNode:
        """
        Recursively build KD-tree

        Args:
            data: List of (mu, radius, index) tuples
            depth: Current depth in tree

        Returns:
            KDNode representing subtree
        """
        if len(data) == 0:
            return None

        # Compute bounding box
        mus = np.array([d[0] for d in data])
        bbox_min = np.min(mus, axis=0)
        bbox_max = np.max(mus, axis=0)

        # Create leaf if small enough
        if len(data) <= self.leaf_size:
            # For simplicity, create one leaf per point at leaf level
            # In production, could store multiple points per leaf
            if len(data) == 1:
                mu, radius, index = data[0]
                return KDNode(
                    index=index,
                    mu=mu,
                    radius=radius,
                    bbox_min=bbox_min,
                    bbox_max=bbox_max
                )
            else:
                # Split even if at leaf size to maintain tree structure
                pass

        # Choose split dimension (cycle through dimensions)
        split_dim = depth % self.dim

        # Sort by split dimension
        data_sorted = sorted(data, key=lambda x: x[0][split_dim])

        # Find median
        median_idx = len(data_sorted) // 2
        split_value = data_sorted[median_idx][0][split_dim]

        # Split data
        left_data = data_sorted[:median_idx]
        right_data = data_sorted[median_idx:]

        # Create internal node
        node = KDNode(
            split_dim=split_dim,
            split_value=split_value,
            bbox_min=bbox_min,
            bbox_max=bbox_max
        )

        # Recursively build subtrees
        node.left = self._build_recursive(left_data, depth + 1)
        node.right = self._build_recursive(right_data, depth + 1)

        return node

    def range_query(self, query_point: np.ndarray,
                    query_radius: float = 0.0) -> List[int]:
        """
        Find all Gaussians whose support regions intersect with query point

        For each Gaussian i, check if ||query_point - mu_i|| <= radius_i + query_radius

        Args:
            query_point: Query location, shape (d,)
            query_radius: Additional search radius (default 0)

        Returns:
            List of Gaussian indices whose support intersects query point
        """
        if self.root is None:
            return []

        active_set = []
        self._range_query_recursive(
            self.root, query_point, query_radius, active_set
        )

        return active_set

    def _range_query_recursive(self, node: KDNode, query_point: np.ndarray,
                                query_radius: float, active_set: List[int]):
        """
        Recursive range query

        Args:
            node: Current node
            query_point: Query location
            query_radius: Search radius
            active_set: List to append results to (modified in place)
        """
        if node is None:
            return

        # Check if query point could intersect with this node's bounding box
        # (considering query_radius)
        if not self._bbox_intersects_sphere(
            node.bbox_min, node.bbox_max, query_point, query_radius
        ):
            return

        # Leaf node: check if Gaussian intersects
        if node.is_leaf():
            distance = np.linalg.norm(query_point - node.mu)
            if distance <= node.radius + query_radius:
                active_set.append(node.index)
            return

        # Internal node: recurse on both children if they could intersect
        split_dim = node.split_dim
        split_value = node.split_value

        # Check left subtree
        if query_point[split_dim] - query_radius <= split_value:
            self._range_query_recursive(
                node.left, query_point, query_radius, active_set
            )

        # Check right subtree
        if query_point[split_dim] + query_radius >= split_value:
            self._range_query_recursive(
                node.right, query_point, query_radius, active_set
            )

    def _bbox_intersects_sphere(self, bbox_min: np.ndarray, bbox_max: np.ndarray,
                                center: np.ndarray, radius: float) -> bool:
        """
        Check if axis-aligned bounding box intersects with sphere

        Args:
            bbox_min: Min corner of bbox
            bbox_max: Max corner of bbox
            center: Sphere center
            radius: Sphere radius

        Returns:
            True if bbox intersects sphere
        """
        # Find closest point in bbox to sphere center
        closest_point = np.clip(center, bbox_min, bbox_max)

        # Check if closest point is within radius
        distance = np.linalg.norm(center - closest_point)
        return distance <= radius

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
        Get tree statistics for analysis

        Returns:
            Dictionary with tree statistics
        """
        stats = {
            'num_points': self.num_points,
            'dimension': self.dim,
            'max_depth': self._compute_max_depth(self.root),
            'num_leaves': self._count_leaves(self.root),
        }
        return stats

    def _compute_max_depth(self, node: KDNode, current_depth: int = 0) -> int:
        """Compute maximum depth of tree"""
        if node is None:
            return current_depth

        if node.is_leaf():
            return current_depth

        left_depth = self._compute_max_depth(node.left, current_depth + 1)
        right_depth = self._compute_max_depth(node.right, current_depth + 1)

        return max(left_depth, right_depth)

    def _count_leaves(self, node: KDNode) -> int:
        """Count number of leaf nodes"""
        if node is None:
            return 0

        if node.is_leaf():
            return 1

        return self._count_leaves(node.left) + self._count_leaves(node.right)

    def rebuild_if_needed(self, mus: np.ndarray, radii: np.ndarray,
                          change_threshold: float = 0.3) -> bool:
        """
        Rebuild tree if data has changed significantly

        Args:
            mus: New Gaussian centers
            radii: New cutoff radii
            change_threshold: Fraction of changed points to trigger rebuild

        Returns:
            True if tree was rebuilt
        """
        # For now, always rebuild (incremental updates can be added later)
        self.build(mus, radii)
        return True


class KDTreeAdapter:
    """
    Adapter to use scipy.spatial.cKDTree as an alternative

    Provides same interface as our KDTree but uses scipy's implementation
    """

    def __init__(self, leaf_size: int = 10):
        from scipy.spatial import cKDTree
        self.kdtree = None
        self.radii = None
        self.leaf_size = leaf_size
        self.dim = None
        self.num_points = 0

    def build(self, mus: np.ndarray, radii: np.ndarray):
        """Build tree using scipy"""
        from scipy.spatial import cKDTree

        self.num_points = len(mus)
        self.dim = mus.shape[1]
        self.radii = radii

        self.kdtree = cKDTree(mus, leafsize=self.leaf_size)

    def range_query(self, query_point: np.ndarray,
                    query_radius: float = 0.0) -> List[int]:
        """Find active Gaussians using scipy's query_ball_point"""
        if self.kdtree is None:
            return []

        # We need to find all points i where ||query - mu_i|| <= radius_i
        # scipy's query_ball_point finds points within a fixed radius
        # So we need to search with max possible radius
        max_radius = np.max(self.radii) + query_radius

        # Get candidate indices
        candidates = self.kdtree.query_ball_point(query_point, max_radius)

        # Filter by actual radii
        active_set = []
        for idx in candidates:
            mu_i = self.kdtree.data[idx]
            distance = np.linalg.norm(query_point - mu_i)
            if distance <= self.radii[idx] + query_radius:
                active_set.append(idx)

        return active_set

    def batch_range_query(self, query_points: np.ndarray,
                          query_radius: float = 0.0) -> List[List[int]]:
        """Batch range query"""
        results = []
        for query_point in query_points:
            active_set = self.range_query(query_point, query_radius)
            results.append(active_set)
        return results

    def get_statistics(self) -> dict:
        """Get tree statistics"""
        return {
            'num_points': self.num_points,
            'dimension': self.dim,
            'implementation': 'scipy.cKDTree'
        }
