"""
AR-AMR-PIGF Solver
Implements the complete adaptive refinement solver with time stepping
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Callable, List, Tuple, Optional, Dict
from dataclasses import dataclass
import copy

from .gaussian import GaussianField, GaussianPrimitive
from .evaluator import FastEvaluator
from .amr import AMRRefiner


@dataclass
class SolverConfig:
    """Configuration for AR-AMR-PIGF solver"""
    # Time stepping
    dt: float = 0.01
    t_final: float = 1.0

    # Spatial domain
    domain_min: np.ndarray = None
    domain_max: np.ndarray = None

    # Optimization
    num_collocation_points: int = 1000
    num_solve_iters: int = 50
    learning_rate: float = 0.01
    optimizer: str = 'adam'  # 'adam', 'lbfgs', 'sgd'

    # AMR parameters
    max_adapt_iters: int = 10
    tol_step: float = 1e-3
    theta_refine: float = 0.5
    theta_coarsen: float = 0.1
    error_metric: str = 'gradient'

    # Acceleration
    eps_rel: float = 1e-6
    use_acceleration: bool = True
    use_scipy_kdtree: bool = True

    # Constraints
    max_gaussians: int = 10000
    min_gaussians: int = 10

    def __post_init__(self):
        """Validate configuration"""
        if self.domain_min is None:
            self.domain_min = np.array([0.0])
        if self.domain_max is None:
            self.domain_max = np.array([1.0])


class PDEProblem:
    """
    Abstract base class for PDE problems

    Subclasses should implement:
    - initial_condition(x)
    - boundary_condition(x, t)
    - residual(u, u_old, grad_u, grad_u_old, lapl_u, lapl_u_old, dt)
    """

    def __init__(self, config: SolverConfig):
        self.config = config
        self.dim = len(config.domain_min)

    def initial_condition(self, x: np.ndarray) -> float:
        """Evaluate initial condition at x"""
        raise NotImplementedError

    def boundary_condition(self, x: np.ndarray, t: float) -> float:
        """Evaluate boundary condition at x, t"""
        raise NotImplementedError

    def residual(self, u_new: float, u_old: float,
                 grad_u_new: np.ndarray, grad_u_old: np.ndarray,
                 lapl_u_new: float, lapl_u_old: float,
                 dt: float, x: np.ndarray, t: float) -> float:
        """
        Compute PDE residual for Crank-Nicolson scheme

        R = (u_new - u_old)/dt + 0.5*(N[u_new] + N[u_old])

        where N is the spatial operator

        Args:
            u_new, u_old: Solution values
            grad_u_new, grad_u_old: Gradients
            lapl_u_new, lapl_u_old: Laplacians
            dt: Time step
            x: Spatial location
            t: Current time

        Returns:
            Residual value
        """
        raise NotImplementedError

    def is_on_boundary(self, x: np.ndarray, tol: float = 1e-6) -> bool:
        """Check if point is on domain boundary"""
        on_boundary = False

        for i in range(self.dim):
            if (abs(x[i] - self.config.domain_min[i]) < tol or
                abs(x[i] - self.config.domain_max[i]) < tol):
                on_boundary = True
                break

        return on_boundary


class ARSolver:
    """
    Adaptive Refinement solver for time-dependent PDEs

    Implements Algorithm 6.1 (SOLVE_ADAPT_Accelerated)
    """

    def __init__(self, problem: PDEProblem, config: SolverConfig):
        """
        Initialize AR solver

        Args:
            problem: PDE problem definition
            config: Solver configuration
        """
        self.problem = problem
        self.config = config

        # Solution storage
        self.field_current = None
        self.field_new = None
        self.evaluator_current = None
        self.evaluator_new = None
        self.amr = None

        # History
        self.history = {
            'times': [],
            'num_gaussians': [],
            'errors': [],
            'losses': [],
        }

        self.current_time = 0.0
        self.step_count = 0

    def initialize_field(self, num_initial_gaussians: int = 50):
        """
        Initialize Gaussian field from initial condition

        Args:
            num_initial_gaussians: Number of initial Gaussians
        """
        print(f"Initializing field with {num_initial_gaussians} Gaussians...")

        # Create initial grid of Gaussians
        field = self._create_initial_field(num_initial_gaussians)

        # Fit initial condition
        self._fit_initial_condition(field)

        # Setup evaluator and AMR
        self.field_current = field
        self.evaluator_current = FastEvaluator(
            field, eps_rel=self.config.eps_rel,
            use_scipy=self.config.use_scipy_kdtree
        )
        self.evaluator_current.build_tree()

        self.amr = AMRRefiner(field, self.evaluator_current)

        print(f"Initialization complete. "
              f"Final Gaussians: {self.field_current.num_gaussians}")

    def _create_initial_field(self, num_gaussians: int) -> GaussianField:
        """Create initial Gaussian field on uniform grid"""
        dim = self.problem.dim
        domain_min = self.config.domain_min
        domain_max = self.config.domain_max

        # Create uniform grid
        points_per_dim = int(np.ceil(num_gaussians ** (1.0 / dim)))
        grid_1d = [np.linspace(domain_min[i], domain_max[i], points_per_dim)
                   for i in range(dim)]

        # Meshgrid
        grids = np.meshgrid(*grid_1d, indexing='ij')
        centers = np.stack([g.ravel() for g in grids], axis=1)

        # Create Gaussians
        field = GaussianField()

        # Initial covariance (fraction of domain size)
        domain_size = domain_max - domain_min
        initial_scale = np.min(domain_size) / (2 * points_per_dim)

        for center in centers:
            # Isotropic covariance
            sigma = (initial_scale ** 2) * np.eye(dim)

            primitive = GaussianPrimitive(
                weight=1.0 / len(centers),  # Uniform weights initially
                mu=center,
                sigma=sigma
            )

            field.add_primitive(primitive)

        return field

    def _fit_initial_condition(self, field: GaussianField,
                                num_iters: int = 200):
        """
        Fit Gaussian field to initial condition using optimization

        Args:
            field: GaussianField to optimize
            num_iters: Number of optimization iterations
        """
        print("Fitting initial condition...")

        # Sample points for fitting
        M = self.config.num_collocation_points * 5  # More points for IC
        X = self._sample_collocation_points(M)

        # Evaluate initial condition
        u_target = np.array([self.problem.initial_condition(x) for x in X])

        # Setup PyTorch optimization
        weights, mus, sigmas = field.get_parameters()

        # Convert to PyTorch tensors (requires_grad)
        weights_t = torch.tensor(weights, dtype=torch.float64, requires_grad=True)

        # Use simpler optimization: only optimize weights with fixed Gaussians
        optimizer = torch.optim.Adam([weights_t], lr=0.01)

        for iter in range(num_iters):
            optimizer.zero_grad()

            # Evaluate field at collocation points
            u_pred = self._evaluate_field_torch(X, weights_t, mus, sigmas)

            # MSE loss
            loss = torch.mean((u_pred - torch.tensor(u_target, dtype=torch.float64)) ** 2)

            loss.backward()
            optimizer.step()

            if iter % 50 == 0:
                print(f"  Iter {iter}: Loss = {loss.item():.6f}")

        # Update field with optimized weights
        field.set_parameters(weights_t.detach().numpy(), mus, sigmas)

        print(f"IC fitting complete. Final loss: {loss.item():.6f}")

    def _evaluate_field_torch(self, X: np.ndarray, weights: torch.Tensor,
                              mus: np.ndarray, sigmas: np.ndarray) -> torch.Tensor:
        """
        Evaluate Gaussian field in PyTorch (for autodiff)

        Simple version without cutoff for optimization

        Args:
            X: Points to evaluate, shape (M, d)
            weights: Weights, shape (N,)
            mus: Centers, shape (N, d)
            sigmas: Covariances, shape (N, d, d)

        Returns:
            Values, shape (M,)
        """
        M = len(X)
        N = len(weights)
        dim = X.shape[1]

        # Convert to torch
        X_t = torch.tensor(X, dtype=torch.float64)
        mus_t = torch.tensor(mus, dtype=torch.float64)
        sigmas_t = torch.tensor(sigmas, dtype=torch.float64)

        # Compute all Gaussian values
        # Shape: (M, N)
        values = torch.zeros(M, dtype=torch.float64)

        for i in range(N):
            # diff: (M, d)
            diff = X_t - mus_t[i]

            # Mahalanobis distance squared
            sigma_inv = torch.inverse(sigmas_t[i])
            # (M, d) @ (d, d) @ (d, M) -> (M,)
            mahal_sq = torch.sum(diff @ sigma_inv * diff, dim=1)

            # Gaussian value
            det_sigma = torch.det(sigmas_t[i])
            normalizer = 1.0 / torch.sqrt((2 * np.pi) ** dim * det_sigma)
            gaussian = normalizer * torch.exp(-0.5 * mahal_sq)

            values += weights[i] * gaussian

        return values

    def _sample_collocation_points(self, M: int,
                                    include_boundary: bool = True) -> np.ndarray:
        """
        Sample collocation points in domain

        Args:
            M: Number of points
            include_boundary: If True, include some boundary points

        Returns:
            Points, shape (M, d)
        """
        dim = self.problem.dim
        domain_min = self.config.domain_min
        domain_max = self.config.domain_max

        if include_boundary:
            # 80% interior, 20% boundary
            M_interior = int(0.8 * M)
            M_boundary = M - M_interior

            # Interior points (uniform random)
            X_interior = np.random.uniform(
                domain_min, domain_max, size=(M_interior, dim)
            )

            # Boundary points
            X_boundary = self._sample_boundary_points(M_boundary)

            X = np.vstack([X_interior, X_boundary])
        else:
            X = np.random.uniform(domain_min, domain_max, size=(M, dim))

        return X

    def _sample_boundary_points(self, M: int) -> np.ndarray:
        """Sample points on domain boundary"""
        dim = self.problem.dim
        domain_min = self.config.domain_min
        domain_max = self.config.domain_max

        X_boundary = []

        for _ in range(M):
            # Choose random boundary face
            face_dim = np.random.randint(0, dim)
            face_side = np.random.choice([0, 1])  # 0=min, 1=max

            # Random point on that face
            x = np.random.uniform(domain_min, domain_max)

            if face_side == 0:
                x[face_dim] = domain_min[face_dim]
            else:
                x[face_dim] = domain_max[face_dim]

            X_boundary.append(x)

        return np.array(X_boundary)

    def solve_step(self) -> Tuple[float, bool]:
        """
        Solve one time step with adaptive refinement

        Implements SOLVE_ADAPT_Accelerated algorithm

        Returns:
            error: Global error estimate
            converged: True if converged
        """
        print(f"\n{'='*60}")
        print(f"Time step {self.step_count}: t = {self.current_time:.4f}")
        print(f"{'='*60}")

        dt = self.config.dt
        t_new = self.current_time + dt

        # Initialize new field as copy of current
        self.field_new = copy.deepcopy(self.field_current)
        self.evaluator_new = FastEvaluator(
            self.field_new, eps_rel=self.config.eps_rel,
            use_scipy=self.config.use_scipy_kdtree
        )

        # Adaptive loop
        for adapt_iter in range(self.config.max_adapt_iters):
            print(f"\n--- Adapt iteration {adapt_iter} ---")
            print(f"Num Gaussians: {self.field_new.num_gaussians}")

            # (a) SOLVE: Optimize parameters to satisfy PDE
            loss = self._solve_optimization(dt, t_new)

            print(f"Optimization loss: {loss:.6e}")

            # (b) ESTIMATE: Compute error indicators
            self.amr = AMRRefiner(self.field_new, self.evaluator_new)
            local_errors = self.amr.estimate_errors(
                metric=self.config.error_metric
            )
            global_error = self.amr.compute_global_error(local_errors)

            print(f"Global error: {global_error:.6e} (tol: {self.config.tol_step:.6e})")

            # (c) CONVERGE: Check if error acceptable
            if global_error < self.config.tol_step:
                print("✓ Converged!")
                self._finalize_step(loss, global_error)
                return global_error, True

            # (d) MARK: Identify elements to refine/coarsen
            refine_set, coarsen_set = self.amr.dorfler_marking(
                local_errors,
                theta_refine=self.config.theta_refine,
                theta_coarsen=self.config.theta_coarsen
            )

            print(f"Marked: {len(refine_set)} refine, {len(coarsen_set)} coarsen")

            # (e) REFINE/COARSEN
            if len(refine_set) > 0:
                self.amr.refine(refine_set, num_children=2)

            if len(coarsen_set) > 0 and self.field_new.num_gaussians > self.config.min_gaussians:
                self.amr.coarsen(coarsen_set)

            # Check max Gaussians
            if self.field_new.num_gaussians > self.config.max_gaussians:
                print(f"Warning: Exceeded max Gaussians ({self.config.max_gaussians})")
                break

        print("\n⚠ Did not converge within max adapt iterations")
        self._finalize_step(loss, global_error)
        return global_error, False

    def _solve_optimization(self, dt: float, t_new: float) -> float:
        """
        Optimize field parameters to minimize PDE residual

        Args:
            dt: Time step
            t_new: New time level

        Returns:
            Final loss value
        """
        # Build/rebuild KD-tree
        self.evaluator_new.build_tree()
        self.evaluator_current.build_tree()

        # Sample collocation points
        M = self.config.num_collocation_points
        X = self._sample_collocation_points(M)

        # Get current parameters
        weights, mus, sigmas = self.field_new.get_parameters()

        # Convert to torch tensors
        weights_t = torch.tensor(weights, dtype=torch.float64, requires_grad=True)
        mus_t = torch.tensor(mus, dtype=torch.float64, requires_grad=True)

        # Don't optimize sigmas for simplicity (can be added)
        params = [weights_t, mus_t]

        # Setup optimizer
        if self.config.optimizer == 'adam':
            optimizer = torch.optim.Adam(params, lr=self.config.learning_rate)
        elif self.config.optimizer == 'lbfgs':
            optimizer = torch.optim.LBFGS(params, lr=self.config.learning_rate,
                                          max_iter=20)
        else:
            optimizer = torch.optim.SGD(params, lr=self.config.learning_rate)

        # Optimization loop
        for iter in range(self.config.num_solve_iters):
            def closure():
                optimizer.zero_grad()

                # Update field with current parameters
                self.field_new.set_parameters(
                    weights_t.detach().numpy(),
                    mus_t.detach().numpy(),
                    sigmas
                )
                self.evaluator_new.invalidate_tree()
                self.evaluator_new.build_tree()

                # Compute residuals at collocation points
                residuals = []

                for x in X:
                    # Evaluate new and old solutions
                    u_new = self.evaluator_new.evaluate(
                        x, use_acceleration=self.config.use_acceleration
                    )
                    u_old = self.evaluator_current.evaluate(
                        x, use_acceleration=self.config.use_acceleration
                    )

                    grad_u_new = self.evaluator_new.gradient(
                        x, use_acceleration=self.config.use_acceleration
                    )
                    grad_u_old = self.evaluator_current.gradient(
                        x, use_acceleration=self.config.use_acceleration
                    )

                    lapl_u_new = self.evaluator_new.laplacian(
                        x, use_acceleration=self.config.use_acceleration
                    )
                    lapl_u_old = self.evaluator_current.laplacian(
                        x, use_acceleration=self.config.use_acceleration
                    )

                    # Compute residual
                    if self.problem.is_on_boundary(x):
                        # Boundary condition
                        u_bc = self.problem.boundary_condition(x, t_new)
                        r = u_new - u_bc
                    else:
                        # PDE residual
                        r = self.problem.residual(
                            u_new, u_old,
                            grad_u_new, grad_u_old,
                            lapl_u_new, lapl_u_old,
                            dt, x, t_new
                        )

                    residuals.append(r ** 2)

                # MSE loss
                loss = np.mean(residuals)
                loss_t = torch.tensor(loss, dtype=torch.float64, requires_grad=True)

                return loss_t

            loss_t = closure()

            if iter % 10 == 0:
                print(f"  Optim iter {iter}: loss = {loss_t.item():.6e}")

            # For LBFGS
            if self.config.optimizer == 'lbfgs':
                optimizer.step(closure)
            else:
                loss_t.backward()
                optimizer.step()

        # Final update
        self.field_new.set_parameters(
            weights_t.detach().numpy(),
            mus_t.detach().numpy(),
            sigmas
        )
        self.evaluator_new.invalidate_tree()

        return loss_t.item()

    def _finalize_step(self, loss: float, error: float):
        """Finalize time step and update history"""
        # Update current solution
        self.field_current = copy.deepcopy(self.field_new)
        self.evaluator_current = FastEvaluator(
            self.field_current, eps_rel=self.config.eps_rel,
            use_scipy=self.config.use_scipy_kdtree
        )

        # Update time
        self.current_time += self.config.dt
        self.step_count += 1

        # Record history
        self.history['times'].append(self.current_time)
        self.history['num_gaussians'].append(self.field_current.num_gaussians)
        self.history['errors'].append(error)
        self.history['losses'].append(loss)

    def solve(self):
        """Solve full time-dependent problem"""
        print("\n" + "="*60)
        print("Starting AR-AMR-PIGF Solver")
        print("="*60)
        print(f"Time: [0, {self.config.t_final}], dt = {self.config.dt}")
        print(f"Domain: {self.config.domain_min} to {self.config.domain_max}")
        print(f"Collocation points: {self.config.num_collocation_points}")
        print(f"Acceleration: {self.config.use_acceleration}")
        print("="*60)

        # Solve time steps
        num_steps = int(np.ceil(self.config.t_final / self.config.dt))

        for step in range(num_steps):
            error, converged = self.solve_step()

            if not converged:
                print("⚠ Warning: Step did not converge fully")

        print("\n" + "="*60)
        print("Solve complete!")
        print("="*60)

        self._print_summary()

    def _print_summary(self):
        """Print solution summary"""
        print("\nSolution Summary:")
        print(f"  Total time steps: {self.step_count}")
        print(f"  Final time: {self.current_time:.4f}")
        print(f"  Final Gaussians: {self.field_current.num_gaussians}")
        print(f"  Final error: {self.history['errors'][-1]:.6e}")
        print(f"  Average Gaussians: {np.mean(self.history['num_gaussians']):.1f}")

        # Evaluator stats
        stats = self.evaluator_current.get_statistics()
        print(f"\nAcceleration Statistics:")
        print(f"  Average active Gaussians: {stats['avg_active_gaussians']:.1f}")
        print(f"  Estimated speedup: {stats['speedup_estimate']:.2f}x")
