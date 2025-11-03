"""
2D Heat/Diffusion Equation Test Case

PDE: ∂u/∂t = α Δu
Domain: (x,y) ∈ [0,1]², t ∈ [0, T]
IC: u(x,y,0) = exp(-((x-0.5)²+(y-0.5)²)/0.05)
BC: Dirichlet u=0 on boundary
"""

import sys
sys.path.append('../src')

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from ar_amr_pigf.solver import PDEProblem, ARSolver, SolverConfig


class Diffusion2D(PDEProblem):
    """2D heat/diffusion equation"""

    def __init__(self, config: SolverConfig, alpha: float = 0.1):
        """
        Initialize diffusion equation

        Args:
            config: Solver configuration
            alpha: Diffusivity coefficient
        """
        super().__init__(config)
        self.alpha = alpha

    def initial_condition(self, x: np.ndarray) -> float:
        """IC: Gaussian bump at center"""
        x0, y0 = 0.5, 0.5
        sigma = 0.05
        r_sq = (x[0] - x0)**2 + (x[1] - y0)**2
        return np.exp(-r_sq / sigma)

    def boundary_condition(self, x: np.ndarray, t: float) -> float:
        """BC: u = 0 on boundary"""
        return 0.0

    def residual(self, u_new: float, u_old: float,
                 grad_u_new: np.ndarray, grad_u_old: np.ndarray,
                 lapl_u_new: float, lapl_u_old: float,
                 dt: float, x: np.ndarray, t: float) -> float:
        """
        Diffusion residual (Crank-Nicolson):

        R = (u_new - u_old)/dt - α·0.5*(Δu_new + Δu_old)
        """
        time_term = (u_new - u_old) / dt
        diffusion_term = -self.alpha * 0.5 * (lapl_u_new + lapl_u_old)

        return time_term + diffusion_term


def run_diffusion_test():
    """Run 2D diffusion equation test"""

    print("\n" + "="*70)
    print("2D DIFFUSION EQUATION TEST")
    print("="*70)

    # Configuration
    config = SolverConfig(
        # Time stepping
        dt=0.01,
        t_final=0.3,

        # Domain
        domain_min=np.array([0.0, 0.0]),
        domain_max=np.array([1.0, 1.0]),

        # Optimization
        num_collocation_points=300,
        num_solve_iters=30,
        learning_rate=0.01,
        optimizer='adam',

        # AMR
        max_adapt_iters=3,
        tol_step=1e-2,
        theta_refine=0.6,
        theta_coarsen=0.05,
        error_metric='gradient',

        # Acceleration
        eps_rel=1e-6,
        use_acceleration=True,
        use_scipy_kdtree=True,

        # Constraints
        max_gaussians=1000,
        min_gaussians=20,
    )

    # Create problem
    problem = Diffusion2D(config, alpha=0.1)

    # Create solver
    solver = ARSolver(problem, config)

    # Initialize
    solver.initialize_field(num_initial_gaussians=50)

    # Solve
    solver.solve()

    # Visualize results
    visualize_solution_2d(solver)

    return solver


def visualize_solution_2d(solver: ARSolver):
    """Visualize 2D solution and statistics"""

    print("\nGenerating visualizations...")

    fig = plt.figure(figsize=(18, 12))

    # Evaluation grid
    n_grid = 50
    x_eval = np.linspace(0, 1, n_grid)
    y_eval = np.linspace(0, 1, n_grid)
    X_grid, Y_grid = np.meshgrid(x_eval, y_eval)

    # Evaluate solution on grid
    U_grid = np.zeros((n_grid, n_grid))
    for i in range(n_grid):
        for j in range(n_grid):
            xy = np.array([X_grid[i, j], Y_grid[i, j]])
            U_grid[i, j] = solver.evaluator_current.evaluate(
                xy, use_acceleration=True
            )

    # 1. 3D Surface plot
    ax1 = plt.subplot(2, 3, 1, projection='3d')
    surf = ax1.plot_surface(X_grid, Y_grid, U_grid, cmap=cm.viridis,
                            linewidth=0, antialiased=True)
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_zlabel('u(x,y,t)')
    ax1.set_title(f'Solution at t={solver.current_time:.3f}')
    fig.colorbar(surf, ax=ax1, shrink=0.5)

    # 2. Contour plot
    ax2 = plt.subplot(2, 3, 2)
    contour = ax2.contourf(X_grid, Y_grid, U_grid, levels=20, cmap=cm.viridis)
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_title('Solution Contours')
    ax2.set_aspect('equal')
    fig.colorbar(contour, ax=ax2)

    # Overlay Gaussian centers
    weights, mus, sigmas = solver.field_current.get_parameters()
    ax2.scatter(mus[:, 0], mus[:, 1], c='red', s=20, alpha=0.5,
                edgecolors='black', linewidths=0.5, label='Gaussian centers')
    ax2.legend()

    # 3. Number of Gaussians over time
    ax3 = plt.subplot(2, 3, 3)
    ax3.plot(solver.history['times'], solver.history['num_gaussians'],
             'bo-', linewidth=2, markersize=6)
    ax3.set_xlabel('Time')
    ax3.set_ylabel('Number of Gaussians')
    ax3.set_title('Adaptive Refinement')
    ax3.grid(True, alpha=0.3)

    # 4. Error over time
    ax4 = plt.subplot(2, 3, 4)
    ax4.semilogy(solver.history['times'], solver.history['errors'],
                 'ro-', linewidth=2, markersize=6)
    ax4.axhline(y=solver.config.tol_step, color='k', linestyle='--',
                label='Tolerance')
    ax4.set_xlabel('Time')
    ax4.set_ylabel('Error')
    ax4.set_title('Error Evolution')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # 5. Gaussian centers distribution
    ax5 = plt.subplot(2, 3, 5)
    abs_weights = np.abs(weights)
    sizes = 100 * abs_weights / np.max(abs_weights)
    colors = ['red' if w < 0 else 'blue' for w in weights]

    ax5.scatter(mus[:, 0], mus[:, 1], s=sizes, c=colors, alpha=0.6,
                edgecolors='black', linewidths=0.5)
    ax5.set_xlabel('x')
    ax5.set_ylabel('y')
    ax5.set_title(f'Gaussian Centers (N={len(weights)})')
    ax5.set_aspect('equal')
    ax5.set_xlim(0, 1)
    ax5.set_ylim(0, 1)
    ax5.grid(True, alpha=0.3)

    # 6. Statistics
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('off')

    stats = solver.evaluator_current.get_statistics()

    stats_text = f"""
    SOLUTION STATISTICS
    {'='*40}

    Final time: {solver.current_time:.4f}
    Total steps: {solver.step_count}

    Final Gaussians: {solver.field_current.num_gaussians}
    Avg Gaussians: {np.mean(solver.history['num_gaussians']):.1f}

    Final error: {solver.history['errors'][-1]:.6e}
    Final loss: {solver.history['losses'][-1]:.6e}

    ACCELERATION STATS
    {'='*40}

    Avg active: {stats['avg_active_gaussians']:.1f}
    Speedup: {stats['speedup_estimate']:.1f}x

    Evaluations: {stats['num_evaluations']}
    Build time: {stats['build_time']:.3f}s
    Query time: {stats['query_time']:.3f}s
    """

    ax6.text(0.1, 0.5, stats_text, fontfamily='monospace',
             fontsize=9, verticalalignment='center')

    plt.tight_layout()
    plt.savefig('../results/diffusion_2d_results.png', dpi=150, bbox_inches='tight')
    print("Saved results to ../results/diffusion_2d_results.png")

    plt.show()


if __name__ == '__main__':
    # Create results directory
    import os
    os.makedirs('../results', exist_ok=True)

    # Run test
    solver = run_diffusion_test()

    print("\n" + "="*70)
    print("TEST COMPLETE!")
    print("="*70)
