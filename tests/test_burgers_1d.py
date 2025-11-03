"""
1D Burgers Equation Test Case

PDE: ∂u/∂t + u ∂u/∂x = ν ∂²u/∂x²
Domain: x ∈ [0, 1], t ∈ [0, T]
IC: u(x, 0) = sin(2πx)
BC: Periodic boundaries
"""

import sys
sys.path.append('../src')

import numpy as np
import matplotlib.pyplot as plt
from ar_amr_pigf.solver import PDEProblem, ARSolver, SolverConfig


class Burgers1D(PDEProblem):
    """1D Burgers equation with periodic boundaries"""

    def __init__(self, config: SolverConfig, nu: float = 0.01):
        """
        Initialize Burgers equation

        Args:
            config: Solver configuration
            nu: Viscosity coefficient
        """
        super().__init__(config)
        self.nu = nu

    def initial_condition(self, x: np.ndarray) -> float:
        """IC: u(x, 0) = sin(2πx)"""
        return np.sin(2 * np.pi * x[0])

    def boundary_condition(self, x: np.ndarray, t: float) -> float:
        """Periodic boundaries (enforced implicitly)"""
        return 0.0

    def residual(self, u_new: float, u_old: float,
                 grad_u_new: np.ndarray, grad_u_old: np.ndarray,
                 lapl_u_new: float, lapl_u_old: float,
                 dt: float, x: np.ndarray, t: float) -> float:
        """
        Burgers residual (Crank-Nicolson):

        R = (u_new - u_old)/dt + 0.5*(u_new·∂u_new/∂x + u_old·∂u_old/∂x)
            - ν·0.5*(∂²u_new/∂x² + ∂²u_old/∂x²)
        """
        # Time derivative term
        time_term = (u_new - u_old) / dt

        # Nonlinear advection term (Crank-Nicolson averaging)
        advection_new = u_new * grad_u_new[0]
        advection_old = u_old * grad_u_old[0]
        advection_term = 0.5 * (advection_new + advection_old)

        # Diffusion term
        diffusion_term = -self.nu * 0.5 * (lapl_u_new + lapl_u_old)

        # Total residual
        residual = time_term + advection_term + diffusion_term

        return residual

    def is_on_boundary(self, x: np.ndarray, tol: float = 1e-6) -> bool:
        """Periodic boundaries - no hard BCs"""
        return False


def run_burgers_test():
    """Run 1D Burgers equation test"""

    print("\n" + "="*70)
    print("1D BURGERS EQUATION TEST")
    print("="*70)

    # Configuration
    config = SolverConfig(
        # Time stepping
        dt=0.01,
        t_final=0.5,

        # Domain
        domain_min=np.array([0.0]),
        domain_max=np.array([1.0]),

        # Optimization
        num_collocation_points=200,
        num_solve_iters=30,
        learning_rate=0.005,
        optimizer='adam',

        # AMR
        max_adapt_iters=3,
        tol_step=5e-3,
        theta_refine=0.6,
        theta_coarsen=0.05,
        error_metric='gradient',

        # Acceleration
        eps_rel=1e-6,
        use_acceleration=True,
        use_scipy_kdtree=True,

        # Constraints
        max_gaussians=500,
        min_gaussians=10,
    )

    # Create problem
    problem = Burgers1D(config, nu=0.01)

    # Create solver
    solver = ARSolver(problem, config)

    # Initialize
    solver.initialize_field(num_initial_gaussians=30)

    # Solve
    solver.solve()

    # Visualize results
    visualize_solution(solver)

    return solver


def visualize_solution(solver: ARSolver):
    """Visualize solution and statistics"""

    print("\nGenerating visualizations...")

    fig = plt.figure(figsize=(15, 10))

    # 1. Solution at different times
    ax1 = plt.subplot(2, 3, 1)
    times_to_plot = [0.0, 0.1, 0.2, 0.3, 0.5]
    x_eval = np.linspace(0, 1, 200)

    # We only have final solution, so plot that
    u_vals = []
    for x in x_eval:
        u = solver.evaluator_current.evaluate(
            np.array([x]), use_acceleration=True
        )
        u_vals.append(u)

    ax1.plot(x_eval, u_vals, 'b-', linewidth=2, label=f't={solver.current_time:.2f}')
    ax1.set_xlabel('x')
    ax1.set_ylabel('u(x,t)')
    ax1.set_title('1D Burgers Solution')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Number of Gaussians over time
    ax2 = plt.subplot(2, 3, 2)
    ax2.plot(solver.history['times'], solver.history['num_gaussians'],
             'bo-', linewidth=2, markersize=6)
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Number of Gaussians')
    ax2.set_title('Adaptive Refinement')
    ax2.grid(True, alpha=0.3)

    # 3. Error over time
    ax3 = plt.subplot(2, 3, 3)
    ax3.semilogy(solver.history['times'], solver.history['errors'],
                 'ro-', linewidth=2, markersize=6)
    ax3.axhline(y=solver.config.tol_step, color='k', linestyle='--',
                label='Tolerance')
    ax3.set_xlabel('Time')
    ax3.set_ylabel('Error')
    ax3.set_title('Error Evolution')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. Gaussian centers (final time)
    ax4 = plt.subplot(2, 3, 4)
    weights, mus, sigmas = solver.field_current.get_parameters()

    # Plot Gaussian centers with sizes proportional to weights
    abs_weights = np.abs(weights)
    sizes = 100 * abs_weights / np.max(abs_weights)
    colors = ['red' if w < 0 else 'blue' for w in weights]

    ax4.scatter(mus[:, 0], np.zeros_like(mus[:, 0]),
                s=sizes, c=colors, alpha=0.6)
    ax4.set_xlabel('x')
    ax4.set_title('Gaussian Centers (final time)')
    ax4.set_ylim(-0.5, 0.5)
    ax4.grid(True, alpha=0.3)

    # 5. Loss over time
    ax5 = plt.subplot(2, 3, 5)
    ax5.semilogy(solver.history['times'], solver.history['losses'],
                 'go-', linewidth=2, markersize=6)
    ax5.set_xlabel('Time')
    ax5.set_ylabel('Loss')
    ax5.set_title('Optimization Loss')
    ax5.grid(True, alpha=0.3)

    # 6. Statistics table
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

    Avg active Gaussians: {stats['avg_active_gaussians']:.1f}
    Speedup estimate: {stats['speedup_estimate']:.1f}x

    Total evaluations: {stats['num_evaluations']}
    Build time: {stats['build_time']:.3f}s
    Query time: {stats['query_time']:.3f}s
    Eval time: {stats['eval_time']:.3f}s
    """

    ax6.text(0.1, 0.5, stats_text, fontfamily='monospace',
             fontsize=9, verticalalignment='center')

    plt.tight_layout()
    plt.savefig('../results/burgers_1d_results.png', dpi=150, bbox_inches='tight')
    print("Saved results to ../results/burgers_1d_results.png")

    plt.show()


if __name__ == '__main__':
    # Create results directory
    import os
    os.makedirs('../results', exist_ok=True)

    # Run test
    solver = run_burgers_test()

    print("\n" + "="*70)
    print("TEST COMPLETE!")
    print("="*70)
