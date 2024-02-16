from Utils.PDEs import build_2D_poisson
from LargeSystems.solvers import JacobiMethodSolver, GaussSeidelSolver, SuccessiveOverrelaxationSolver, GradientMethodSolver, ConjugateGradientMethodSolver, SimpleSolver, Tolerance
import matplotlib.pyplot as plt
import numpy as np

# Random function for right hand side
f = lambda x, y: np.random.rand()

# First, create our matrix
system = build_2D_poisson(N=20, boundary=lambda x, y: 0, f=f, stencil=5)
A = system.A
b = system.b
N = b.shape[0]

# Jacobi method. Zero tolerance for error, and go for N // 2 iterations
jacobi = JacobiMethodSolver(tol_type=Tolerance.ZERO, maxit=N // 2, sparse=True)

# Gauss-Seidel solver. See above.
gauss_seidel = GaussSeidelSolver(tol_type=Tolerance.ZERO, maxit=N // 2, sparse=True)

# Successive Overrelaxation solver. See above, and set omega=1.9
sor = SuccessiveOverrelaxationSolver(tol_type=Tolerance.ZERO, maxit=N // 2, ω=1.7, sparse=True)

# Gradient method solver. Go for infinite iterations, until the relative residual is less than 1e-15.
gradient = GradientMethodSolver(tol_type=Tolerance.RELATIVE_RESIDUAL, tol=1e-15, maxit=float('inf'), sparse=True)

# Conjugate gradient method solver. See above.
conjugate_gradient = ConjugateGradientMethodSolver(tol_type=Tolerance.RELATIVE_RESIDUAL, tol=1e-15, maxit=float('inf'), sparse=True)

# Before continuing, find the true solution so we can get the relative errors
true_x = np.asmatrix(SimpleSolver(sparse=True).solve(system).values).T

def inner(x, y):
    # Numpy's inner product is broken, this fixes it
    return np.dot(x.T, y)[0,0]

def evaluate_solver(solver):
    # Evaluate a solver and return a list of relative residuals and relative errors (logged)

    norm_x = np.linalg.norm(true_x) # precalculate these terms
    norm_b = np.linalg.norm(b)

    # First, construct metadata object to store residuals and iterands
    # Then, solve system
    class Metadata:
        def __init__(self):
            self.relative_errors = []
            self.relative_residuals = []
            self.diff_cost = []

        def save(self, x, r, **kwargs):
            # Save results to our lists at each iteration
            self.relative_errors.append(np.linalg.norm(x - true_x) / norm_x)
            self.relative_residuals.append(np.linalg.norm(r) / norm_b)
            if isinstance(solver, ConjugateGradientMethodSolver) or isinstance(solver, GradientMethodSolver):
                # Also save the cost difference
                self.diff_cost.append(.5 * inner(x - true_x, A * (x - true_x)))

    metadata = Metadata()

    # Now we solve our system
    solver.solve(system, metadata=metadata)

    return np.log(metadata.relative_residuals), np.log(metadata.relative_errors), np.log(metadata.diff_cost)

# Get the errors for each method
jacobi_log_resids, jacobi_log_errors, _ = evaluate_solver(jacobi)
gs_log_resids, gs_log_errors, _ = evaluate_solver(gauss_seidel)
sor_log_resids, sor_log_errors, _ = evaluate_solver(sor)

# For the descent methods, note that our matrix A is *negative definite*. So, we negate A and b to
# solve the same system, and make -A *positive definite*. 
system.A = -system.A
A = -A
system.b = -system.b
b = -b
grad_log_resids, grad_log_errors, grad_log_diff_costs = evaluate_solver(gradient)
cg_log_resids, cg_log_errors, cg_log_diff_costs = evaluate_solver(conjugate_gradient)

# Now, plot a semilog for the residuals for each method
plt.plot(jacobi_log_resids, label="Jacobi") # FIXME plots have bad bounds
plt.plot(gs_log_resids, label="Gauss-Seidel")
plt.plot(sor_log_resids, label="SOR")
plt.plot(grad_log_resids, label="Steepest Descent")
plt.plot(cg_log_resids, label="CG")
plt.xlabel("Iteration k")
plt.ylabel("log(error)")
plt.title("The Relative Residuals over each Iteration")
plt.legend()
plt.show()

# Finally, plot semilog for relative errors
plt.plot(jacobi_log_errors, label="Jacobi") # FIXME formulas for GS and SOR are wrong
plt.plot(gs_log_errors, label="Gauss-Seidel")
plt.plot(sor_log_errors, label="SOR")
plt.plot(grad_log_errors, label="Steepest Descent")
plt.plot(cg_log_errors, label="CG")
plt.xlabel("Iteration k")
plt.ylabel("log(error)")
plt.title("The Relative Errors over each Iteration")
plt.legend()
plt.show()

# Now, we want to plot cost(x) - cost(x*) over each iteration for CG and Gradient Descent
# If f(x)=xT*A*x-bT*x, it can be shown that
# f(x) - f(x*) = .5 * (x - x*)T * A * (x - x*) FIXME different than heinkenschloss

plt.plot(grad_log_diff_costs, label="Steepest Descent")
plt.plot(cg_log_diff_costs, label="CG")
plt.xlabel("log(iteration k)")
plt.ylabel("log(cost)")
plt.title("The Effect of log(Iteration) on log(Cost)")
plt.legend()
plt.show()