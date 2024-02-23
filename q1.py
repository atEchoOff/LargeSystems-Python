import numpy as np
import inspect
from scipy.sparse.linalg import cg
import matplotlib.pyplot as plt
from matrices import MatrixWithBoundaryConditions, Tridiagonal
    
f = lambda x: np.pi ** 2 * np.cos(np.pi * x) # f(x) = π²cos(πx)
y = lambda x: np.cos(np.pi * x) # true solution, y(x)=cos(πx)

def run_approximation_1(N):
    # Run the finite difference approximation given N
    # Here we use the first scheme, with initial iterate (1,0,...,0,-1)
    # And right hand side (1, b, -1)

    # Initialize our constants
    h = 1 / (N + 1)
    A = MatrixWithBoundaryConditions(N)
    b = np.asmatrix(np.hstack(([1], h ** 2 * f(h * np.arange(1, N + 1)), [-1]))).T
    y0 = (np.eye(1, N + 2, 0) - np.eye(1, N + 2, N + 1)).T

    # Store residuals as we run
    res_norms = [np.linalg.norm(b - A * y0)] # scipy forgets the first residual
    def store_residuals(_):
        frame = inspect.currentframe().f_back
        res_norms.append(np.linalg.norm(frame.f_locals['r']))
    
    # Go! (Until tolerance of 1e-13 is reached)
    return cg(A.aslinop(), b, x0=y0, rtol=1e-13, callback=store_residuals)[0], res_norms

def run_approximation_2(N):
    # Run the finite difference approximation given N
    # Here we use the second scheme, with initial iterate 0
    # and with right hand side b

    # Initialize our constants
    h = 1 / (N + 1)
    A = Tridiagonal(N, -1, 2, -1)
    b = np.asmatrix((h **2 * f(h * np.arange(1, N + 1)) + np.eye(1, N, 0) - np.eye(1, N, N - 1))).T
    y0 = None # Shorthand for 0s matrix

    # Store residuals as we run
    res_norms = [np.linalg.norm(b)] # scipy forgets the first residual
    def store_residuals(_):
        frame = inspect.currentframe().f_back
        res_norms.append(np.linalg.norm(frame.f_locals['r']))
    
    # Go! (Until tolerance of 1e-13 is reached)
    return np.hstack(([1], cg(A.aslinop(), b, x0=y0, rtol=1e-13, callback=store_residuals)[0], [-1])), res_norms

# Get our solutions and norms
y1, res_norms1 = run_approximation_1(30)
y2, res_norms2 = run_approximation_2(30)
domain_1 = np.linspace(0, 1, len(y1))
domain_2 = np.linspace(0, 1, len(y2))

# Plot y1 data
plt.plot(domain_1, y(domain_1), label="True Solution y")
plt.scatter(domain_1, y1, label="Estimated Solution y", color="orange")
plt.xlabel("x")
plt.ylabel("y")
plt.title("The Solution of the First Finite Difference Scheme")
plt.legend()
plt.show()

# Plot y1 norms
plt.plot(res_norms1, label="Residual L2 Norms")
plt.xlabel("k")
plt.ylabel("Norm")
plt.legend()
plt.title("The Residual Norms at each Iteration k for the First Scheme")
plt.show()

# Plot y2 data
plt.plot(domain_2, y(domain_2), label="True Solution y")
plt.scatter(domain_2, y2, label="Estimated Solution y", color="orange")
plt.xlabel("x")
plt.ylabel("y")
plt.title("The Solution of the Second Finite Difference Scheme")
plt.legend()
plt.show()

# Plot y2 norms
plt.plot(res_norms2, label="Residual L2 Norms")
plt.xlabel("k")
plt.ylabel("Norm")
plt.legend()
plt.title("The Residual Norms at each Iteration k for the Second Scheme")
plt.show()