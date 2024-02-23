# (That rhymes)
from cgnr import CGNRSolver
import numpy as np

# Generate some random matrices
A = np.asmatrix(np.random.rand(100, 10), dtype=np.float64)
b = np.asmatrix(np.random.rand(100, 1), dtype=np.float64)

# Get true solution
true_x = np.linalg.lstsq(A, b, rcond=None)[0]

# Create multiplication functions
def A_func(v):
    return A * v

def At_func(w):
    return A.T * w

# Initialize our solver, and solve the system
solver = CGNRSolver()
estimated_x = solver.solve(A_func, At_func, b, 10)

print(f"The error between the true and estimated solutions is {np.linalg.norm(true_x - estimated_x)}")