from LargeSystems.densesystem import DenseSystem
import numpy as np
from LargeSystems.solution import Solution

class SimpleSolver:
    # Solve Ax = b using numpy linalg solve in a dense format
    def solve(self, system:DenseSystem):
        # Sovle the build system, return a solution vector
        x = np.linalg.solve(system.A, system.b)

        # Make x flat, and convert to list
        x = x.T.tolist()[0]
        return Solution(x, system.var_idxs)
    
class GradientMethodSolver:
    # Solve Ax = b using the gradient method with exact step size
    pass