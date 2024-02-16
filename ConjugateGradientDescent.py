from LargeSystems.solvers import ConjugateGradientMethodSolver
import numpy as np
from matplotlib import pyplot as plt
from Utils.PDEs import build_1D_poisson

def f(x):
    # Our function f(x)
    return np.pi ** 2 * np.cos(np.pi * x)

def u(x):
    # True solution u(x)
    return np.cos(np.pi * x)

system = build_1D_poisson(30, 1, -1, f)

class Metadata:
    def __init__(self):
        self.x_norms = []

    def save(self, x, **kwargs):
        # Save certain arguments. Here, we save just the norm of the iterands
        self.x_norms.append(np.linalg.norm(x))

metadata = Metadata()
solver = ConjugateGradientMethodSolver(sparse=True)

x = solver.solve(system, metadata)
x = np.vstack(([1], x.values, [-1]))

# Fun fact about the conjugate gradient method: If you start at x0=0, then the norms of the iterands are strictly increasing!
x_norms = []
for i, norm in enumerate(metadata.x_norms):
    print(f"||x_{i}|| \t = \t {norm}")

domain = np.linspace(0, 1, len(x))
plt.plot(domain, u(domain), label="True u")
plt.plot(domain, x, label="Estimated u")
plt.legend()
plt.show()