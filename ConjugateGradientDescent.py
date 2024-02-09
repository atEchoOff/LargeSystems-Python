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
solver = ConjugateGradientMethodSolver(sparse=True)

x, _, _ = solver.solve(system)
x = np.vstack(([1], x, [-1]))

domain = np.linspace(0, 1, len(x))
plt.plot(domain, u(domain), label="True u")
plt.plot(domain, x, label="Estimated u")
plt.legend()
plt.show()