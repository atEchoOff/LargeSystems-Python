from TMOCSolver import *
from LargeSystems.linear import *
import numpy as np
from matplotlib import pyplot as plt

def grad_lf(y):
    return ZERO

def grad_y_l(y, u, t):
    return 2 * y

def grad_u_l(y, u, t):
    return 1 * u

def f(y, u, t):
    return .5 * y + 1 * u

jacobi_f_y = np.asmatrix(.5, dtype=np.float64)

jacobi_f_u = np.asmatrix(1, dtype=np.float64)

t0 = 0
y0 = 1
h = .01
K = 100
ny = 1
nu = 1


solver = TMOCSolver(t0, y0, h, K, ny, nu)\
         .with_grad_lf(grad_lf)\
         .with_grad_y_l(grad_y_l)\
         .with_grad_u_l(grad_u_l)\
         .with_f(f)\
         .with_jacobi_f_y(jacobi_f_y)\
         .with_jacobi_f_u(jacobi_f_u)

solution = solver.solve(.5)

y = [f"y{i}1" for i in range(0, K + 1)]
u = [f"u{i}1" for i in range(0, K + 1)]
_lambda = [f"lambda{i}1" for i in range(1, K + 1)]

y = solution[y]
u = solution[u]
_lambda = solution[_lambda]

def true_y(t):
    return (2 * np.exp(3 * t) + np.exp(3)) / (np.exp(3 * t/2) * (2 + np.exp(3)))

def true_u(t):
    return 2 * (np.exp(3 * t) - np.exp(3)) / (np.exp(3 * t / 2) * (2 + np.exp(3)))

def true_lambda(t):
    return -2 * (np.exp(3 * t) - np.exp(3)) / (np.exp(3 * t / 2) * (2 + np.exp(3)))

domain_y_u = np.linspace(0, 1, len(y))
domain_lambda = np.linspace(0, 1, len(_lambda))

plt.plot(domain_y_u, y, label="y")
plt.plot(domain_y_u, true_y(domain_y_u), label="true_y")
plt.legend(loc="best")
plt.show()


plt.plot(domain_y_u, y, label="u")
plt.plot(domain_y_u, true_u(domain_y_u), label="true_u")
plt.legend(loc="best")
plt.show()


plt.plot(domain_lambda, _lambda, label="lambda")
plt.plot(domain_lambda, true_lambda(domain_lambda), label="true_lambda")
plt.legend(loc="best")
plt.show()