from TMOCSolver import *
from LargeSystems.linear import *
import numpy as np
from matplotlib import pyplot as plt

def Δlᶠ(y):
    return ZERO

def Δᵧl(y, u, t):
    return 2 * y

def Δᵤl(y, u, t):
    return u

def f(y, u, t):
    return .5 * y + u

fᵧ = np.asmatrix(.5, dtype=np.float64)

fᵤ = np.asmatrix(1, dtype=np.float64)

t0 = 0
y0 = 1
h = .01
K = 100
ny = 1
nu = 1


solver = TMOCSolver(t0, y0, h, K, ny, nu)\
         .with_Δlᶠ(Δlᶠ)\
         .with_Δᵧl(Δᵧl)\
         .with_Δᵤl(Δᵤl)\
         .with_f(f)\
         .with_fᵧ(fᵧ)\
         .with_fᵤ(fᵤ)

solution = solver.solve(.5)

y = [f"y{i}1" for i in range(0, K + 1)]
u = [f"u{i}1" for i in range(0, K + 1)]
λ = [f"λ{i}1" for i in range(1, K + 1)]

y = solution[y]
u = solution[u]
λ = solution[λ]

def true_y(t):
    return (2 * np.exp(3 * t) + np.exp(3)) / (np.exp(3 * t/2) * (2 + np.exp(3)))

def true_u(t):
    return 2 * (np.exp(3 * t) - np.exp(3)) / (np.exp(3 * t / 2) * (2 + np.exp(3)))

def true_λ(t):
    return -2 * (np.exp(3 * t) - np.exp(3)) / (np.exp(3 * t / 2) * (2 + np.exp(3)))

domain_y_u = np.linspace(0, 1, len(y))
domain_λ = np.linspace(0, 1, len(λ))

plt.plot(domain_y_u, y, label="y")
plt.plot(domain_y_u, true_y(domain_y_u), label="y*")
plt.legend(loc="best")
plt.show()


plt.plot(domain_y_u, u, label="u")
plt.plot(domain_y_u, true_u(domain_y_u), label="u*")
plt.legend(loc="best")
plt.show()


plt.plot(domain_λ, λ, label="λ")
plt.plot(domain_λ, true_λ(domain_λ), label="λ*")
plt.legend(loc="best")
plt.show()