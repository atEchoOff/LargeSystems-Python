from TMOCSolver import *
from LargeSystems.linear import *
import numpy as np
from matplotlib import pyplot as plt

def Δlᶠ(y):
    return 0

def Δᵧl(y, u, t):
    return 2 * y

def Δᵤl(y, u, t):
    return u

def f(y, u, t):
    return .5 * y + u

fᵧ = np.asmatrix(.5, dtype=np.float64)

fᵤ = np.asmatrix(1, dtype=np.float64)

t0 = 0
tf = 1
y0 = 1
ny = 1
nu = 1


solver = TMOCSolver(t0, tf, y0, ny, nu)\
         .with_Δlᶠ(Δlᶠ)\
         .with_Δᵧl(Δᵧl)\
         .with_Δᵤl(Δᵤl)\
         .with_f(f)\
         .with_fᵧ(fᵧ)\
         .with_fᵤ(fᵤ)

def evaluate_system(θ, K, plot=False):
    solution = solver.solve(θ, K)

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
    if plot:
        plt.plot(domain_y_u, y, label="y")
        plt.plot(domain_y_u, true_y(domain_y_u), label="y*")
        plt.legend(loc="best")
        plt.xlabel("t")
        plt.ylabel("y")
        plt.title("Plot of Estimated Solution y Against True y")
        plt.show()


        plt.plot(domain_y_u, u, label="u")
        plt.plot(domain_y_u, true_u(domain_y_u), label="u*")
        plt.legend(loc="best")
        plt.xlabel("t")
        plt.ylabel("y")
        plt.title("Plot of Estimated Solution u Against True u")
        plt.show()


        plt.plot(domain_λ, λ, label="λ")
        plt.plot(domain_λ, true_λ(domain_λ), label="λ*")
        plt.legend(loc="best")
        plt.xlabel("t")
        plt.ylabel("y")
        plt.title("Plot of Estimated Solution λ Against True λ")
        plt.show()

    max_diff_y = np.max(np.abs(y - true_y(domain_y_u)))
    max_diff_u = np.max(np.abs(u - true_u(domain_y_u)))
    max_diff_λ = np.max(np.abs(λ + true_λ(domain_λ))) # FIXME this is temporary until we know if we can flip lambda
    return max_diff_y, max_diff_u, max_diff_λ

evaluate_system(.5, 10, plot=True)

diff_y = []
diff_u = []
diff_λ = []
for K in [10, 20, 40, 80]:
    max_diff_y, max_diff_u, max_diff_λ = evaluate_system(.5, K, plot=False)
    diff_y.append(max_diff_y)
    diff_u.append(max_diff_u)
    diff_λ.append(max_diff_λ)

plt.plot([1/10, 1/20, 1/40, 1/80], diff_y, label="Max Absolute Error in y")
plt.legend(loc="best")
plt.xlabel("h")
plt.ylabel("y")
plt.title("The Effect of h on the Accuracy of y")
plt.show()

plt.plot([1/10, 1/20, 1/40, 1/80], diff_u, label="Max Absolute Error in u")
plt.legend(loc="best")
plt.xlabel("h")
plt.ylabel("y")
plt.title("The Effect of h on the Accuracy of u")
plt.show()

plt.plot([1/10, 1/20, 1/40, 1/80], diff_λ, label="Max Absolute Error in λ")
plt.legend(loc="best")
plt.xlabel("h")
plt.ylabel("y")
plt.title("The Effect of h on the Accuracy of λ")
plt.show()