from TMOCSolver import *
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
    domain_y = np.linspace(0, 1, len(y))
    λ = [f"λ{i}1" for i in range(1, K + 1)]
    domain_λ = np.linspace(1 / K, 1, len(λ))

    # If theta was 0 or 1, we got rid of u0 or uK from the system
    if θ == 0:
        u = [f"u{i}1" for i in range(1, K + 1)]
        domain_u = np.linspace(1 / K, 1, len(u))
    elif θ == 1:
        u = [f"u{i}1" for i in range(0, K)]
        domain_u = np.linspace(0, 1 - 1 / K, len(u))
    else:
        u = [f"u{i}1" for i in range(0, K + 1)]
        domain_u = np.linspace(0, 1, len(u))

    y = solution[y]
    u = solution[u]
    λ = solution[λ]

    def true_y(t):
        return (2 * np.exp(3 * t) + np.exp(3)) / (np.exp(3 * t/2) * (2 + np.exp(3)))

    def true_u(t):
        return 2 * (np.exp(3 * t) - np.exp(3)) / (np.exp(3 * t / 2) * (2 + np.exp(3)))

    def true_λ(t):
        return 2 * (np.exp(3 * t) - np.exp(3)) / (np.exp(3 * t / 2) * (2 + np.exp(3)))

    if plot:
        plt.plot(domain_y, y, label="y")
        plt.plot(domain_y, true_y(domain_y), label="y*")
        plt.legend(loc="best")
        plt.xlabel("t")
        plt.ylabel("y")
        plt.title(f"Plot of Estimated Solution y Against True y for θ={θ}")
        plt.show()


        plt.plot(domain_u, u, label="u")
        plt.plot(domain_u, true_u(domain_u), label="u*")
        plt.legend(loc="best")
        plt.xlabel("t")
        plt.ylabel("y")
        plt.title(f"Plot of Estimated Solution u Against True u for θ={θ}")
        plt.show()


        plt.plot(domain_λ, λ, label="λ")
        plt.plot(domain_λ, true_λ(domain_λ), label="λ*")
        plt.legend(loc="best")
        plt.xlabel("t")
        plt.ylabel("y")
        plt.title(f"Plot of Estimated Solution λ Against True λ for θ={θ}")
        plt.show()

    max_diff_y = np.max(np.abs(y - true_y(domain_y)))
    max_diff_u = np.max(np.abs(u - true_u(domain_u)))
    max_diff_λ = np.max(np.abs(λ - true_λ(domain_λ)))
    return max_diff_y, max_diff_u, max_diff_λ

def evaluate_errors(errors):
    diff_y = []
    diff_u = []
    diff_λ = []
    for K in [10, 20, 40, 80]:
        max_diff_y, max_diff_u, max_diff_λ = errors(K)
        diff_y.append(max_diff_y)
        diff_u.append(max_diff_u)
        diff_λ.append(max_diff_λ)

    print("Function\tK=10\t\tK=20\t\tK=40\t\tK=80")
    print("y\t\t", "\t".join(map("{0:.6g}".format, diff_y)))
    print("u\t\t", "\t".join(map("{0:.6g}".format, diff_u)))
    print("λ\t\t", "\t".join(map("{0:.6g}".format, diff_λ)))

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

print(evaluate_system(0, 10, plot=True))
print(evaluate_system(.5, 10, plot=True))
print(evaluate_system(1, 10, plot=True))

evaluate_errors(lambda K: evaluate_system(0, K, plot=False))
evaluate_errors(lambda K: evaluate_system(.5, K, plot=False))
evaluate_errors(lambda K: evaluate_system(1, K, plot=False))