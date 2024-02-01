from TMOCSolver import *
import numpy as np
from ShiftedList import ShiftedList
from matplotlib import pyplot as plt

A = np.matrix([
    [2, -1, 0],
    [-1, 2, -1],
    [0, -1, 2]
], dtype=np.float64)

B = np.matrix([
    [4, -1, 2],
    [-1, 5, 3],
    [2, 3, 6]
], dtype=np.float64)

Q = np.matrix([
    [7, 1, -1],
    [1, 8, 0],
    [-1, 0, 9]
], dtype=np.float64)

R = np.matrix([
    [10, 2, -1],
    [2, 12, 4],
    [-1, 4, 8]
], dtype=np.float64)

def Δlᶠ(y):
    return 0

def Δᵧl(y, u, t):
    return Q * y

def Δᵤl(y, u, t):
    return R * u

def f(y, u, t):
    return A * y + B * u

fᵧ = A

fᵤ = B

t0 = 0
tf = 1
y0 = np.asmatrix(np.ones((3, 1)))
ny = 3
nu = 3

solver = TMOCSolver(t0, tf, y0, ny, nu)\
        .with_Δlᶠ(Δlᶠ)\
        .with_Δᵧl(Δᵧl)\
        .with_Δᵤl(Δᵤl)\
        .with_f(f)\
        .with_fᵧ(fᵧ)\
        .with_fᵤ(fᵤ)

K = 100
solution = solver.solve(.5, K)

yT = ShiftedList(1, [["y" + str(j) + str(i) for j in range(0, K + 1)] for i in range(1, ny + 1)])

yT_sols = []
for dim in range(1, ny + 1):
    # Get the solution for dimension "dim" from y
    yT_sols.append(solution[yT[dim]])

yT_sols = ShiftedList(1, yT_sols)

# Now we can plot it if we want
domain = np.linspace(0, 1, K + 1)
for dim in range(1, ny + 1):
    plt.plot(domain, yT_sols[dim], label=f"Solution in dimension {dim}")

plt.legend(loc="best")
plt.xlabel("t")
plt.ylabel("y")
plt.title("All Dimensions of the Solution y")
plt.show()