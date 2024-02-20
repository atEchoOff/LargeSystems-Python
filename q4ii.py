from ill_conditioned_data_assim import solve as solve_ill_conditioned
from well_conditioned_data_assim import solve as solve_well_conditioned
from scipy.sparse.linalg import expm
from scipy.sparse import csc_matrix, identity
from matrices import compute_K, compute_H
import numpy as np
import matplotlib.pyplot as plt

# To get consistent results
np.random.seed(1)

# The constants are kept the same as specified on page 87
α = .01
β = 1
θ = .5
n = 100
m = 5
h = .01
Δt = .01
nt = 50
mt = 25
T = nt * Δt

# Initialize our weights
W = identity(mt * m, dtype=np.float64, format="csr")

# Compute the matrix K and H
K = compute_K(α, β, h, n)
H = compute_H(n, m)

domain = np.linspace(0, 1, n)

# Calculate the true solution at 0
y0 = lambda x: np.exp(-100 * (x - .5)**2)
y0 = np.asmatrix(y0(domain)).T

# Now, we take samples of our solution for z.
z = np.zeros((0, 1), dtype=np.float64)
csc_K = csc_matrix(K) # More efficient for expm
for i in range(nt // mt, nt + 1, nt // mt):
    sample_pt = i * Δt
    z = np.vstack((z, H * expm(csc_K * sample_pt) * y0))

# First lets solve the system for no noise
y0_est_no_noise, res_est_no_noise = solve_ill_conditioned(θ, n, m, Δt, nt, mt, W, K, H, z)

# Now lets add some noise and do it over again
z = z + np.multiply(np.random.normal(0, .01, size=z.shape), z)
y0_est_with_noise, res_est_with_noise = solve_ill_conditioned(θ, n, m, Δt, nt, mt, W, K, H, z)

# Plot noiseless solution from ill-conditioned system
plt.plot(domain, y0, label="True y0")
plt.plot(domain, y0_est_no_noise, label="Estimated y0")
plt.title("Estimating the Initial Condition Without Noise")
plt.xlabel("x")
plt.legend()
plt.show()

# Show its residuals
plt.plot(np.log(res_est_no_noise), label="Log(Residual Norms)")
plt.title("Residuals for Ill-Conditioned System without Noise")
plt.xlabel("k")
plt.legend()
plt.show()

# Now show the plot with noise
plt.plot(domain, y0, label="True y0")
plt.plot(domain, y0_est_with_noise, label="Estimated y0")
plt.title("Estimating the Initial Condition With Noise")
plt.xlabel("x")
plt.legend()
plt.show()

# Show its residuals
plt.plot(np.log(res_est_with_noise), label="Log(Residual Norms)")
plt.title("Residuals for Ill-Conditioned System with Noise")
plt.xlabel("k")
plt.legend()
plt.show()

# We can see that noise had a big hold on the solution
# We now run the same problem on the same noisy vector using 2.116
R = identity(n, dtype=np.float64, format="csr")
y0_est_with_noise_and_reg, res_est_with_noise_and_reg = solve_well_conditioned(θ, n, m, Δt, nt, mt, W, K, H, z, R, 1e-3)

# Plot the solution with noise and regularization
plt.plot(domain, y0, label="True y0")
plt.plot(domain, y0_est_with_noise_and_reg, label="Estimated y0")
plt.title("Estimating the Initial Condition With Noise and Regularization")
plt.xlabel("x")
plt.legend()
plt.show()

# Show its residuals
plt.plot(np.log(res_est_with_noise_and_reg), label="Log(Residual Norms)")
plt.title("Residuals for Well-Conditioned System with Noise")
plt.xlabel("k")
plt.legend()
plt.show()