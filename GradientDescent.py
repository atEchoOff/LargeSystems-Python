from LargeSystems.linear import V
from LargeSystems.system import DenseSystem
from LargeSystems.solvers import GradientMethodSolver

x, y, z = V("x"), V("y"), V("z")
system = DenseSystem(["x", "y", "z"])

system.add_constraint(2 * x + 3 * y - 4 * z == 1)
system.add_constraint(4 * x - 3 * y         == 2)
system.add_constraint(x + y + z == 1)

