from LargeSystems.linear import V
from LargeSystems.solvers import SimpleSolver
from LargeSystems.system import System

x, y, z = V("x", "y", "z")
system = System(["x", "y", "z"])

system.add_constraint(2 * x + 3 * y - z == 2)
system.add_constraint(2 * x - z + y == 1)
system.add_constraint(x + y + z == 1)

solver = SimpleSolver()
print(solver.solve(system).values)