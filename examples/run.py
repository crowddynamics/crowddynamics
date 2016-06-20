
from crowd_dynamics.simulation import Simulation
from crowd_dynamics.gui.qt import gui
from examples.simulations.hallway import initialize

# List of thing to implement
# TODO: np.dot -> check performance and gil
# TODO: check continuity -> numpy.ascontiguousarray
# TODO: Tables of anthropometric data
# TODO: Egress flow magnitude
# TODO: Measure crowd densities
# TODO: Should not see trough walls


if __name__ == '__main__':
    for num in range(1):
        simulation = Simulation(*initialize())
        gui(simulation)
