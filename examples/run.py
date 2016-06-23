from crowd_dynamics.simulation import Simulation
from crowd_dynamics.gui.qt import gui
from examples.simulations.hallway import initialize

# List of thing to implement
# TODO: check continuity -> numpy.ascontiguousarray
# TODO: Egress flow magnitude
# TODO: Measure crowd densities
# TODO: Should not see trough walls


path = "/home/jaan/Dropbox/Projects/Crowd-Dynamics-Simulations/results"

if __name__ == '__main__':
    simulation = Simulation(*initialize(), dirpath=path)
    gui(simulation)
