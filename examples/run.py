
from crowd_dynamics.simulation import Simulation
from crowd_dynamics.gui.qt import gui
from examples.simulations.hallway import initialize

if __name__ == '__main__':
    for num in range(1):
        print('Simulation: {:3d} '.format(num))
        print(80 * "=")
        simulation = Simulation(*initialize())
        gui(simulation)
        # simulation.run()
