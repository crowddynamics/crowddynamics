from simulations.one_exit_box.config import constant, linear_wall, agent, goal
from source.system import System
from source.visualization import plots


if __name__ == '__main__':
    simulation = System(constant, agent, linear_wall, goal)
    plots.plot_animation(simulation, (0, 55), (0, 50))
