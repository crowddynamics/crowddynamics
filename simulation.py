from simulations.bottleneck.config import constant, linear_wall, agent, goal
from source.system import System
from source.visualization import plots


if __name__ == '__main__':
    simulation = System(constant, agent, linear_wall, goal)
    # plots.plot_animation(simulation, (0, 55), (0, 50), save=True, frames=2300)
    plots.plot_animation(simulation, (0, 55), (0, 50))
    # for _ in simulation:
    #     pass
