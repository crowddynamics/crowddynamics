from source.parameters import constant, linear_wall, agent, x_dims, y_dims
from source.system import System
from source.visualization import plots

if __name__ == '__main__':
    simulation = System(constant, agent, linear_wall)
    plots.plot_animation(simulation, agent, linear_wall, x_dims, y_dims)
