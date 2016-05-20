from source.visualization import plots
from source.system import system
from source.parameters import constant, linear_wall, agent, x_dims, y_dims


if __name__ == '__main__':
    # visualization.plot_field(agent, x_dims, y_dims, linear_wall)
    simulation = system(constant, agent, linear_wall)
    # visualization.consume(simulation)
    plots.plot_animation(simulation, agent, linear_wall, x_dims, y_dims)
