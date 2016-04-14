import sys

from source import visualization

try:
    # Find "Source" module to perform import
    module_path = '/home/jaan/Dropbox/Projects/Crowd-Dynamics/'
    sys.path.append(module_path)
except:
    pass

from source.core.system import system


if __name__ == '__main__':
    simulation = system()
    simulation_gen = simulation()
    visualization.func_plot(*simulation_gen)
