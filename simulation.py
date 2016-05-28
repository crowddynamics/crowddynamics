import numba
import pandas as pd

from simulations.bottleneck.config import dirpath, name, initialize
from source.io.path import SimulationPath
from source.system import System


# if False:
#     from source.visualization import plots
#     plots.plot_animation(simulation, (0, 55), (0, 50),
#                          save=True, frames=2300)
#
# if False:
#     from source.visualization import plots
#     plots.plot_animation(simulation, (0, 55), (0, 50))


@numba.jit()
def run(iterator):
    for _ in iterator:
        pass


if __name__ == '__main__':
    path = SimulationPath(dirpath)
    result_path = path.result("{}.csv".format(name))
    results = None

    for num in range(1):
        simulation = System(*initialize())
        run(simulation)

        a = simulation.result.agents_in_goal_times
        a = a.reshape((1, a.size))
        df = pd.DataFrame(a)

        if results is None:
            results = df
        else:
            results = results.append(df, ignore_index=True)

        results.to_csv(result_path)
        print('{:03d} '.format(num), 76*'=')
