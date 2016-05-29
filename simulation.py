from simulations.hallway.config import dirpath, name, initialize
from source.io.path import SimulationPath
from source.system import System

if __name__ == '__main__':
    path = SimulationPath(dirpath)
    anim_path = path.animation("{}.mp4".format(name))
    result_path = path.result("{}.csv".format(name))
    results = None

    for num in range(1):
        simulation = System(*initialize())
        dims = (-5, 55), (-30, 30)
        simulation.animation(*dims, save=True, frames=2000, filepath=anim_path)

        # a = simulation.result.agents_in_goal_times
        # a = a.reshape((1, a.size))
        # df = pd.DataFrame(a)
        #
        # if results is None:
        #     results = df
        # else:
        #     results = results.append(df, ignore_index=True)
        #
        # results.to_csv(result_path)

        print('{:03d} '.format(num), 76*'=')
