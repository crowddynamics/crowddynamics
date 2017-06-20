from line_profiler import LineProfiler
import memory_profiler

from crowddynamics.examples.collective_motion import FourExits
from crowddynamics.logging import setup_logging
from crowddynamics.simulation.agents import ThreeCircle, Circular

# TODO: https://nvbn.github.io/2017/05/29/complexity/


def main(simulation, iterations: int, **kwargs):
    hallway = simulation(**kwargs)
    hallway.update()
    for i in range(iterations - 1):
        hallway.update()


if __name__ == '__main__':
    setup_logging()
    kw = dict(simulation=FourExits,
              iterations=100,
              size_active=10,
              size_herding=190,
              agent_type=Circular)

    profiler = LineProfiler(main)
    profiler.runcall(main, **kw)
    profiler.print_stats()

    # prof = memory_profiler.LineProfiler(backend='psutil')
    # prof(main)(**kw)
    # memory_profiler.show_results(prof, precision=1)
