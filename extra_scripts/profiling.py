from line_profiler import LineProfiler

from crowddynamics.examples.simulations import FourExits
from crowddynamics.logging import setup_logging
from crowddynamics.simulation.agents import ThreeCircle, Circular


def main(simulation, iterations: int, **kwargs):
    hallway = simulation(**kwargs)
    hallway.update()
    for i in range(iterations - 1):
        hallway.update()


if __name__ == '__main__':
    # TODO: memory profiler
    # TODO: https://nvbn.github.io/2017/05/29/complexity/

    setup_logging()
    profiler = LineProfiler(main)
    profiler.runcall(main, simulation=FourExits, iterations=100,
                     size=100,
                     agent_type=Circular)
    profiler.print_stats()
