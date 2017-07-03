import os
from crowddynamics.examples.collective_motion import FourExits
from crowddynamics.simulation.agents import ThreeCircle, Circular


if __name__ == '__main__':
    # os.environ.update(NUMBA_DISABLE_JIT='1')
    simulation = FourExits(agent_type=Circular)
    simulation.update()
    simulation.update()
    simulation.update()
    simulation.update()
    simulation.update()
    simulation.update()
