from crowddynamics.simulation.base import LogicNodeBase, FieldBase, \
    AgentsBase, SimulationBase


def test_fieldbase():
    field = FieldBase()
    assert True


def test_agentbase():
    agents = AgentsBase()
    assert True


def test_logicnodebase():
    node = [LogicNodeBase(str(i)) for i in range(4)]
    tree = node[0] << node[1] << (
        node[2],
        node[3]
    )
    assert node[0].is_root
    assert node[0].children == (node[1], node[2], node[3])


def test_simulationbase():
    simu = SimulationBase()
    assert True
