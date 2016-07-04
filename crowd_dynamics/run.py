path = "/home/jaan/Dropbox/Projects/Crowd-Dynamics-Simulations/"


def hallway100():
    from crowd_dynamics.examples.hallway import initialize
    return initialize(size=100, path=path)


def hallway200():
    from crowd_dynamics.examples.hallway import initialize
    return initialize(size=200, path=path)


def evacuation30():
    """Low crowd density."""
    from crowd_dynamics.examples.evacuation import initialize
    return initialize(size=30, width=15, path=path, dt_max=0.01)


def evacuation100():
    """Medium crowd density."""
    from crowd_dynamics.examples.evacuation import initialize
    return initialize(size=100, width=10, height=10, path=path,
                      egress_model=True, t_aset=45)


def evacuation_high():
    """Medium crowd density."""
    from crowd_dynamics.examples.evacuation import initialize
    return initialize(size=700, width=15, height=30, path=path,
                      egress_model=True, t_aset=350)


def outdoor():
    from crowd_dynamics.examples.outdoor import initialize
    return initialize(path=path)


def crossing_flows():
    from crowd_dynamics.examples.crossing_flows import initialize
    return initialize()


if __name__ == '__main__':
    # simulation = hallway100()
    simulation = evacuation_high()
    # simulation = evacuation100()
    from crowd_dynamics.qui import main
    main(simulation)
