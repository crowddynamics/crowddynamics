path = "/home/jaan/Dropbox/Projects/Crowd-Dynamics-Simulations/"
path2 = "/media/storage3/"


def outdoor(density):
    from crowd_dynamics.examples.outdoor import initialize
    # FIXME: nan array, forces
    kwargs = {}
    if density == "low":
        kwargs.update(size=30, width=10, height=10, path=path)
    elif density == "medium":
        kwargs.update(size=100, width=15, height=15, path=path)
    elif density == "high":
        kwargs.update(size=200, width=20, height=20, path=path)
    return initialize(**kwargs)


def hallway(density):
    from crowd_dynamics.examples.hallway import initialize
    kwargs = {}
    if density == "low":
        kwargs.update(size=30, path=path)
    elif density == "medium":
        kwargs.update(size=100, path=path)
    elif density == "high":
        kwargs.update(size=200, path=path)
    return initialize(**kwargs)


def evacuation(density):
    from crowd_dynamics.examples.evacuation import initialize
    kwargs = {}
    if density == "low":
        kwargs.update(size=30, width=5, height=10, path=path)
    elif density == "medium":
        kwargs.update(size=200, width=7, height=14, egress_model=True,
                      t_aset=45, path=path)
    elif density == "high":
        kwargs.update(size=700, width=15, height=30, egress_model=True,
                      t_aset=250, path=path, name="evacuation700")
    return initialize(**kwargs)


if __name__ == '__main__':
    from crowd_dynamics.qui import main
    # density = {"low", "medium", "high"}
    # spawn_shape = {"circ", "rect"}

    simulation = outdoor("medium")
    # simulation = hallway("medium")
    # simulation = evacuation("high")

    # FIXME: When updating position or angle -> Update_shoulder_positions()
    # TODO: Better agent initialization.
    main(simulation)
    # simulation.run(simu_time_limit=250)
