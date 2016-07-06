path = "/home/jaan/Dropbox/Projects/Crowd-Dynamics-Simulations/"
path2 = "/media/storage3/"


def outdoor(density):
    from crowd_dynamics.examples.outdoor import initialize
    pass


def hallway(density):
    from crowd_dynamics.examples.hallway import initialize

    if density == "low":
        return initialize(size=30, path=path)
    elif density == "medium":
        return initialize(size=100, path=path)
    elif density == "high":
        return initialize(size=200, path=path)


def evacuation(density):
    from crowd_dynamics.examples.evacuation import initialize

    if density == "low":
        return initialize(size=30, width=5, height=10,
                          path=path)
    elif density == "medium":
        return initialize(size=200, width=7, height=14,
                          egress_model=True,  t_aset=45,
                          path=path)
    elif density == "high":
        return initialize(size=700, width=15, height=30,
                          egress_model=True, t_aset=250,
                          path=path2, name="evacuation700")


if __name__ == '__main__':
    # density = {"low", "medium", "high"}
    # spawn_shape = {"circ", "rect"}

    from crowd_dynamics.qui import main
    # simulation = evacuation("high")
    simulation = hallway("medium")

    # main(simulation)
    simulation.run(simu_time_limit=250)
