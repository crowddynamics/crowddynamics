Agent
=====

Properties
----------

.. csv-table:: Body types
    :header-rows: 1

    name,symbol,adult,male,female,child,eldery,explanation
    radius,r,0.255,0.27,0.24,0.21,0.25,Total radius of the agent
    dr,dr,0.035,0.02,0.02,0.015,0.02,Difference bound for total radius
    k_t,k_t,0.5882,0.5926,0.5833,0.5714,0.6,Ratio of total radius and radius torso
    k_s,k_s,0.3725,0.3704,0.375,0.3333,0.36,Ratio of total radius and radius shoulder
    k_ts,k_ts,0.6275,0.6296,0.625,0.6667,0.64,Ratio of total radius and distance from torso to shoulder
    v,v,1.25,1.35,1.15,0.9,0.8,Walking speed of agent
    dv,dv,0.3,0.2,0.2,0.3,0.3,Difference bound for walking speed
    mass,m,73.5,80,67,57,70,Mass of an agent
    mass_scale,dm,8,8,6.7,5.7,7,Standard deviation of mass of the agent


.. csv-table:: Agent class
    :header-rows: 1

    name,symbol,value,unit,source,explanation
    size,,,,,Number of agents (N)
    shape,,,,,"Shape for arrays (N, 2)"
    three_circles_flag,,,,,Boolean indicating if agent is modeled with three circle model
    orientable_flag,,,,,Boolean indicating if agent is orientable
    active,,,,,Boolean indicating if agent is active
    goal_reached,,,,,Boolean indicating if goal is reahed
    mass,m,,kg,fds+evac,Mass
    radius,r,,m,fds+evac,Radius
    r_t,r_t,,m,fds+evac,Radius of torso
    r_s,r_s,,m,fds+evac,Radius of shoulder
    r_ts,r_ts,,m,fds+evac,Distance from torso to shoulder
    position,x,,m,,Position
    velocity,v,,m / s,,Velocity
    target_velocity,v_0,5,m / s,,Target velocity
    target_direction,e,,,,Target direction
    force,f,,N,,Force
    force_adjust,f_adj,,N,,Adjusting force
    force_agent,f_agent,,N,,Agent to agent force
    force_wall,f_wall,,N,,Agent to wall force
    inertia_rot,I_rot,4,kg * m^2,fds+evac,Rotational moment for agent of weight 80 kg and radius 0.27
    angle,varphi,"Interval(-pi, pi)",rad,,Angle
    angular_velocity,omega,,rad / s,,Angular velocity
    target_angle,varphi_0,"Interval(-pi, pi)",rad,,Target angle
    target_angular_velocity,omega_0,4 * pi,rad / s,fds+evac,Target angular velocity
    torque,M,,N * m,,Torque
    position_ls,x_ls,,m,,Position of the left shoulder
    position_rs,x_rs,,m,,Position of the right shoulder
    front,x_front,,m,,Position of the front
    tau_adj,tau_adj,0.5,s,fds+evac,Characteristic time for agent adjusting its  movement
    tau_adj_rot,tau_adjrot,0.2,s,fds+evac,Characteristic time for agent adjusting its  rotational movement
    k,k,1.5,N,power law,Social force scaling constant
    tau_0,tau_0,3,s,power law,Interaction time horizon
    mu,mu,12000,kg / s^2,fds+evac,Compression counteraction constant
    kappa,kappa,40000,kg / (m * s),fds+evac,Sliding friction constant
    damping,c_d,500,N,fds+evac,Damping coefficient for contact force
    a,A,2000,N,helbing,Scaling coefficient for social force
    b,B,0.08,m,helbing,Coefficient for social force
    std_rand_force,xi / m,0.1,,fds+evac,Standard deviation for random force from truncated normal distribution
    std_rand_torque,eta / I_rot,0.1,,fds+evac,Standard deviation for random torque from truncated normal distribution
    f_soc_ij_max,,2000,N,,Truncation for social force with agent to agent interaction
    f_soc_iw_max,,2000,N,,Truncation for social force with agent to wall interaction
    sight_soc,,7,m,,Maximum distance for social force to effect
    sight_wall,,7,m,,Maximum distance for social force to effect
