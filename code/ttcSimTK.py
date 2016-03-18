# This code is relased for the purpose of review and timely desimination.
# All rights are retained by the authors and the University of Minnesota
# Authors: Ioannis Karamouzas, Brian Skinner, and Stephen J. Guy
# Contact: sjguy@cs.umn.edu


import random
import time
import tkinter as tk
from math import sin, cos, sqrt, exp

import numpy as np

# Environmental Specifciation
agents_num = 18  # number of agents
size = 4  # environment size in metters

# Agent Parametrs (play with these)
k = 1.5
m = 2.0
t0 = 3
radius = .2  # Collision radius
sight = 7  # Neighbor search range
max_force = 5  # Maximum force/acceleration

pixelsize = 600
framedelay = 30
draw_velocities = True

win = tk.Tk()
canvas = tk.Canvas(win, width=pixelsize, height=pixelsize, background="#444")
canvas.pack()

# Initalized variables
itterations = 0
center = []  # center of agent
velocity = []  # velocity
goal_velocity = []  # goal velocity
neighbor = []  # neighbor list
neighbor_distance = []  # neighbor distence list

quit = False
paused = False
step = False

circles = []
velocity_lines = []
goalv_lines = []


def init_sim():
    global radius

    print("")
    print("Simulation of Agents on a flat 2D torus.")
    print("Agents avoid collisions using prinicples based on the laws of "
          "anticipation seen in human pedestrians.")
    print("Agents are white circles, Red agent moves faster.")
    print("Green Arrow is Goal Velocity, Red Arrow is Current Velocity")
    print("SPACE to pause, 'S' to step frame-by-frame, "
          "'V' to turn the velocity display on/off.")
    print("")

    for i in range(agents_num):
        circles.append(canvas.create_oval(0, 0, radius, radius, fill="white"))
        velocity_lines.append(canvas.create_line(0, 0, 10, 10, fill="red"))
        goalv_lines.append(canvas.create_line(0, 0, 10, 10, fill="green"))

        center.append(np.zeros(2))
        velocity.append(np.zeros(2))
        goal_velocity.append(np.zeros(2))

        center[i][0] = random.uniform(0, size)
        center[i][1] = random.uniform(0, size)
        angle = random.uniform(0, 2 * 3.141592)

        velocity[i][0] = cos(angle)
        velocity[i][1] = sin(angle)
        goal_velocity[i] = 1.5 * np.copy(velocity[i])

        if i == 0:
            goal_velocity[i] *= 2
            canvas.itemconfig(circles[i], fill="#FAA")


def draw_world():
    global radius, size

    for i in range(agents_num):
        scale = pixelsize / size
        canvas.coords(circles[i], scale * (center[i][0] - radius),
                      scale * (center[i][1] - radius), scale * (center[i][0] + radius),
                      scale * (center[i][1] + radius))
        canvas.coords(velocity_lines[i], scale * center[i][0], scale * center[i][1],
                      scale * (center[i][0] + 1. * radius * velocity[i][0]),
                      scale * (center[i][1] + 1. * radius * velocity[i][1]))
        canvas.coords(goalv_lines[i], scale * center[i][0], scale * center[i][1],
                      scale * (center[i][0] + 1. * radius * goal_velocity[i][0]),
                      scale * (center[i][1] + 1. * radius * goal_velocity[i][1]))
        if draw_velocities:
            canvas.itemconfigure(velocity_lines[i], state="normal")
            canvas.itemconfigure(goalv_lines[i], state="normal")
        else:
            canvas.itemconfigure(velocity_lines[i], state="hidden")
            canvas.itemconfigure(goalv_lines[i], state="hidden")
        double = False
        newX = center[i][0]
        newY = center[i][1]
        if center[i][0] < radius:
            newX += size
            double = True
        if center[i][0] > size - radius:
            newX -= size
            double = True
        if center[i][1] < radius:
            newY += size
            double = True
        if center[i][1] > size - radius:
            newY -= size
            double = True
        if double:
            pass
            # canvas.coords(circles[i],scale*(newX-rad),scale*(newY-rad),scale*(newX+rad),scale*(newY+rad))

            # indicate velocity and goal velocities


def find_neighbors():
    global neighbor, neighbor_distance, center

    neighbor = []
    neighbor_distance = []
    for i in range(agents_num):
        neighbor.append([])
        neighbor_distance.append([])
        for j in range(agents_num):
            if i == j:
                continue
            d = center[i] - center[j]
            if d[0] > size / 2.:
                d[0] = size - d[0]
            if d[1] > size / 2.:
                d[1] = size - d[1]
            if d[0] < -size / 2.:
                d[0] += size
            if d[1] < -size / 2.:
                d[1] += size
            l2 = d.dot(d)
            s2 = sight ** 2
            if l2 < s2:
                neighbor[i].append(j)
                neighbor_distance[i].append(sqrt(l2))


def E(t):
    return (B / t ** m) * exp(-t / t0)


def rdiff(pa, pb, va, vb, ra, rb):
    p = pb - pa  # relative position
    return sqrt(p.dot(p))


def ttc(pa, pb, va, vb, ra, rb):
    maxt = 999

    p = pb - pa  # relative position

    if p[0] > size / 2.:
        p[0] -= size
    if p[1] > size / 2.:
        p[1] -= size
    if p[0] < -size / 2.:
        p[0] += size
    if p[1] < -size / 2.:
        p[1] += size

    rv = vb - va  # relative velocity

    a = rv.dot(rv)
    b = 2 * rv.dot(p)
    c = p.dot(p) - (ra + rb) ** 2

    det = b * b - 4 * a * c
    t1 = maxt
    t2 = maxt

    if det > 0:
        t1 = (-b + sqrt(det)) / (2 * a)
        t2 = (-b - sqrt(det)) / (2 * a)

    t = min(t1, t2)

    if t < 0 and max(t1, t2) > 0:  # we are colliding
        t = 100  # maybe should be 0?
    if t < 0:
        t = maxt
    if t > maxt:
        t = maxt

    # if t < 10: print(t)

    return t


def dE(pa, pb, va, vb, ra, rb):
    global k, m, t0
    INFTY = 999
    maxt = 999

    w = pb - pa
    if w[0] > size / 2.:
        w[0] -= size  # wrap around for torus
    if w[1] > size / 2.:
        w[1] -= size
    if w[0] < -size / 2.:
        w[0] += size
    if w[1] < -size / 2.:
        w[1] += size

    v = va - vb
    radius = ra + rb
    dist = sqrt(w[0] ** 2 + w[1] ** 2)

    if radius > dist:
        radius = .99 * dist

    a = v.dot(v)
    b = w.dot(v)
    c = w.dot(w) - radius * radius
    discr = b * b - a * c

    if (discr < 0) or (0.001 > a > - 0.001):
        return np.array([0, 0])

    discr = sqrt(discr)
    t1 = (b - discr) / a

    t = t1

    if t < 0:
        return np.array([0, 0])
    if t > maxt:
        return np.array([0, 0])

    d = k * exp(-t / t0) * (v - (v * b - w * a) / (discr)) / (a * t ** m) * (
        m / t + 1 / t0)

    return d


def update(dt):
    global center
    find_neighbors()

    F = []  # force

    for i in range(agents_num):
        F.append(np.zeros(2))

    for i in range(agents_num):
        # F.append(np.zeros(2))

        # vp = 1.4*v[i]/sqrt(v[i].dot(v[i]))
        F[i] += (goal_velocity[i] - velocity[i]) / .5
        F[i] += 1 * np.array([random.uniform(-1., 1.), random.uniform(-1., 1.)])

        for n, j in enumerate(neighbor[i]):  # j is neighboring agent
            # if j < i: continue

            t = ttc(center[i], center[j], velocity[i], velocity[j], radius, radius)

            d = center[i] - center[j]
            if d[0] > size / 2.:
                d[0] -= size  # wrap around for torus
            if d[1] > size / 2.:
                d[1] -= size
            if d[0] < -size / 2.:
                d[0] += size
            if d[1] < -size / 2.:
                d[1] += size

            r = radius
            dist = sqrt(d.dot(d))
            if dist < 2 * radius: r = dist / 2.001;  # shrink overlapping agents

            dEdx = dE(center[i], center[j], velocity[i], velocity[j], r, r)
            FAvoid = -dEdx

            mag = np.sqrt(FAvoid.dot(FAvoid))
            if mag > max_force:
                FAvoid = max_force * FAvoid / mag

            F[i] += FAvoid
            # F[j] -= FAvoid

            # if (t < 999): print t, dEdx

    for i in range(agents_num):
        a = F[i]
        velocity[i] += a * dt
        center[i] += velocity[i] * dt

        if center[i][0] < 0:
            center[i][0] = size  # wrap around for torus
        if center[i][1] < 0:
            center[i][1] = size
        if center[i][0] > size:
            center[i][0] = 0
        if center[i][1] > size:
            center[i][1] = 0


def on_key_press(event):
    global paused, step, quit, draw_velocities
    if event.keysym == "space":
        paused = not paused
    if event.keysym == "s":
        step = True
        paused = False
    if event.keysym == "v":
        draw_velocities = not draw_velocities
    if event.keysym == "Escape":
        quit = True


def draw_frame(dt=.05):
    global start_time, step, paused, itterations

    if itterations > max_itetrations or quit:  # Simulation Loop
        print("%s itterations ran ... quitting" % itterations)
        win.destroy()
    else:
        elapsed_time = time.time() - start_time
        start_time = time.time()
        if not paused:
            # if ittr%100 == 0 : print ittr,"/",maxIttr
            update(0.02)
            itterations += 1
        draw_world()
        if step == True:
            step = False
            paused = True

        # win.title("K.S.G. 2014 (Under Review) - " + str(round(1/elapsed_time,1)) +  " FPS")
        win.title("K.S.G. 2014 (Under Review)")
        win.after(framedelay, draw_frame)


# win.on_resize=resize

win.bind("<space>", on_key_press)
win.bind("s", on_key_press)
win.bind("<Escape>", on_key_press)
win.bind("v", on_key_press)

init_sim()
max_itetrations = 5000

start_time = time.time()
win.after(framedelay, draw_frame)
tk.mainloop()
