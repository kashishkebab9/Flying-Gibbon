import numpy as np

# Physical Parameters
g = 9.81        # gravity (m/s^2)
l = 1.0         # pendulum length (m)
m = 100.0       # pendulum mass (kg)
I = 2.0         # moment of inertia of rotating body (kg.m^2)
body_w = 0.5    # body width (m)
body_h = 0.2    # body height (m)

# Initial Conditions
theta_0 = -3 * np.pi / 4  # initial angle (rad)
theta_release = 0.7       # angle to detach (rad)
theta_release_vel = 10.0  # angular velocity to detach (rad/s)
phi_0 = 0.0               # initial body angle (rad)
phi_dot_0 = 0.0           # initial body angular velocity (rad/s)
x0 = np.array([theta_0, 0.0, phi_0, phi_dot_0])

# Simulation Parameters
fps = 60
dt = 1 / fps
total_time = 3.0
frames = int(total_time / dt)