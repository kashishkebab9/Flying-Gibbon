import numpy as np

# Physical Parameters
g = 9.81        # gravity (m/s^2)
l = 1.0         # pendulum length (m)
m = 1.0       # pendulum mass (kg)
I = 2.0         # moment of inertia of rotating body (kg.m^2)
body_w = 0.5    # body width (m)
body_h = 0.2    # body height (m)

# Initial Conditions
theta_0 = -3 * np.pi / 4  # initial angle (rad)
theta_release = np.pi/4       # angle to detach (rad)
theta_release_vel = 10.0  # angular velocity to detach (rad/s)
phi_0 = 0.0               # initial body angle (rad)
phi_dot_0 = 0.0           # initial body angular velocity (rad/s)
x0 = np.array([theta_0, 0.0, phi_0, phi_dot_0])
xf = np.array([theta_release, theta_release_vel, 0.0, 0.0])

# Pendulum optimization parameters
damping = 0.1
T = 4.0
N = 100
h = T / N
nx = 4
nu = 3
F_min, F_max = 0.0, 5.0  # Thrust bounds (N)
tau_min, tau_max = -2.0, 2.0  # Torque bounds (Nm)
# No bounds for state variables (None)
state_bounds = [(None, None)] * (N * nx)

# Bounds for control variables
control_bounds = []
for _ in range(N):
    control_bounds += [(F_min, F_max),  # Fl
                    (F_min, F_max),  # Fr
                    (tau_min, tau_max)]  # tau

# Combine
bounds = state_bounds + control_bounds

# Simulation Parameters
fps = 60
dt = 1 / fps
total_time = 10.0
frames = int(total_time / dt)