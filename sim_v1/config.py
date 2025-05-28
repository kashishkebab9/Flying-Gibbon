import numpy as np

# Physical Parameters
g = 9.81        # gravity (m/s^2)
l = 1.0         # pendulum length (m)
m = 1.0       # pendulum mass (kg)
I = 2.0         # moment of inertia of rotating body (kg.m^2)
body_w = 0.5    # body width (m)
body_h = 0.2    # body height (m)

# Initial Conditions
theta_0 = 0
theta_release = np.pi/4       # angle to detach (rad)
theta_release_vel = 8  # angular velocity to detach (rad/s)
phi_release_vel = .2
phi_0 = 0.0               # initial body angle (rad)
phi_dot_0 = 0.0           # initial body angular velocity (rad/s)
x0 = np.array([theta_0, 0.0, phi_0, phi_dot_0])
xf = np.array([theta_release, theta_release_vel, 1.0, phi_release_vel])

arm_mass = 0.1 # mass of the arm
arm_I = 1/3 * arm_mass * l**2  # moment of inertia of the arm
# Pendulum optimization parameters
damping = 0.1
T = 2.0
N = 100
h = T / N
nx = 4
nu = 3
F_min, F_max = 0.0, 5.0  # Thrust bounds (N)
tau_min, tau_max = 0.0, 2.0  # Torque bounds (Nm)
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
total_time = 5.0
frames = int(total_time / dt)