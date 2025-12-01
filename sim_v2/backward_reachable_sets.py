
import numpy as np
import matplotlib.pyplot as plt

# Simplified dynamics constants
g = 9.81
m = 1.0
L = 1.0
damping = 0.1
body_w = 0.5
I = 2.0

# Time discretization
dt = 0.05
T = 1.0
N = int(T / dt)

# Control bounds (force and torque)
f1_vals = np.linspace(-1, 1, 3)
f2_vals = np.linspace(-1, 1, 3)
tau_vals = np.linspace(-1, 1, 3)

# Grid (only over theta and theta_dot)
theta_vals = np.linspace(-np.pi, np.pi, 101)
theta_dot_vals = np.linspace(-4, 4, 101)
THETA, THETA_DOT = np.meshgrid(theta_vals, theta_dot_vals, indexing='ij')

reachable = np.zeros_like(THETA, dtype=bool)

# Target set (was centered at 0)
target_theta = np.pi
target_theta_dot = 0.0
target_radius = 0.1  # radius of target region

for i in range(len(theta_vals)):
    for j in range(len(theta_dot_vals)):
        if np.sqrt((theta_vals[i] - target_theta)**2 + (theta_dot_vals[j] - target_theta_dot)**2) < target_radius:
            reachable[i, j] = True


print(reachable)

# Discretized dynamics (2D slice only, assume phi and phi_dot are fixed for now)
def simple_dynamics(theta, theta_dot, u):
    f1_k, f2_k, tau_k = u
    phi = 0.0
    phi_dot = 0.0
    sin_diff = np.sin(theta - phi)
    theta_ddot = -g * np.sin(theta) + (1 / m) * sin_diff * (f1_k + f2_k) - damping * theta_dot
    return theta + dt * theta_dot, theta_dot + dt * theta_ddot

# Backward reachability
for step in range(N):
    new_reachable = reachable.copy()
    for i in range(len(theta_vals)):
        for j in range(len(theta_dot_vals)):
            if reachable[i, j]:
                theta = theta_vals[i]
                theta_dot = theta_dot_vals[j]
                for f1 in f1_vals:
                    for f2 in f2_vals:
                        for tau in tau_vals:
                            u = [f1, f2, tau]
                            prev_theta, prev_theta_dot = simple_dynamics(theta, theta_dot, u)
                            i_prev = np.searchsorted(theta_vals, prev_theta) - 1
                            j_prev = np.searchsorted(theta_dot_vals, prev_theta_dot) - 1
                            if 0 <= i_prev < len(theta_vals) and 0 <= j_prev < len(theta_dot_vals):
                                new_reachable[i_prev, j_prev] = True
    reachable = new_reachable.copy()


plt.figure(figsize=(8, 6))
plt.contourf(THETA, THETA_DOT, reachable, levels=1, cmap='Blues')
plt.xlabel('theta')
plt.ylabel('theta_dot')
plt.title('Backward Reachable Set (BRS) to Target at t = 0')
plt.grid(True)
plt.show()

