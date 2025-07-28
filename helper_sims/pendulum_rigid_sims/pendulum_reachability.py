
import numpy as np
import matplotlib.pyplot as plt

# Grid setup
theta_vals = np.linspace(-2*np.pi, 2*np.pi, 201)
omega_vals = np.linspace(-8, 8, 201)
dtheta = theta_vals[1] - theta_vals[0]
domega = omega_vals[1] - omega_vals[0]
T = 5.0
dt = 0.05
N = int(T / dt)

# Meshgrid
THETA, OMEGA = np.meshgrid(theta_vals, omega_vals, indexing='ij')

# Target set around (π, 0)
target = (np.pi, 0.0)
target_radius = 0.2
V = np.sqrt((THETA - target[0])**2 + (OMEGA - target[1])**2) - target_radius  # signed distance

# Dynamics
g, l = 9.81, 1.0
u_bounds = [-1.0, 0.0, 1.0]

# Value iteration: backward in time
for _ in range(N):
    V_next = np.copy(V)
    for i in range(1, len(theta_vals)-1):
        for j in range(1, len(omega_vals)-1):
            theta, omega = theta_vals[i], omega_vals[j]
            dV_dtheta = (V[i+1, j] - V[i-1, j]) / (2*dtheta)
            dV_domega = (V[i, j+1] - V[i, j-1]) / (2*domega)

            min_H = float('inf')
            for u in u_bounds:
                f1 = omega
                f2 = -g/l * np.sin(theta) + u
                H = dV_dtheta * f1 + dV_domega * f2
                min_H = min(min_H, H)

            V_next[i, j] = V[i, j] - dt * min_H

    V = np.copy(V_next)

# Plot zero sublevel set (reachable set)
plt.contour(THETA, OMEGA, V, levels=[0], colors='red')
plt.xlabel('θ (rad)')
plt.ylabel('ω (rad/s)')
plt.title('HJ Approximate Reachable Set (Backward Reachability)')
plt.grid(True)
plt.show()
