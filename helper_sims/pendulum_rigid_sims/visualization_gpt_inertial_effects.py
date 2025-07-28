
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# === Parameters ===
L = 1.0             # Pendulum length (m)
g = 9.81            # Gravity (m/s^2)
dt = 0.02           # Time step
T = 20              # Total time (s)
steps = int(T / dt)

# Body shape
body_w = 0.4        # Width of rectangular body (m)
body_h = 0.15       # Height

# === Initial Conditions ===
theta = np.pi / 3           # initial pendulum angle (rad)
theta_dot = 0.0             # initial angular velocity

phi = 0.0                   # initial body angle (inertial frame)
phi_dot = 0.2               # body rotates with constant angular velocity

# === Equations of Motion ===
def pendulum_ddot(theta, theta_dot):
    return - (g / L) * np.sin(theta)

# === Storage ===
theta_hist = []
phi_hist = []

for _ in range(steps):
    # Integrate pendulum dynamics (semi-implicit Euler)
    theta_dd = pendulum_ddot(theta, theta_dot)
    theta_dot += theta_dd * dt
    theta += theta_dot * dt

    # Body angle evolves independently with constant angular velocity
    phi += phi_dot * dt

    # Store for animation
    theta_hist.append(theta)
    phi_hist.append(phi)

# === Animation ===
fig, ax = plt.subplots()
ax.set_aspect('equal')
ax.set_xlim(-1.5 * L, 1.5 * L)
ax.set_ylim(-1.5 * L, 1.2 * L)

# Pendulum line
pend_line, = ax.plot([], [], 'k-', lw=2)

# Rectangular body (drawn as 4 corners)
body_rect = plt.Polygon([[0,0],[0,0],[0,0],[0,0]], closed=True, fc='green', ec='black')
ax.add_patch(body_rect)

def update(i):
    theta = theta_hist[i]
    phi = phi_hist[i]

    # Pendulum end position
    x_p = L * np.sin(theta)
    y_p = -L * np.cos(theta)

    # Update pendulum line
    pend_line.set_data([0, x_p], [0, y_p])

    # Body orientation in inertial frame = phi
    # Compute rectangle corners in body frame
    corners = np.array([
        [-body_w/2, -body_h/2],
        [ body_w/2, -body_h/2],
        [ body_w/2,  body_h/2],
        [-body_w/2,  body_h/2]
    ])

    # Rotate by phi and translate to (x_p, y_p)
    R = np.array([
        [np.cos(phi), -np.sin(phi)],
        [np.sin(phi),  np.cos(phi)]
    ])
    rotated = corners @ R.T + np.array([x_p, y_p])
    body_rect.set_xy(rotated)

    return pend_line, body_rect

ani = FuncAnimation(fig, update, frames=steps, interval=dt*1000)
plt.show()
