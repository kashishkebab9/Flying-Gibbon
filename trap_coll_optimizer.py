import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.transforms as tr
from scipy.optimize import minimize

from matplotlib.animation import FFMpegWriter
# === System Parameters ===
L = 1.0
body_w = 0.3
body_h = 0.1
g = 9.81
m = 2.0
I = 2.0
damping = 0.5

# === Optimization Parameters ===
T = 4.0
N = 80
h = T / N
nx = 4
nu = 3

x0 = np.array([0.0, 0.0, 0.0, 0.0])
xf = np.array([np.pi, 0.0, 0.0, 0.0])

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

# === Dynamics ===
def dynamics(x, u):
    theta, theta_dot, phi, phi_dot = x
    Fl, Fr, tau = u
    sin_diff = np.sin(theta - phi)
    theta_ddot = -g * np.sin(theta) + (1 / m) * sin_diff * (Fl + Fr) - damping * theta_dot
    phi_ddot = -(body_w / 2) * Fl + (body_w / 2) * Fr + I * tau
    return np.array([theta_dot, theta_ddot, phi_dot, phi_ddot])

# === Helper: Pack/Unpack ===
def pack(X, U):
    return np.concatenate([X.flatten(), U.flatten()])

def unpack(z):
    X = z[:N * nx].reshape(N, nx)
    U = z[N * nx:].reshape(N, nu)
    return X, U

# === Cost Function ===
def cost(z):
    _, U = unpack(z)
    return np.sum(U[:, :2] ** 2) * h  # penalize thrusts

# === Constraints ===
def constraints(z):
    X, U = unpack(z)
    cons = []

    for k in range(N - 1):
        f1 = dynamics(X[k], U[k])
        f2 = dynamics(X[k + 1], U[k + 1])
        defect = X[k + 1] - X[k] - 0.5 * h * (f1 + f2)
        cons.append(defect)

    cons.append(X[0] - x0)
    cons.append(X[-1] - xf)

    return np.concatenate(cons)

# === Solve Optimization ===
X_init = np.linspace(x0, xf, N)
U_init = np.zeros((N, nu))
z0 = pack(X_init, U_init)

res = minimize(cost, z0, constraints={'type': 'eq', 'fun': constraints}, bounds=bounds,
               options={'disp': True, 'maxiter': 100}, method='SLSQP')

X_opt, U_opt = unpack(res.x)

# === Animation ===
dt = h
fig, ax = plt.subplots()
ax.set_aspect('equal')
ax.set_xlim(-1.5, 1.5)
ax.set_ylim(-1.5, 1.5)

line, = ax.plot([], [], 'ko-', lw=2)
body_rect = plt.Rectangle((0, 0), body_w, body_h, color='blue', alpha=0.5)
ax.add_patch(body_rect)

def init():
    line.set_data([], [])
    body_rect.set_xy((-10, -10))
    return line, body_rect

def update(frame):
    theta, _, phi, _ = X_opt[frame]

    x_p = L * np.sin(theta)
    y_p = -L * np.cos(theta)

    rot = np.array([[np.cos(theta + phi), -np.sin(theta + phi)],
                    [np.sin(theta + phi),  np.cos(theta + phi)]])
    
    corners = np.array([
        [-body_w/2, -body_h/2],
        [ body_w/2, -body_h/2],
        [ body_w/2,  body_h/2],
        [-body_w/2,  body_h/2]
    ])
    rotated = (rot @ corners.T).T + np.array([x_p, y_p])

    line.set_data([0, x_p], [0, y_p])

    transform = (tr.Affine2D()
                 .rotate_around(x_p, y_p, phi + theta)
                 + ax.transData)
    body_rect.set_xy((x_p - body_w / 2, y_p - body_h / 2))
    body_rect.set_transform(transform)
    return line, body_rect

ani = FuncAnimation(fig, update, frames=N, init_func=init,
                    blit=True, interval=dt*1000, repeat=False)
plt.title("Optimized Trajectory: Pendulum with Actuated Body")
plt.show()

writer = FFMpegWriter(fps=int(1 / dt), metadata=dict(artist='Trajectory Opt'), bitrate=1800)
ani.save("optimized_pendulum_trajectory.mp4", writer=writer)
print("Video saved as optimized_pendulum_trajectory.mp4")

def unpack_decision_variables(z):
    X = z[:N*nx].reshape((N, nx))
    U = z[N*nx:].reshape((N, nu))
    return X, U

X_opt, U_opt = unpack_decision_variables(res.x)
time = np.linspace(0, T, N)

# Plot
plt.figure(figsize=(10, 6))
plt.subplot(2, 1, 1)
plt.plot(time, X_opt[:, 0], label='theta')
plt.plot(time, X_opt[:, 2], label='phi')
plt.legend()
plt.title('State Trajectories')

plt.subplot(2, 1, 2)
plt.plot(time, U_opt[:, 0], label='Fl')
plt.plot(time, U_opt[:, 1], label='Fr')
plt.plot(time, U_opt[:, 2], label='tau')
plt.legend()
plt.title('Control Inputs')
plt.xlabel('Time (s)')
plt.tight_layout()
plt.show()

