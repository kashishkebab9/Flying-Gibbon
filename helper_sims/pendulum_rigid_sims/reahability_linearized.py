
import numpy as np
import matplotlib.pyplot as plt

# === System Parameters ===
g = 9.81
L = 1.0
m = 1.0
damping = 0.5
body_w = 0.3
I = 1.0

# === Pendulum Dynamics (returns only 4D) ===
def pendulum_dynamics(x, u, t, g=g, L=L):
    theta, theta_dot, phi, phi_dot = x[0], x[1], x[2], x[3]
    f1_k, f2_k, tau_k = u[0], u[1], u[2]
    sin_diff = np.sin(theta - phi)
    theta_ddot = -g * np.sin(theta) + (1 / m) * sin_diff * (f1_k + f2_k) - damping * theta_dot
    phi_ddot = -(body_w / 2) * f1_k + (body_w / 2) * f2_k + I * tau_k
    return np.array([theta_dot, theta_ddot, phi_dot, phi_ddot])

# === Finite Difference Linearization ===
def finite_difference_jacobian(f, x, u, eps=1e-5):
    nx, nu = len(x), len(u)
    A = np.zeros((nx, nx))
    B = np.zeros((nx, nu))
    fx = f(x, u, 0)

    for i in range(nx):
        dx = np.zeros(nx)
        dx[i] = eps
        A[:, i] = (f(x + dx, u, 0) - fx) / eps

    for i in range(nu):
        du = np.zeros(nu)
        du[i] = eps
        B[:, i] = (f(x, u + du, 0) - fx) / eps

    return A, B

# === Prune Duplicates Using Rounding ===
def prune_duplicates(states, tol=1e-2):
    seen = set()
    unique = []
    for s in states:
        key = tuple(np.round(s / tol) * tol)
        if key not in seen:
            seen.add(key)
            unique.append(s)
    return unique

# === Propagate Reachable Set ===
def propagate_linearized_reachability(x0, input_samples, dt, steps, f):
    reachable_set = [x0]

    for _ in range(steps):
        new_states = []
        for x in reachable_set:
            for u in input_samples:
                A, B = finite_difference_jacobian(f, x, u)
                x_next = x + dt * (A @ x + B @ u)
                new_states.append(x_next)
        reachable_set = prune_duplicates(new_states)
    
    return reachable_set

# === Sample Input Space ===
u_samples = []
for f1 in np.linspace(0, 5, 5):
    for f2 in np.linspace(0, 5, 5):
        for tau in np.linspace(-2, 2, 5):
            u_samples.append(np.array([f1, f2, tau]))

# === Initial State and Propagation ===
x0 = np.array([0.0, 0.0, 0.0, 0.0])
reachable = propagate_linearized_reachability(x0, u_samples, dt=0.05, steps=10, f=pendulum_dynamics)

# === Convert to Array for Plotting ===
reachable_arr = np.array(reachable)
# theta_vals = reachable_arr[:, 0]
# theta_dot_vals = reachable_arr[:, 1]

# # === Plotting ===
# plt.figure(figsize=(8, 6))
# plt.scatter(theta_vals, theta_dot_vals, alpha=0.6, s=10, c='blue')
# plt.xlabel("theta")
# plt.ylabel("theta_dot")
# plt.title("Reachable Set Projection (theta vs theta_dot)")
# plt.grid(True)
# plt.tight_layout()
# plt.show()

# === Extract theta and phi only ===
theta_vals = reachable_arr[:, 0]
phi_vals = reachable_arr[:, 2]

# === Plotting (theta vs phi) ===
plt.figure(figsize=(8, 6))
plt.scatter(theta_vals, phi_vals, alpha=0.6, s=10, c='green')
plt.xlabel("theta")
plt.ylabel("phi")
plt.title("Reachable Set Projection (theta vs phi)")
plt.grid(True)
plt.tight_layout()
plt.show()

