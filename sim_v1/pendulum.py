from config import *
import casadi as ca
import numpy as np
import matplotlib.pyplot as plt
import os

def pendulum_position(theta):
    x = l * np.sin(theta)
    y = -l * np.cos(theta)
    return x, y

def run_optimization_casadi(x0, xf, N=100, h=0.01):
    opti = ca.Opti()

    # Problem dimensions
    nx = 4  
    nu = 3            

    # Decision variables
    X = opti.variable(nx, N + 1)   # state trajectory
    U = opti.variable(nu, N)       # control trajectory

    # Unpack control inputs
    theta = X[0, :]
    theta_dot = X[1, :]
    phi = X[2, :]
    phi_dot = X[3, :]

    # Unpack controls
    f1 = U[0, :]
    f2 = U[1, :]
    tau_arm = U[2, :]

    # Initial and terminal constraints
    opti.subject_to(X[:, 0] == x0)
    opti.subject_to(X[:, -1] == xf)


    # Dynamics constraints via trapezoidal integration
    for k in range(N):
        theta_k = theta[k]
        theta_dot_k = theta_dot[k]
        phi_k = phi[k]
        phi_dot_k = phi_dot[k]
        f1_k = f1[k]
        f2_k = f2[k]
        tau_k = tau_arm[k]

        sin_diff_k = ca.sin(theta_k - phi_k)
        theta_ddot_k = -g * ca.sin(theta_k) + (1 / m) * sin_diff_k * (f1_k + f2_k) - damping * theta_dot_k
        phi_ddot_k = -(body_w / 2) * f1_k + (body_w / 2) * f2_k + I * tau_k

        # Euler integration
        opti.subject_to(X[0, k + 1] == theta_k + dt * theta_dot_k)
        opti.subject_to(X[1, k + 1] == theta_dot_k + dt * theta_ddot_k)
        opti.subject_to(X[2, k + 1] == phi_k + dt * phi_dot_k)
        opti.subject_to(X[3, k + 1] == phi_dot_k + dt * phi_ddot_k)


    # Control constraints (optional)
    u_min, u_max = -10.0, 10.0
    opti.subject_to(opti.bounded(u_min, U, u_max))

    # Objective function: minimize squared thrusts
    objective = 0
    for k in range(N):
        objective += (f1[k]**2 + f2[k]**2)

    opti.minimize(objective)

    # Solver options
    opts = {"ipopt.print_level": 0, "print_time": False}
    opti.solver("ipopt", opts)

    try:
        sol = opti.solve()
        X_opt = sol.value(X).T
        U_opt = sol.value(U).T
        return X_opt, U_opt
    except RuntimeError:
        print("Optimization failed.")
        return None, None


def simulate_pendulum():
    X_opt, U_opt = run_optimization_casadi(x0, xf)

    theta_values = X_opt[:, 0]
    omega_values = X_opt[:, 1]
    phi_values = X_opt[:, 2]
    phi_dot_values = X_opt[:, 3]
    t_values = np.linspace(0, T, N + 1)

    # Determine release time (first time theta crosses theta_release threshold)
    t_release = None
    for i, theta in enumerate(theta_values):
        if theta >= theta_release:
            t_release = t_values[i]
            break
    print(theta_values[-1])
    print(phi_values[-1])

    return t_values, theta_values, omega_values, phi_values, phi_dot_values, t_release, U_opt
