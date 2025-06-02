import casadi as cs
import yaml
import numpy as np
import sys

def load_config(filename="config.yaml"):
    with open(filename, "r") as file:
        config = yaml.safe_load(file)
    return config

def pendulum_solve_traj(x0, xf, h=0.01, T_max=2.0, config_file="config.yaml"):
    config = load_config(config_file)
    
    opti = cs.Opti()

    # Parsing config parameters
    nx = config["pendulum_optimization"]["state_dimension"]
    nu = config["pendulum_optimization"]["control_dimension"]
    damping = config["pendulum_optimization"]["damping"]
    body_w = config["physical_parameters"]["body_width"]
    g = config["physical_parameters"]["gravity"]
    I = config["physical_parameters"]["moment_of_inertia_body"]
    m = config["physical_parameters"]["body_mass"]

    # Decision variables
    T = opti.variable()
    N = int(T_max/h) # Number of frames
    X = opti.variable(nx, N + 1)   # state trajectory
    U = opti.variable(nu, N)       # control trajectory

    opti.subject_to(T>=0.5)
    opti.subject_to(T<=T_max)

    # Unpack states 
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

        sin_diff_k = cs.sin(theta_k - phi_k)
        theta_ddot_k = -g * cs.sin(theta_k) + (1 / m) * sin_diff_k * (f1_k + f2_k) - damping * theta_dot_k
        phi_ddot_k = -(body_w / 2) * f1_k + (body_w / 2) * f2_k + I * tau_k

        # Euler integration
        opti.subject_to(X[0, k + 1] == theta_k + h * theta_dot_k)
        opti.subject_to(X[1, k + 1] == theta_dot_k + h * theta_ddot_k)
        opti.subject_to(X[2, k + 1] == phi_k + h * phi_dot_k)
        opti.subject_to(X[3, k + 1] == phi_dot_k + h * phi_ddot_k)


    # Control constraints (optional)
    u_range = config["pendulum_optimization"]["control_bounds"]["force_rotor"]
    u_min = u_range[0]
    u_max = u_range[1]
    opti.subject_to(opti.bounded(u_min, U, u_max))

    # Objective function: minimize squared thrusts
    objective = 0
    for k in range(N):
        objective += (f1[k]**2 + f2[k]**2 + tau_arm[k]**2)

    opti.minimize(objective)

    # Solver options
    opts = {"ipopt.print_level": 0, "print_time": False}
    opti.solver("ipopt", opts)

    try:
        sol = opti.solve()
        X_opt = sol.value(X).T
        U_opt = sol.value(U).T
        T_opt = sol.value(T)
        # N = int(T_opt/h)
        return X_opt, U_opt, T_opt, N
    except RuntimeError:
        print("Optimization failed.")
        return [], [], [], []

def simulate_pendulum(config_file="config.yaml", release_state=None):
    config = load_config(config_file)
    x0 = config["boundary_conditions"]["initial_state"]
    if release_state == None:
        xf = config["boundary_conditions"]["release_state"]
    else:
        xf = release_state
    
    X_opt, U_opt, T_opt, N = pendulum_solve_traj(x0, xf)
    if len(X_opt) == 0:
        return [], [], [], [], [], [], []

    theta_values = X_opt[:, 0]
    omega_values = X_opt[:, 1]
    phi_values = X_opt[:, 2]
    phi_dot_values = X_opt[:, 3]
    t_values = np.linspace(0, T_opt, N + 1)

    # Determine release time (first time theta crosses theta_release threshold)
    t_release = None
    for i, theta in enumerate(theta_values):
        if theta >= xf[0]:
            t_release = t_values[i]
            break
    print(theta_values[-1])
    print(phi_values[-1])
    print(f"T_release : {t_release}")

    return t_values, theta_values, omega_values, phi_values, phi_dot_values, t_release, U_opt