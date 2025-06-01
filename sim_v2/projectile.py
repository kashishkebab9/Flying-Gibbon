import casadi as cs
import numpy as np
import yaml
import sys

def load_config(filename="config.yaml"):
    with open(filename, "r") as file:
        config = yaml.safe_load(file)
    return config

def pendulum_position(theta, config_file="config.yaml"):
    config = load_config(config_file)
    l = config["physical_parameters"]["pendulum_length"]
    x = l * np.sin(theta)
    y = -l * np.cos(theta)
    return x, y

def flight_control(x0, xf, N=200, T_max=10.0, config_file="config.yaml"):
    # Create an optimization problem
    config = load_config(config_file)
    opti = cs.Opti()
    
    # State dimensions: [x, y, theta, x_dot, y_dot, theta_dot, alpha, alpha_dot]
    nx = config["flight_optimization"]["state_dimension"]
    nu = config["flight_optimization"]["control_dimension"]
    l = config["physical_parameters"]["pendulum_length"]
    I = config["physical_parameters"]["moment_of_inertia_body"]
    arm_I = config["physical_parameters"]["arm_inertia"]
    m = config["physical_parameters"]["body_mass"]
    g = config["physical_parameters"]["gravity"]
    
    T = opti.variable()
    h = T/N
    X = opti.variable(nx, N+1)  # State trajectory
    U = opti.variable(nu, N)    # Control trajectory

    opti.subject_to(T>=1.0)
    opti.subject_to(T<=T_max)

    # Unpack states for better readability
    x = X[0, :]
    y = X[1, :]
    theta = X[2, :]
    x_dot = X[3, :]
    y_dot = X[4, :]
    theta_dot = X[5, :]
    alpha = X[6, :]
    alpha_dot = X[7, :]
    
    # Unpack controls
    f1 = U[0, :]
    f2 = U[1, :]
    tau_arm = U[2, :]
    
    # Initial state constraint
    opti.subject_to(X[:, 0] == x0)
    
    # Dynamics constraints
    for k in range(N):
        # Total force and torque
        total_f = f1[k] + f2[k]
        torque = (f2[k] - f1[k]) * l
        
        # Accelerations
        x_ddot = -total_f * cs.sin(theta[k]) / m
        y_ddot = total_f * cs.cos(theta[k]) / m - g
        theta_ddot = torque / I
        alpha_ddot = tau_arm[k] / arm_I
        
        # Euler integration
        opti.subject_to(X[0, k+1] == x[k] + h * x_dot[k])
        opti.subject_to(X[1, k+1] == y[k] + h * y_dot[k])
        opti.subject_to(X[2, k+1] == theta[k] + h * theta_dot[k])
        opti.subject_to(X[3, k+1] == x_dot[k] + h * x_ddot)
        opti.subject_to(X[4, k+1] == y_dot[k] + h * y_ddot)
        opti.subject_to(X[5, k+1] == theta_dot[k] + h * theta_ddot)
        opti.subject_to(X[6, k+1] == alpha[k] + h * alpha_dot[k])
        opti.subject_to(X[7, k+1] == alpha_dot[k] + h * alpha_ddot)

    
    # Compute arm end position at final time - MODIFIED: alpha=0 means upright
    arm_end_x = x[N] + l * cs.sin(theta[N] + alpha[N])
    arm_end_y = y[N] - l * cs.cos(theta[N] + alpha[N])

    # Terminal constraints: arm position reaches target
    opti.subject_to(arm_end_x == 10)
    opti.subject_to(arm_end_y == 0)
    opti.subject_to(x[N] <= 10)
    
    # Terminal velocity constraints for smooth stopping
    opti.subject_to(x_dot[N] <= 4)
    opti.subject_to(y_dot[N] == 0)
    # opti.subject_to(theta_dot[N] == 0)
    # opti.subject_to(alpha_dot[N] == 0)

    # Prevent Crazy flips
    theta_min = -np.pi/2
    theta_max = np.pi/2

    # Control constraints
    f_min = 0.0     # Minimum thrust (can't push)
    f_max = 15.0    # Maximum thrust
    tau_max = 5.0   # Maximum arm torque
    
    for k in range(N):
        opti.subject_to(y[k] >= -2.0)
        opti.subject_to(x[k] <= 10.0)
        opti.subject_to(f1[k] >= f_min)
        opti.subject_to(f1[k] <= f_max)
        opti.subject_to(f2[k] >= f_min)
        opti.subject_to(f2[k] <= f_max)
    #     opti.subject_to(tau_arm[k] >= -tau_max)
    #     opti.subject_to(tau_arm[k] <= tau_max)
    #     opti.subject_to(x[k] <= xf[0])
    #     opti.subject_to(theta[k] >= theta_min)
    #     opti.subject_to(theta[k] <= theta_max)
        # opti.subject_to(alpha[k] >= -cs.pi/2)
        # opti.subject_to(alpha[k] <= cs.pi/2)
    #     opti.subject_to(x[k] <= xf[0] + l)
    opti.subject_to(alpha[N] <= cs.pi/2)
    opti.subject_to(alpha[N] >= -cs.pi/2)

    # opti.subject_to(alpha[N] <= -np.pi/6)
    # opti.subject_to(alpha[N] >= -np.pi/3)

    # Objective: minimize control effort
    objective = 0
    objective +=  (T) ** 2  # Encourage shorter flight
    for k in range(N):
        total_thrust = f1[k] + f2[k] 
        upward_thrust_component = total_thrust * cs.cos(theta[k])
        objective += 10 * f1[k] ** 2
        objective += 10 * f2[k] ** 2
        objective += tau_arm[k] ** 2
    
    opti.minimize(objective)
    
    # Solver options
    opts = {"ipopt.print_level": 3, "print_time": True}
    opti.solver('ipopt', opts)
    # Solve the optimization problem
    try:
        sol = opti.solve()
        X_opt = sol.value(X)
        U_opt = sol.value(U)
        T_opt = sol.value(T)
        print(f"topt: {T_opt}")
        print(f"dt: {T_opt/N}")
        dt_out = T_opt/N
        # N = int(T_opt/h)
        return X_opt, U_opt, T_opt, dt_out
    except:
        print("Optimization failed. Check constraints and initial conditions.")
        sys.exit()
        return None, None, None

def simulate_projectile(t_release, config_file="config.yaml", release_state=None):
    config = load_config(config_file)
    
    # Load physical parameters
    l = config["physical_parameters"]["pendulum_length"]
    dt_zoh = config["physical_parameters"]["dynamics_rate"]  # fixed sampling interval

    if release_state == None:
        release_state = config["boundary_conditions"]["release_state"]
        theta_release = release_state[0]
        theta_dot_release = release_state[1]
        phi_release = release_state[2]
        phi_dot_release = release_state[3]
    else:
        theta_release = release_state[0]
        theta_dot_release = release_state[1]
        phi_release = release_state[2]
        phi_dot_release = release_state[3]

    
    # Initial conditions
    x_release, y_release = pendulum_position(theta_release)
    v_tan = l * theta_dot_release
    v_x = v_tan * np.cos(theta_release)
    v_y = v_tan * np.sin(theta_release)
    
    dx = x_release
    dy = y_release
    angle_rad = np.arctan2(dy, dx)
    print("HERE")
    print(phi_release)
    if phi_release == 0:
        alpha_release = np.pi/2 +  theta_release - angle_rad
    elif phi_release > 0:
        alpha_release = theta_release - angle_rad
    elif phi_release < 0:
        alpha_release = np.pi/2  + theta_release - angle_rad 
    
    initial_state = np.array([x_release, y_release, theta_release+phi_release, v_x, v_y, phi_dot_release, alpha_release, 0])
    final_state = np.array([10, -1, 0, 0, 0, 0, 0, 0])

    # Optimize flight
    X_opt, U_opt, T_opt, dt_opt = flight_control(initial_state, final_state, config_file=config_file)
    
    if X_opt is None:
        print("Flight control optimization failed.")
        return [], [], [], [], [], [], None
    
    states = X_opt.T  # shape (N+1, 8)
    controls = U_opt.T  # shape (N, 3)
    N = states.shape[0] - 1
    
    # Zero-order hold: sample fixed-rate output by repeating states
    t_values = []
    x_values = []
    y_values = []
    theta_values = []
    alpha_values = []
    control_values = []

    total_steps = int(np.floor(T_opt / dt_zoh))
    print(total_steps)

    epsilon = 1e-6
    for i in range(total_steps):
        t = t_release + i * dt_zoh
        idx = int(np.floor(i * dt_zoh / dt_opt + epsilon))

        if idx >= N:
            idx = N - 1

        state = states[idx]
        control = controls[idx]

        t_values.append(t)
        x_values.append(state[0])
        y_values.append(state[1])
        theta_values.append(state[2])
        alpha_values.append(state[6])
        control_values.append(control)

        if t > t_release + 10.0:
            break

    print(len(t_values))
    print(len(x_values))
    
    return (
        t_values,
        x_values,
        y_values,
        theta_values,
        alpha_values,
        control_values,
        T_opt
    )