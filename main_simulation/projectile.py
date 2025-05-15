from config import *
import casadi as cs
import os

def pendulum_position(theta):
    x = l * np.sin(theta)
    y = -l * np.cos(theta)
    return x, y

def flight_control(initial_state, target_position, N=1000):
    # Create an optimization problem
    opti = cs.Opti()
    
    # State dimensions: [x, y, theta, x_dot, y_dot, theta_dot, alpha, alpha_dot]
    nx = 8

    # Control dimensions: [f1, f2, tau_arm]
    nu = 3
    
    # Optimization variables
    X = opti.variable(nx, N+1)  # State trajectory
    U = opti.variable(nu, N)    # Control trajectory
    
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
    opti.subject_to(X[:, 0] == initial_state)
    
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
        opti.subject_to(X[0, k+1] == x[k] + dt * x_dot[k])
        opti.subject_to(X[1, k+1] == y[k] + dt * y_dot[k])
        opti.subject_to(X[2, k+1] == theta[k] + dt * theta_dot[k])
        opti.subject_to(X[3, k+1] == x_dot[k] + dt * x_ddot)
        opti.subject_to(X[4, k+1] == y_dot[k] + dt * y_ddot)
        opti.subject_to(X[5, k+1] == theta_dot[k] + dt * theta_ddot)
        opti.subject_to(X[6, k+1] == alpha[k] + dt * alpha_dot[k])
        opti.subject_to(X[7, k+1] == alpha_dot[k] + dt * alpha_ddot)

    
    # Compute arm end position at final time - MODIFIED: alpha=0 means upright
    arm_end_x = x[N] + l * cs.cos(theta[N] + alpha[N] + cs.pi/2)
    arm_end_y = y[N] + l * cs.sin(theta[N] + alpha[N] + cs.pi/2)
    
    # Terminal constraints: arm position reaches target
    opti.subject_to(arm_end_x == target_position[0])
    opti.subject_to(arm_end_y == target_position[1])
    
    # Terminal velocity constraints for smooth stopping
    opti.subject_to(x_dot[N] == 0)
    opti.subject_to(y_dot[N] == 0)
    # opti.subject_to(theta_dot[N] == 0)
    # opti.subject_to(alpha_dot[N] == 0)

    # Prevent Crazy flips
    theta_min = -np.pi/2
    theta_max = np.pi/2

    # Control constraints
    f_min = -20.0     # Minimum thrust (can't push)
    f_max = 20.0    # Maximum thrust
    tau_max = 10.0   # Maximum arm torque
    
    for k in range(N):
        opti.subject_to(f1[k] >= f_min)
        opti.subject_to(f1[k] <= f_max)
        opti.subject_to(f2[k] >= f_min)
        opti.subject_to(f2[k] <= f_max)
        opti.subject_to(tau_arm[k] >= -tau_max)
        opti.subject_to(tau_arm[k] <= tau_max)
        opti.subject_to(theta[k] >= theta_min)
        opti.subject_to(theta[k] <= theta_max)
        opti.subject_to(x[k] <= target_position[0] + l)

    # opti.subject_to(alpha[N] <= -np.pi/6)
    # opti.subject_to(alpha[N] >= -np.pi/3)

    # for k in range(N+1):
    #     opti.subject_to(alpha[k] >= -np.pi/2)
    #     opti.subject_to(alpha[k] <= np.pi/2)

    # Objective: minimize control effort
    objective = 0
    for k in range(N):
        total_thrust = f1[k] + f2[k]
        upward_thrust_component = total_thrust * cs.cos(theta[k])
        objective += 10* upward_thrust_component + 5* f1[k]**2 + 5 * f2[k]**2 + 2 * tau_arm[k]**2
        if k > 0:
            height_decrease = cs.fmax(0, y[k-1] - y[k])  # Positive when descending
            objective -= 20 * height_decrease
    
    opti.minimize(objective)
    
    # Solver options
    opts = {"ipopt.print_level": 3, "print_time": True}
    opti.solver('ipopt', opts)
    
    # Solve the optimization problem
    try:
        sol = opti.solve()
        X_opt = sol.value(X)
        U_opt = sol.value(U)
        return X_opt, U_opt
    except:
        print("Optimization failed. Check constraints and initial conditions.")
        return None, None



def simulate_projectile(t_release, theta_release, omega_release, phi_release_vel):
    # initialization 
    x_release, y_release = pendulum_position(theta_release)
    v_tan = l * omega_release
    v_x = v_tan * np.cos(theta_release)
    v_y = v_tan * np.sin(theta_release)
    
    # Calculate initial angle for the arm
    pendulum_point = (0, 0)
    dx = x_release - pendulum_point[0]
    dy = y_release - pendulum_point[1]
    angle_rad = np.arctan2(dy, dx)
    theta = theta_release  # Using the release angle as the initial theta
    alpha_release = np.pi/2 + theta - angle_rad
    
    # Set up initial state and target
    initial_state = np.array([x_release, y_release, theta, v_x, v_y, phi_release_vel, alpha_release, 0])
    target_position = np.array([8.0, 0.0])
    
    # Get optimal trajectory
    X_opt, U_opt = flight_control(initial_state, target_position)
    
    # Check if optimization was successful
    if X_opt is None or U_opt is None:
        print("Flight control optimization failed.")
        return [], [], [], [], []
    
    # Initialize result lists
    t_values, x_values, y_values, theta_values, alpha_values = [], [], [], [], []
    t = t_release
    
    # Extract states from the optimization result (transpose to get time as the first dimension)
    states = X_opt.T  # Shape: (N+1, 8)
    controls = U_opt.T  # Shape: (N, 3)
    
    # Append results at each time step
    for i in range(len(states)):
        state = states[i]
        x, y, th, xd, yd, thd, alpha, alphad = state
        
        t_values.append(t)
        x_values.append(x)
        y_values.append(y)
        theta_values.append(th)
        alpha_values.append(alpha)
        
        t += dt
        
        # Optional: Break if we hit the ground or go too far
        if y < -1.5 * l or t > t_release + 10.0:  # 10 seconds max simulation time
            break
    
    return t_values, x_values, y_values, theta_values, alpha_values