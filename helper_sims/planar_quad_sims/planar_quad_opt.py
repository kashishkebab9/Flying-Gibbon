import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import casadi as cs

# System parameters
m = 1.0        # mass of quadrotor (kg)
I = 0.02       # moment of inertia of quadrotor
l = 0.5        # half-length of the quadrotor (distance from center to rotor)
g = 9.81       # gravity
arm_len = 2.0  # length of the arm
arm_mass = 0.1 # mass of the arm
arm_I = 1/3 * arm_mass * arm_len**2  # moment of inertia of the arm

# Time discretization
dt = 0.02
T = 5.0  # Total trajectory time (seconds)
N = int(T/dt)  # Number of time steps

def dynamics(state, u):
    x, y, th, xd, yd, thd, alpha, alphad = state
    f1, f2, taum = u  # left and right thrusts, and torque on arm
    # Total force and torque on quadrotor
    total_f = f1 + f2
    torque = (f2 - f1) * l
    # Quadrotor translational acceleration
    xdd = -total_f * np.sin(th) / m
    ydd = total_f * np.cos(th) / m - g
    # Quadrotor angular acceleration
    thdd = torque / I
    # Arm angular acceleration (relative to body)
    alphadd = taum / arm_I
    return np.array([xd, yd, thd, xdd, ydd, thdd, alphad, alphadd])

def planar_quadrotor_arm_optimization(initial_state, target_position, N=10000):
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
    arm_end_x = x[N] + arm_len * cs.cos(theta[N] + alpha[N] + cs.pi/2)
    arm_end_y = y[N] + arm_len * cs.sin(theta[N] + alpha[N] + cs.pi/2)
    
    # Terminal constraints: arm position reaches target
    opti.subject_to(arm_end_x == target_position[0])
    opti.subject_to(arm_end_y == target_position[1])
    
    # Terminal velocity constraints for smooth stopping
    opti.subject_to(x_dot[N] == 0)
    opti.subject_to(y_dot[N] == 0)
    opti.subject_to(theta_dot[N] == 0)
    opti.subject_to(alpha_dot[N] == 0)

    # Prevent Crazy flips
    theta_min = -np.pi/2
    theta_max = np.pi/2

    # Control constraints
    f_min = 0.0     # Minimum thrust (can't push)
    f_max = 20.0    # Maximum thrust
    tau_max = 1.0   # Maximum arm torque
    
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

    opti.subject_to(alpha[N] <= -np.pi/6)
    opti.subject_to(alpha[N] >= -np.pi/3)

    for k in range(N+1):
        opti.subject_to(alpha[k] >= -np.pi/2)
        opti.subject_to(alpha[k] <= np.pi/2)

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

# Initial state: [x, y, theta, x_dot, y_dot, theta_dot, alpha, alpha_dot]
# MODIFIED: Changed alpha to -np.pi/4 + np.pi/2 to account for new zero position
initial_state = np.array([-10.0, 4.0, -np.pi/4, 6.43, 5.4, 0.0, np.pi/4, 0.0])

# Target position for the arm end-effector
target_position = np.array([10.0, 6.0])

# Solve the optimization problem
X_opt, U_opt = planar_quadrotor_arm_optimization(initial_state, target_position, N)

if X_opt is None:
    print("Optimization failed, using default controller for visualization")
    # Original controller function for fallback
    def controller(state, t):
        f1 = m * g / 2
        f2 = m * g / 2
        taum = 0.05 * np.sin(2 * np.pi * t)
        return np.array([f1, f2, taum])
    
    # Setup for simulation with original controller
    state = initial_state.copy()
    states = np.zeros((N+1, 8))
    states[0] = state
    controls = np.zeros((N, 3))
    
    for i in range(N):
        t = i * dt
        u = controller(state, t)
        controls[i] = u
        state = state + dt * dynamics(state, u)
        states[i+1] = state
else:
    # Use optimized trajectories
    states = X_opt.T  # Shape: (N+1, 8)
    controls = U_opt.T  # Shape: (N, 3)
    
# Setup the plot
fig, ax = plt.subplots(figsize=(10, 6))
ax.set_xlim(-10, 10)
ax.set_ylim(0, 10)
quad_line, = ax.plot([], [], 'k-', lw=4)
arm_line, = ax.plot([], [], 'r-', lw=2)
target_point, = ax.plot([], [], 'go', markersize=8)
traj_line, = ax.plot([], [], 'b--', lw=1, alpha=0.7)

# Initialize lists to collect trajectory points
arm_end_positions_x = []
arm_end_positions_y = []

def init():
    quad_line.set_data([], [])
    arm_line.set_data([], [])
    target_point.set_data([target_position[0]], [target_position[1]])
    traj_line.set_data([], [])
    return quad_line, arm_line, target_point, traj_line

def update(frame):
    state = states[frame]
    x, y, th, _, _, _, alpha, _ = state
    
    # Quadrotor corners
    x1 = x - l * np.cos(th)
    y1 = y - l * np.sin(th)
    x2 = x + l * np.cos(th)
    y2 = y + l * np.sin(th)
    
    # MODIFIED: Arm end position - alpha=0 means arm points up (rotated by pi/2)
    arm_x = x + arm_len * np.cos(th + alpha + np.pi/2)
    arm_y = y + arm_len * np.sin(th + alpha + np.pi/2)
    
    # Store arm end position for trajectory visualization
    arm_end_positions_x.append(arm_x)
    arm_end_positions_y.append(arm_y)
    
    quad_line.set_data([x1, x2], [y1, y2])
    arm_line.set_data([x, arm_x], [y, arm_y])
    traj_line.set_data(arm_end_positions_x, arm_end_positions_y)
    
    return quad_line, arm_line, target_point, traj_line

ani = FuncAnimation(fig, update, frames=N, init_func=init, blit=True, interval=dt*1000)

# Plot controls
fig_controls, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
time = np.arange(N) * dt

ax1.plot(time, controls[:, 0], 'b-', label='Left Thrust (f1)')
ax1.plot(time, controls[:, 1], 'r-', label='Right Thrust (f2)')
ax1.set_ylabel('Thrust (N)')
ax1.legend()
ax1.grid(True)

ax2.plot(time, controls[:, 2], 'g-', label='Arm Torque')
ax2.set_ylabel('Torque (Nm)')
ax2.legend()
ax2.grid(True)

# Plot states
ax3.plot(time, states[:-1, 0], 'b-', label='x position')
ax3.plot(time, states[:-1, 1], 'r-', label='y position')
ax3.plot(time, states[:-1, 2], 'g-', label='theta')
ax3.plot(time, states[:-1, 6], 'm-', label='alpha')
ax3.set_xlabel('Time (s)')
ax3.set_ylabel('States')
ax3.legend()
ax3.grid(True)

plt.tight_layout()

# Main animation plot settings
ax.set_xlabel("X (m)")
ax.set_ylabel("Y (m)")
ax.set_title("Planar Quadrotor with Arm - Optimal Trajectory")
ax.grid(True)

plt.show()