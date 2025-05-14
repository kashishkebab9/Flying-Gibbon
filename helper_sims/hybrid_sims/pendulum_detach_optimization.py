import casadi as ca
import numpy as np
import matplotlib.pyplot as plt

# Problem parameters
g = 9.81  # gravity (m/s^2)
L = 2.0   # length of the pendulum (m)
x_2 = 10.0  # target x position (m)
y_2 = 0.0   # target y position (m)
N = 50     # number of time steps

# Create optimization problem
opti = ca.Opti()

# Decision variables
v = opti.variable()      # initial velocity
theta = opti.variable()  # launch angle

# Time grid
T = opti.variable()      # Time of flight
t = np.linspace(0, 1, N)  # normalized time grid

# State trajectories based on the dynamics
x = lambda tau: v * ca.cos(theta) * tau * T
y = lambda tau: v * ca.sin(theta) * tau * T - 0.5 * g * (tau * T)**2

# Final position
x_N = x(1)
y_N = y(1)

# Objective: Minimize initial velocity
opti.minimize(v)

# Constraints
# Landing constraints
opti.subject_to(y_2 - L*ca.sin(ca.pi/4) <= y_N)  # Lower bound on y
opti.subject_to(y_N <= y_2 + L*ca.sin(ca.pi/4))  # Upper bound on y
opti.subject_to(x_2 - L <= x_N)                  # Lower bound on x
opti.subject_to(x_N <= x_2 - L*ca.cos(ca.pi/4))  # Upper bound on x

# Physical constraints
opti.subject_to(v >= 0)                          # Positive velocity
opti.subject_to(-2*ca.pi <= theta )  
# opti.subject_to(theta <= 2*ca.pi  )          # Angle between 0 and 90 degrees
opti.subject_to(T >= 0)                          # Positive time

# Set initial guesses
opti.set_initial(v, 10)
opti.set_initial(theta, ca.pi/4)
opti.set_initial(T, 1)

# Solver options
p_opts = {"expand": True}
s_opts = {"max_iter": 1000, "print_level": 0}
opti.solver("ipopt", p_opts, s_opts)

# Solve the optimization problem
try:
    sol = opti.solve()
    
    # Extract optimal values
    v_opt = sol.value(v)
    theta_opt = sol.value(theta)
    T_opt = sol.value(T)
    
    print(f"Optimal initial velocity: {v_opt:.3f} m/s")
    print(f"Optimal launch angle: {theta_opt*180/ca.pi:.3f} degrees")
    print(f"Time of flight: {T_opt:.3f} s")
    
    # Visualize the trajectory
    t_full = np.linspace(0, T_opt, 100)
    x_traj = v_opt * np.cos(theta_opt) * t_full
    y_traj = v_opt * np.sin(theta_opt) * t_full - 0.5 * g * t_full**2
    
    # Plot the trajectory and target region
    plt.figure(figsize=(10, 6))
    plt.plot(x_traj, y_traj, 'b-', label='Trajectory')
    
    # Plot target region
    x_target = [x_2 - L, x_2 - L*np.cos(np.pi/4), x_2 - L*np.cos(np.pi/4), x_2 - L]
    y_target = [y_2 - L*np.sin(np.pi/4), y_2 - L*np.sin(np.pi/4), y_2 + L*np.sin(np.pi/4), y_2 + L*np.sin(np.pi/4)]
    plt.fill(x_target, y_target, 'r', alpha=0.3, label='Target Region')
    
    plt.grid(True)
    plt.axis('equal')
    plt.xlabel('x (m)')
    plt.ylabel('y (m)')
    plt.title('Optimal Trajectory')
    plt.legend()
    plt.show()
    
except Exception as e:
    print(f"Optimization failed: {str(e)}")