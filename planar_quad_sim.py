import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# System parameters
m = 1.0        # mass of quadrotor (kg)
I = 0.02       # moment of inertia of quadrotor
l = 0.5        # half-length of the quadrotor (distance from center to rotor)
g = 9.81       # gravity
arm_len = 1.0  # length of the arm
arm_mass = 0.3 # mass of the arm
arm_I = 1/3 * arm_mass * arm_len**2  # moment of inertia of the arm

# Initial state: [x, y, theta, x_dot, y_dot, theta_dot, alpha, alpha_dot]
state = np.array([-4.0, 2.0, 0.0,   0.0, 0.0, 0.0,     np.pi/4, 0.0])

dt = 0.02

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

def step(state, u):
    # Simple Euler integration
    k1 = dynamics(state, u)
    next_state = state + dt * k1
    return next_state

# Control input: (f1, f2, tau_arm)
def controller(state, t):
    # Simple hover and oscillating torque to show arm movement
    f1 = m * g / 2
    f2 = m * g / 2
    taum = 0.05 * np.sin(2 * np.pi * t)
    return np.array([f1, f2, taum])

# Setup the plot
fig, ax = plt.subplots()
ax.set_xlim(-10, 10)
ax.set_ylim(-10, 10)
quad_line, = ax.plot([], [], 'k-', lw=4)
arm_line, = ax.plot([], [], 'r-', lw=2)

def init():
    quad_line.set_data([], [])
    arm_line.set_data([], [])
    return quad_line, arm_line

def update(frame):
    global state
    t = frame * dt
    u = controller(state, t)
    state = step(state, u)

    x, y, th, _, _, _, alpha, _ = state

    # Quadrotor corners
    x1 = x - l * np.cos(th)
    y1 = y - l * np.sin(th)
    x2 = x + l * np.cos(th)
    y2 = y + l * np.sin(th)

    # Arm end position
    arm_x = x + arm_len * np.cos(th + alpha)
    arm_y = y + arm_len * np.sin(th + alpha)

    quad_line.set_data([x1, x2], [y1, y2])
    arm_line.set_data([x, arm_x], [y, arm_y])
    return quad_line, arm_line

ani = FuncAnimation(fig, update, frames=600, init_func=init, blit=True, interval=dt*1000)
plt.xlabel("X (m)")
plt.ylabel("Y (m)")
plt.title("Planar Quadrotor with Center Arm")
plt.grid()
plt.show()
