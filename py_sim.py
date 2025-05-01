
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.transforms as tr

# Parameters
L = 1.0         # Length of pendulum
body_w = 0.3    # Width of body
body_h = 0.1    # Height of body
dt = 0.05
T = 10
g = 9.81
m = 2.0
damping = 0.5
I = 2.0

# Initial state: [theta, theta_dot, phi, phi_dot]
x = np.array([0.5, 0.0, 0.0, 0.0])  # swing angle, angular velocity, body angle, body angular velocity

# Control inputs: [F1, F2] (thrusts on propellers at either end of the body)
def control(x, t):
    theta, theta_dot, phi, phi_dot = x

    # Desired behavior
    theta_des = np.pi
    theta_dot_des = 0.0

    # PD gains
    Kp = 20.0
    Kd = 5.0

    # Desired theta acceleration
    theta_ddot_des = -Kp * (theta - theta_des) - Kd * (theta_dot - theta_dot_des)

    # Cancel dynamics to solve for Fl + Fr
    sin_diff = np.sin(theta - phi)
    if abs(sin_diff) < 1e-3:
        sin_diff = np.sign(sin_diff) * 1e-3  # avoid divide by zero

    total_thrust = (m / sin_diff) * (theta_ddot_des + g * np.sin(theta) + damping * theta_dot)

    # Split thrusts evenly (assumes symmetric configuration, torque = 0)
    Fl = total_thrust / 2
    Fr = total_thrust / 2

    tau = 0.0  # No extra body torque for now

    # return np.array([Fl, Fr, tau])
    return np.array([0, 0, 0])

# Dynamics function
def dynamics(x, u):
    theta, theta_dot, phi, phi_dot = x
    Fl, Fr, tau = u

    theta_ddot = -g*np.sin(theta) + (1/m) * np.sin(theta-phi)* Fl + (1/m) * np.sin(theta-phi)* Fr - damping*theta_dot
    phi_ddot = -(body_w/2) * Fl + (body_w/2) * Fr  + I * tau

    return np.array([theta_dot, theta_ddot, phi_dot, phi_ddot])

# Pre-simulate trajectory
time = np.arange(0, T, dt)
X = np.zeros((len(time), len(x)))
X[0] = x
for i in range(1, len(time)):
    u = control(X[i-1], time[i-1])
    x_dot = dynamics(X[i-1], u)
    X[i] = X[i-1] + dt * x_dot

# Animation
fig, ax = plt.subplots()
ax.set_aspect('equal')
ax.set_xlim(-1.5, 1.5)
ax.set_ylim(-1.5, 1.5)

line, = ax.plot([], [], 'ko-', lw=2)  # pendulum + body
body_rect = plt.Rectangle((0, 0), body_w, body_h, color='blue', alpha=0.5)
rect_tr1 = None
ax.add_patch(body_rect)

def init():
    line.set_data([], [])
    body_rect.set_xy((-10, -10))  # hide initially
    return line, body_rect

def update(frame):
    theta, _, phi, _ = X[frame]

    # Pendulum end position
    x_p = L * np.sin(theta)
    y_p = -L * np.cos(theta)

    # Body rotation matrix
    rot = np.array([[np.cos(theta+phi), -np.sin(theta+phi)],
                    [np.sin(theta+phi),  np.cos(theta+phi)]])

    # Body corners centered at tip
    corners = np.array([
        [-body_w/2, -body_h/2],
        [ body_w/2, -body_h/2],
        [ body_w/2,  body_h/2],
        [-body_w/2,  body_h/2]
    ])
    rotated = (rot @ corners.T).T + np.array([x_p, y_p])

    # Update pendulum
    line.set_data([0, x_p], [0, y_p])

    # Update body rectangle
    transform = (tr.Affine2D()
             .rotate_around(x_p, y_p, phi+theta)  # rotate around the rectangle's center
             + ax.transData)
    body_rect.set_xy((x_p - body_w / 2, y_p - body_h / 2))
    body_rect.set_transform(transform)
    return line, body_rect

ani = FuncAnimation(fig, update, frames=len(time), init_func=init,
                    blit=True, interval=dt*1000, repeat=False)
plt.title("Pendulum with Actuated Body")
plt.show()

