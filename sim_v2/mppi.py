
import numpy as np
import matplotlib.pyplot as plt

body_w = 0.5
I = 2.0
m = 1.0
damping = 0.1
arm_I = 0.0333
L = 1.0

def pendulum_dynamics(x, u, t, g=9.81, L=1.0):
    theta, theta_dot, phi, phi_dot = x[0], x[1], x[2], x[3]
    f1_k, f2_k, tau_k = u[0], u[1], u[2]
    sin_diff = np.sin(theta - phi)
    theta_ddot = -g * np.sin(theta) + (1 / m) * sin_diff * (f1_k + f2_k) - damping * theta_dot
    phi_ddot = -(body_w / 2) * f1_k + (body_w / 2) * f2_k + I * tau_k
    return np.array([theta_dot, theta_ddot, phi_dot, phi_ddot, 0, 0, 0, 0])

def projectile_dynamics(x, u, t, g=9.81, L=1.0):
    x, y, theta, x_dot, y_dot, theta_dot, alpha, alpha_dot = x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7]
    f1_k, f2_k, tau_k = u[0], u[1], u[2]
    total_f = f1_k + f2_k
    x_ddot = -total_f * np.sin(theta)/m
    y_ddot = total_f * np.cos(theta)/m - g
    torque = (f2_k - f1_k) * body_w
    theta_ddot = torque / I 
    alpha_ddot = tau_k / arm_I
    return np.array([x_dot, y_dot, theta_dot, x_ddot, y_ddot, theta_ddot, alpha_dot, alpha_ddot])

def rk4_step(f, y, t, dt):
    u = [0,0,0]
    k1 = f(y, u, t)
    k2 = f(y + 0.5 * dt * k1, u, t + 0.5 * dt)
    k3 = f(y + 0.5 * dt * k2, u, t + 0.5 * dt)
    k4 = f(y + dt * k3, u, t + dt)
    return y + (dt / 6) * (k1 + 2*k2 + 2*k3 + k4)

# Initial conditions
def simulate_hybrid_system(dt=.01):
    y0 = [np.pi / 4, 0.0, 0.5, 0.0, 0, 0, 0, 0] # theta, theta_dot, phi, phi_dot
    t0 = 0.0
    tf = 2.0
    t_vals = np.arange(t0, tf, dt) # 0, 0.01, 0.02 ... tf
    y_vals = []

    y = np.array(y0)

    attached = True
    detach_time = None
    for t in t_vals:
        y_vals.append(y)
        if attached:
            y = rk4_step(pendulum_dynamics, y, t, dt)

            if y[1] < -1.0:
                print("Transitioning at t =", t)
                theta, theta_dot, phi, phi_dot = y[0], y[1], y[2], y[3]
                x_pos = L * np.sin(theta)
                y_pos = -L * np.cos(theta)
                x_dot = L * theta_dot * np.cos(theta)
                y_dot = L * theta_dot * np.sin(theta)
                alpha = phi
                alpha_dot = phi_dot
                y = np.array([x_pos, y_pos, theta, x_dot, y_dot, theta_dot, alpha, alpha_dot])
                attached = False
                detach_time = t
        else:
            y = rk4_step(projectile_dynamics, y, t, dt)
        

    y_vals = np.array(y_vals)
    return y_vals, detach_time

    # # Plot
    # plt.plot(t_vals, y_vals[:, 1])
    # plt.xlabel('Time [s]')
    # plt.ylabel('Theta [rad]')
    # plt.title('Simple Pendulum using RK4')
    # plt.grid()
    # plt.show()
