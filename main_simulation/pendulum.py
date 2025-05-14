from config import *

def pendulum_position(theta):
    x = l * np.sin(theta)
    y = -l * np.cos(theta)
    return x, y

def pendulum_control(x, t):
    theta, theta_dot, phi, phi_dot = x
    Kp = 2.0
    Kd = 1.0
    theta_des = theta_release
    theta_dot_des = theta_release_vel
    theta_ddot_des = -Kp * (theta - theta_des) - Kd * (theta_dot - theta_dot_des)
    sin_diff = np.sin(theta - phi)
    if abs(sin_diff) < 1e-3:
        sin_diff = np.sign(sin_diff) * 1e-3
    total_thrust = (m / sin_diff) * (theta_ddot_des + g * np.sin(theta))
    Fl = total_thrust / 2
    Fr = total_thrust / 2
    tau = 0.0
    return np.array([Fl, Fr, tau])

def simulate_pendulum():
    theta, d_theta = theta_0, 0.0
    phi, d_phi = phi_0, phi_dot_0
    t_values, theta_values, omega_values = [0], [theta], [d_theta]
    t, released = 0, False
    x = x0.copy()
    t_release = None

    while t < total_time and not released:
        u = pendulum_control(x, t)
        Fl, Fr, tau = u
        dd_theta = -g / l * np.sin(theta) + (1/m) * np.sin(theta - phi) * (Fl + Fr)
        d_theta += dd_theta * dt
        theta += d_theta * dt
        dd_phi = -(body_w / 2) * Fl + (body_w / 2) * Fr + I * tau
        d_phi += dd_phi * dt
        phi += d_phi * dt
        t += dt
        t_values.append(t)
        theta_values.append(theta)
        omega_values.append(d_theta)
        if theta_0 > 0 and theta < theta_release or theta_0 < 0 and theta > theta_release:
            released = True
            t_release = t
    return t_values, theta_values, omega_values, t_release
