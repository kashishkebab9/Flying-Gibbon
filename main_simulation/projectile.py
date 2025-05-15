from config import *
import casadi as cs

def pendulum_position(theta):
    x = l * np.sin(theta)
    y = -l * np.cos(theta)
    return x, y

#def flight_control(initial_state, target_position, N=1000):


def simulate_projectile(t_release, theta_release, omega_release, phi_release_vel):
    # initialization 
    # TODO: we need to modify this to go from x_pend -> x_flight
    x_release, y_release = pendulum_position(theta_release)
    v_tan = l * omega_release
    v_x = v_tan * np.cos(theta_release)
    v_y = v_tan * np.sin(theta_release)
    theta = xf[0] + xf[2]
    t_values, x_values, y_values, theta_values, alpha_values = [], [], [], [], []
    t = t_release

    pendulum_point = (0,0)
    dx = x_release - pendulum_point[0]
    dy = y_release - pendulum_point[1]
    angle_rad = np.arctan2(dy, dx)
    final_arm_ang = np.pi/2 + theta - angle_rad


    while t < total_time:
        dt_rel = t - t_release
        x = x_release + v_x * dt_rel
        y = y_release + v_y * dt_rel - 0.5 * g * dt_rel**2
        if y < -1.5 * l:
            break

        # Rotates at constant angular velocity after release
        theta = theta + phi_release_vel * dt_rel

        t_values.append(t)
        x_values.append(x)
        y_values.append(y)
        alpha_values.append(final_arm_ang)
        theta_values.append(theta)
        t += dt

    return t_values, x_values, y_values, theta_values, alpha_values