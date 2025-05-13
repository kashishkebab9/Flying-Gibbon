import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.transforms as tr
from matplotlib.patches import Rectangle

# Physical Parameters
g = 9.81  # gravity
l = 1.0   # pendulum length
m = 100.0  # pendulum mass
theta_0 = -3 * np.pi / 4  # initial angle
theta_release = 0.7       # angle to detach
body_w = .5
body_h = 0.2

I = 2.0
# Simulation Parameters
fps = 60
dt = 1 / fps
total_time = 3.0
frames = int(total_time / dt)

# Arrays for Storing Trajectories
pendulum_x, pendulum_y = [], []
projectile_x, projectile_y = [], []

# Release time
t_release = None

phi_0 = 0.0          # initial body angle
phi_dot_0 = 0.0      # initial body angular velocity
x0 = np.array([theta_0, 0.0, phi_0, phi_dot_0])

def pendulum_position(theta):
    x = l * np.sin(theta)
    y = -l * np.cos(theta)
    return x, y

# Plot Setup
fig, ax = plt.subplots(figsize=(10, 8))
ax.set_xlim(-2.5, 10)
ax.set_ylim(-4, 6)
ax.set_aspect('equal')
ax.grid(True)
ax.set_title("Detachable Pendulum with Rotating Rectangle", fontsize=14)
ax.set_xlabel("x position (m)")
ax.set_ylabel("y position (m)")

pivot, = ax.plot(0, 0, 'ko', markersize=8)
pivot, = ax.plot(8, 0, 'ko', markersize=8)
rod, = ax.plot([], [], '-k', lw=2)

# Replace bob with rectangle
body_rect = Rectangle((0, 0), body_w, body_h, color='blue', alpha=0.7)
ax.add_patch(body_rect)

# Trajectory paths
pendulum_path, = ax.plot([], [], '--', color='gray', alpha=0.7)
projectile_path, = ax.plot([], [], '--', color='blue', alpha=0.7)

# Text
time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)
status_text = ax.text(0.02, 0.90, '', transform=ax.transAxes)

def pendulum_control(x, t):
    theta, theta_dot, phi, phi_dot = x

    # Desired behavior
    theta_des = theta_release
    theta_dot_des = 10.0

    # PD gains
    Kp = 2.0
    Kd = 1.0

    # Desired theta acceleration
    theta_ddot_des = -Kp * (theta - theta_des) - Kd * (theta_dot - theta_dot_des)

    # Cancel dynamics to solve for Fl + Fr
    sin_diff = np.sin(theta - phi)
    if abs(sin_diff) < 1e-3:
        sin_diff = np.sign(sin_diff) * 1e-3  # avoid divide by zero

    # total_thrust = (m / sin_diff) * (theta_ddot_des + g * np.sin(theta) + damping * theta_dot)
    total_thrust = (m / sin_diff) * (theta_ddot_des + g * np.sin(theta))

    # Split thrusts evenly (assumes symmetric configuration, torque = 0)
    Fl = total_thrust / 2
    Fr = total_thrust / 2

    tau = 0.0  # No extra body torque for now

    # return np.array([0, 0, 0])
    return np.array([Fl, Fr, tau])


# Simulate Pendulum 
def simulate_pendulum():
    global t_release
    theta, d_theta = theta_0, 0.0
    phi, d_phi = phi_0, phi_dot_0
    t_values, theta_values, omega_values = [0], [theta], [d_theta]
    t, released = 0, False
    x = x0.copy()

    while t < total_time and not released:
        
        u = pendulum_control(x, t)
        Fl = u[0]
        Fr = u[1]
        tau = u[2]
        dd_theta = -g / l * np.sin(theta)+ (1/m) * np.sin(theta-phi)* Fl + (1/m) * np.sin(theta-phi)* Fr 

        d_theta += dd_theta * dt
        theta += d_theta * dt

        dd_phi = -(body_w/2) * Fl + (body_w/2) * Fr  + I * tau
        d_phi += dd_phi * dt
        phi += d_phi*dt

        t += dt
        t_values.append(t)
        theta_values.append(theta)
        omega_values.append(d_theta)

        if theta_0 > 0 and theta < theta_release or theta_0 < 0 and theta > theta_release:
            released = True
            t_release = t

    return t_values, theta_values, omega_values

# Simulate Projectile 
def simulate_projectile(t_release, theta_release, omega_release):
    v_tan = l * omega_release
    v_x = v_tan * np.cos(theta_release)
    v_y = v_tan * np.sin(theta_release)

    x_release, y_release = pendulum_position(theta_release)

    t_values, x_values, y_values, theta_values = [], [], [], []
    t = t_release
    theta = theta_release

    while t < total_time:
        dt_rel = t - t_release
        x = x_release + v_x * dt_rel
        y = y_release + v_y * dt_rel - 0.5 * g * dt_rel**2
        if y < -1.5 * l:
            break
        # Rotates at constant angular velocity after release
        theta = theta_release + omega_release * dt_rel

        t_values.append(t)
        x_values.append(x)
        y_values.append(y)
        theta_values.append(theta)
        t += dt

    return t_values, x_values, y_values, theta_values

# Run Simulation 
t_pend, theta_vals, omega_vals = simulate_pendulum()
x_pend = [l * np.sin(theta) for theta in theta_vals]
y_pend = [-l * np.cos(theta) for theta in theta_vals]

if t_release is not None:
    release_index = t_pend.index(t_release)
    theta_release_actual = theta_vals[release_index]
    omega_release = omega_vals[release_index]
    t_proj, x_proj, y_proj, theta_proj = simulate_projectile(t_release, theta_release_actual, omega_release)

else:
    t_proj, x_proj, y_proj = [], [], []

# Animation Function 
def update(frame):
    if frame < len(x_pend):
        x = x_pend[frame]
        y = y_pend[frame]
        theta = theta_vals[frame]

        # Update pendulum rod
        rod.set_data([0, x], [0, y])

        # Rotate and position rectangle
        transform = (tr.Affine2D()
                     .rotate_around(x, y, theta)
                     + ax.transData)
        body_rect.set_xy((x - body_w / 2, y - body_h / 2))
        body_rect.set_transform(transform)

        pendulum_path.set_data(x_pend[:frame + 1], y_pend[:frame + 1])
        time_text.set_text(f"time: {frame*dt:.2f} s")
        status_text.set_text("status: attached")
    else:
        proj_frame = frame - len(x_pend)
        if proj_frame < len(x_proj):
            x = x_proj[proj_frame]
            y = y_proj[proj_frame]
            theta = theta_proj[proj_frame]

            rod.set_data([-l * np.sin(theta) + x, x], [l * np.cos(theta) + y, y])
            transform = (tr.Affine2D()
                        .rotate_around(x, y, theta)
                        + ax.transData)

            body_rect.set_xy((x - body_w / 2, y - body_h / 2))
            body_rect.set_transform(transform)

            projectile_path.set_data(x_proj[:proj_frame + 1], y_proj[:proj_frame + 1])
            time_text.set_text(f"time: {t_proj[proj_frame]:.2f} s")
            status_text.set_text("status: detached")
    return rod, body_rect, pendulum_path, projectile_path, time_text, status_text

# Animate 
ani = FuncAnimation(fig, update, frames=len(x_pend) + len(x_proj),
                    interval=1000 / fps, blit=True)

plt.tight_layout()
plt.show()

# To save:
# ani.save('pendulum_with_rectangle.mp4', fps=fps, dpi=100, writer='ffmpeg')
