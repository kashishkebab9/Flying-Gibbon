from config import *
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.transforms as tr
from matplotlib.patches import Rectangle
from pendulum import simulate_pendulum, pendulum_position
from projectile import simulate_projectile
from matplotlib.animation import FFMpegWriter

fig1, ax = plt.subplots(figsize=(10, 8))
ax.set_xlim(-2.5, 10)
ax.set_ylim(-4, 4)
ax.set_aspect('equal')
ax.grid(True)
ax.set_title("Detachable Pendulum with Rotating Rectangle")
ax.set_xlabel("x (m)")
ax.set_ylabel("y (m)")

pivot, = ax.plot(0, 0, 'ko', markersize=8)
pivot2, = ax.plot(8, 0, 'ko', markersize=8)
rod, = ax.plot([], [], '-k', lw=2)
body_rect = Rectangle((0, 0), body_w, body_h, color='gray', alpha=0.7)
ax.add_patch(body_rect)
pendulum_path, = ax.plot([], [], '--', color='gray', alpha=0.7)
projectile_path, = ax.plot([], [], '--', color='gray', alpha=0.7)
time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)
status_text = ax.text(0.02, 0.90, '', transform=ax.transAxes)

t_pend, theta_vals, omega_vals, phi_values, phi_dot_values, t_release, pend_u_opt = simulate_pendulum()
x_pend = [l * np.sin(theta) for theta in theta_vals]
y_pend = [-l * np.cos(theta) for theta in theta_vals]

if t_release is not None:
    release_index = np.argmin(np.abs(t_pend - t_release))
    theta_release_actual = theta_vals[release_index]
    omega_release = omega_vals[release_index]
    phi_release_vel = phi_dot_values[release_index]
    t_proj, x_proj, y_proj, theta_proj, alpha_values, proj_u_opt = simulate_projectile(t_release, theta_release_actual, omega_release, phi_release_vel)
else:
    t_proj, x_proj, y_proj, theta_proj = [], [], [], []


def update(frame):
    if frame < len(x_pend):
        x, y = x_pend[frame], y_pend[frame]
        theta = theta_vals[frame]
        phi = phi_values[frame]
        
        rod.set_data([0, x], [0, y])
        transform = tr.Affine2D().rotate_around(x, y, phi + theta) + ax.transData
        body_rect.set_xy((x - body_w / 2, y - body_h / 2))
        body_rect.set_transform(transform)
        pendulum_path.set_data(x_pend[:frame + 1], y_pend[:frame + 1])
        time_text.set_text(f"time: {frame*dt:.2f} s")
        status_text.set_text("status: attached")
    else:
        proj_frame = frame - len(x_pend)
        if proj_frame < len(x_proj):
            x, y = x_proj[proj_frame], y_proj[proj_frame]
            theta = theta_proj[proj_frame]
            alpha = alpha_values[proj_frame]
            rod.set_data([l * np.sin(theta+ alpha ) + x, x], [-l * np.cos(theta + alpha) + y, y])
            transform = tr.Affine2D().rotate_around(x, y, theta) + ax.transData
            body_rect.set_xy((x - body_w / 2, y - body_h / 2))
            body_rect.set_transform(transform)
            projectile_path.set_data(x_proj[:proj_frame + 1], y_proj[:proj_frame + 1])
            time_text.set_text(f"time: {t_proj[proj_frame]:.2f} s")
            status_text.set_text("status: detached")
    return rod, body_rect, pendulum_path, projectile_path, time_text, status_text

# We need to take "theta" in both pendulum and projectile
orientation_pend = theta_vals + phi_values
fig3 = plt.figure(figsize=(10, 6))
plt.plot(t_pend, orientation_pend, 'r-', label='Orientation (Pendulum)')
plt.xlabel('Time')
plt.ylabel('Angle (rad)')
plt.title('Alpha and Theta vs Time')
plt.legend()
plt.grid(True)


plt.tight_layout()

# Show the plot window without blocking
plt.show(block=False)

# Create the second figure for the plot (before animation starts)
fig2 = plt.figure(figsize=(10, 6))
plt.plot(t_proj, alpha_values, 'r-', label='Alpha (Projectile)')
plt.plot(t_proj, theta_proj, 'g-', label='Theta (Projectile)')
plt.xlabel('Time')
plt.ylabel('Angle (rad)')
plt.title('Alpha and Theta vs Time')
plt.legend()
plt.grid(True)
plt.tight_layout()

# Show the plot window without blocking
plt.show(block=False)

# Now set up and run the animation
writer = FFMpegWriter(fps=int(1 / dt), metadata=dict(artist='Trajectory Opt'), bitrate=1800)
ani = FuncAnimation(fig1, update, frames=len(x_pend) + len(x_proj), interval=1000 / fps, blit=True)

# Save the animation
ani.save("main.mp4", writer=writer)

# Show the animation window - keep both windows open
plt.figure(fig1.number)  # Switch back to animation figure
plt.show()  # This will block and keep both windows open

if pend_u_opt is not None:
    plt.figure()
    plt.plot(t_pend[:-1], pend_u_opt[:, 0], 'r-', label='F1 (Pendulum)')
    plt.plot(t_pend[:-1], pend_u_opt[:, 1], 'b-', label='F2 (Pendulum)')
    plt.plot(t_pend[:-1], pend_u_opt[:, 2], 'g-', label='Tau (Arm)')
    plt.xlabel("Time (s)")
    plt.ylabel("Control Input")
    plt.title("Optimized Control Inputs")
    plt.legend()
    plt.grid(True)
    plt.show()
else:
    print("Warning: pend_u_opt is None — skipping control input plots.")



proj_u_opt_trimmed = proj_u_opt[:599, :]

if proj_u_opt is not None:
    plt.figure()
    plt.plot(t_proj[:-1], proj_u_opt_trimmed[:, 0], 'r-', label='F1 (Pendulum)')
    plt.plot(t_proj[:-1], proj_u_opt_trimmed[:, 1], 'b-', label='F2 (Pendulum)')
    plt.plot(t_proj[:-1], proj_u_opt_trimmed[:, 2], 'g-', label='Tau (Arm)')
    plt.xlabel("Time (s)")
    plt.ylabel("Control Input")
    plt.title("Optimized Control Inputs")
    plt.legend()
    plt.grid(True)
    plt.show()
else:
    print("Warning: pend_u_opt is None — skipping control input plots.")