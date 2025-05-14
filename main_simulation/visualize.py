from config import *
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.transforms as tr
from matplotlib.patches import Rectangle
from pendulum import simulate_pendulum, pendulum_position
from projectile import simulate_projectile

fig, ax = plt.subplots(figsize=(10, 8))
ax.set_xlim(-2.5, 10)
ax.set_ylim(-4, 6)
ax.set_aspect('equal')
ax.grid(True)
ax.set_title("Detachable Pendulum with Rotating Rectangle")
ax.set_xlabel("x (m)")
ax.set_ylabel("y (m)")

pivot, = ax.plot(0, 0, 'ko', markersize=8)
rod, = ax.plot([], [], '-k', lw=2)
body_rect = Rectangle((0, 0), body_w, body_h, color='blue', alpha=0.7)
ax.add_patch(body_rect)
pendulum_path, = ax.plot([], [], '--', color='gray', alpha=0.7)
projectile_path, = ax.plot([], [], '--', color='blue', alpha=0.7)
time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)
status_text = ax.text(0.02, 0.90, '', transform=ax.transAxes)

t_pend, theta_vals, omega_vals, t_release = simulate_pendulum()
x_pend = [l * np.sin(theta) for theta in theta_vals]
y_pend = [-l * np.cos(theta) for theta in theta_vals]

if t_release is not None:
    release_index = t_pend.index(t_release)
    theta_release_actual = theta_vals[release_index]
    omega_release = omega_vals[release_index]
    t_proj, x_proj, y_proj, theta_proj = simulate_projectile(t_release, theta_release_actual, omega_release)
else:
    t_proj, x_proj, y_proj, theta_proj = [], [], [], []

def update(frame):
    if frame < len(x_pend):
        x, y = x_pend[frame], y_pend[frame]
        theta = theta_vals[frame]
        rod.set_data([0, x], [0, y])
        transform = tr.Affine2D().rotate_around(x, y, theta) + ax.transData
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
            rod.set_data([-l * np.sin(theta) + x, x], [l * np.cos(theta) + y, y])
            transform = tr.Affine2D().rotate_around(x, y, theta) + ax.transData
            body_rect.set_xy((x - body_w / 2, y - body_h / 2))
            body_rect.set_transform(transform)
            projectile_path.set_data(x_proj[:proj_frame + 1], y_proj[:proj_frame + 1])
            time_text.set_text(f"time: {t_proj[proj_frame]:.2f} s")
            status_text.set_text("status: detached")
    return rod, body_rect, pendulum_path, projectile_path, time_text, status_text

ani = FuncAnimation(fig, update, frames=len(x_pend) + len(x_proj), interval=1000 / fps, blit=True)
plt.tight_layout()
plt.show()