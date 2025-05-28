import yaml
import matplotlib.pyplot as plt
import sys
from pendulum import simulate_pendulum
import numpy as np
from matplotlib.patches import Rectangle
import matplotlib.transforms as tr
from matplotlib.animation import FuncAnimation

def visualize_simulation(filename, traj_output=None):
    # Load the YAML file
    with open(filename, "r") as file:
        config = yaml.safe_load(file)

    dt =.01

    # Parsing the config:
    config_x_min = config["visualization_parameters"]["x_min"]
    config_x_max = config["visualization_parameters"]["x_max"]
    config_y_min = config["visualization_parameters"]["y_min"]
    config_y_max = config["visualization_parameters"]["y_max"]
    config_l = config["physical_parameters"]["pendulum_length"]
    config_body_w = config["physical_parameters"]["body_width"]
    config_body_h = config["physical_parameters"]["body_height"]

    config_hinge_list = config["physical_parameters"]["pendulum_hinge_array"]

    fig1, ax = plt.subplots(figsize=(10, 8))
    ax.set_xlim(config_x_min, config_x_max)
    ax.set_ylim(config_y_min, config_y_max)
    ax.set_aspect('equal')
    ax.grid(True)
    ax.set_title("Detachable Pendulum with Rotating Rectangle")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")

    # Add the hinges to the graphs
    for hinge in config_hinge_list:
        if len(hinge) != 2:
            print("Config Parsing Error: # of hinge elements != 2")
            sys.exit()
            
        pivot, = ax.plot(hinge[0], hinge[1], 'ko', markersize=8)

    rod, = ax.plot([], [], '-k', lw=2)
    body_rect = Rectangle((0, 0), config_body_w, config_body_h, color='gray', alpha=0.7)
    ax.add_patch(body_rect)
    pendulum_path, = ax.plot([], [], '--', color='gray', alpha=0.7)
    projectile_path, = ax.plot([], [], '--', color='gray', alpha=0.7)
    time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)
    status_text = ax.text(0.02, 0.90, '', transform=ax.transAxes)
    t_pend, theta_vals, omega_vals, phi_values, phi_dot_values, t_release, pend_u_opt = simulate_pendulum()
    x_pend = [config_l * np.sin(theta) for theta in theta_vals]
    y_pend = [-config_l * np.cos(theta) for theta in theta_vals]

    def update(frame):
        if frame < len(x_pend):
            x, y = x_pend[frame], y_pend[frame]
            theta = theta_vals[frame]
            phi = phi_values[frame]
            
            rod.set_data([0, x], [0, y])
            transform = tr.Affine2D().rotate_around(x, y, phi + theta) + ax.transData
            body_rect.set_xy((x - config_body_w / 2, y - config_body_h / 2))
            body_rect.set_transform(transform)
            pendulum_path.set_data(x_pend[:frame + 1], y_pend[:frame + 1])
            time_text.set_text(f"time: {frame*dt:.2f} s")
            status_text.set_text("status: attached")
        return rod, body_rect, pendulum_path, projectile_path, time_text, status_text

    ani = FuncAnimation(fig1, update, frames=len(x_pend), interval=1000*dt, blit=True)
    plt.figure(fig1.number)
    plt.show()


if __name__=="__main__":
    visualize_simulation("config.yaml")