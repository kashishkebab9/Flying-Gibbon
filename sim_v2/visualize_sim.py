import yaml
import matplotlib.pyplot as plt
import sys
from pendulum import simulate_pendulum
from projectile import simulate_projectile
import numpy as np
from matplotlib.patches import Rectangle
import matplotlib.transforms as tr
from matplotlib.animation import FuncAnimation
from matplotlib.animation import FFMpegWriter
from mppi import simulate_hybrid_system

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
    config_controller_type = config["controller"]["type"]

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

    output_states = None
    detach_time = None

    theta_pend = None
    phi_pend = None
    x_pend = None
    y_pend = None

    x_flight = None
    y_flight = None
    theta_flight = None
    alpha_flight = None

    if config_controller_type == 0:
        t_pend, theta_vals, omega_vals, phi_values, phi_dot_values, t_release, pend_u_opt = simulate_pendulum()
        x_pend = [config_l * np.sin(theta) for theta in theta_vals]
        y_pend = [-config_l * np.cos(theta) for theta in theta_vals]

        if t_release is not None:
            release_index = np.argmin(np.abs(t_pend - t_release))
            theta_release_actual = theta_vals[release_index]
            omega_release = omega_vals[release_index]
            phi_release_vel = phi_dot_values[release_index]
            t_proj, x_proj, y_proj, theta_proj, alpha_values, proj_u_opt, proj_t_opt = simulate_projectile(t_release)
        else:
            t_proj, x_proj, y_proj, theta_proj = [], [], [], []

        release_index = np.argmin(np.abs(t_pend - t_release)) 
        new_t_pend = t_pend[:release_index+1]

        
        control_output = np.vstack((pend_u_opt, proj_u_opt))
        time_output = np.concatenate((new_t_pend, t_proj))

        min_len = min(control_output.shape[0], time_output.shape[0])
        control_output = control_output[:min_len]
        time_output = time_output[:min_len]

    elif config_controller_type == 1:
        output_states, detach_time = simulate_hybrid_system(dt)
        switch_idx = int(detach_time/dt)
        pendulum_states = output_states[:switch_idx]
        flight_states = output_states[switch_idx:]
        # pendulum state conversion:
        theta_pend = pendulum_states[:, 0]
        phi_pend = pendulum_states[:, 2]
        x_pend = [config_l * np.sin(theta) for theta in theta_pend]
        y_pend = [-config_l * np.cos(theta) for theta in theta_pend]

        # flight states_conversion
        x_flight = flight_states[:, 0] 
        y_flight = flight_states[:, 1] 
        theta_flight = flight_states[:, 2] 
        alpha_flight = flight_states[:, 3] 

    def update(frame):
        if config_controller_type == 0:
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
            else:
                proj_frame = frame - len(x_pend)
                if proj_frame < len(x_proj):
                    x, y = x_proj[proj_frame], y_proj[proj_frame]
                    theta = theta_proj[proj_frame]
                    alpha = alpha_flight[proj_frame]
                    rod.set_data([config_l * np.sin(theta+ alpha) + x, x], [-config_l * np.cos(theta + alpha) + y, y])
                    transform = tr.Affine2D().rotate_around(x, y, theta) + ax.transData
                    body_rect.set_xy((x - config_body_w / 2, y - config_body_h / 2))
                    body_rect.set_transform(transform)
                    projectile_path.set_data(x_proj[:proj_frame + 1], y_proj[:proj_frame + 1])
                    time_text.set_text(f"time: {t_proj[proj_frame]:.2f} s")
                    status_text.set_text("status: detached")
            return rod, body_rect, pendulum_path, projectile_path, time_text, status_text

        if config_controller_type == 1 and output_states is not None:

            if frame < switch_idx:
                x, y = x_pend[frame], y_pend[frame]
                theta = theta_pend[frame]
                phi = phi_pend[frame]
                
                rod.set_data([0, x], [0, y])
                transform = tr.Affine2D().rotate_around(x, y, phi + theta) + ax.transData
                body_rect.set_xy((x - config_body_w / 2, y - config_body_h / 2))
                body_rect.set_transform(transform)
                pendulum_path.set_data(x_pend[:frame + 1], y_pend[:frame + 1])
                # time_text.set_text(f"time: {frame*dt:.2f} s")
                status_text.set_text("status: attached")
            else:
                flight_frame = frame - switch_idx
                x, y = x_flight[flight_frame], y_flight[flight_frame]
                theta = theta_flight[flight_frame]
                alpha = alpha_flight[flight_frame]

                rod.set_data([config_l * np.sin(theta+ alpha) + x, x], [-config_l * np.cos(theta + alpha) + y, y])
                transform = tr.Affine2D().rotate_around(x, y, theta) + ax.transData
                body_rect.set_xy((x - config_body_w / 2, y - config_body_h / 2))
                body_rect.set_transform(transform)
                projectile_path.set_data(x_flight[:flight_frame + 1], y_flight[:flight_frame + 1])
                # time_text.set_text(f"time: {t_proj[flight_frame]:.2f} s")
                status_text.set_text("status: detached")
            return rod, body_rect, pendulum_path, projectile_path, time_text, status_text




                

    if config_controller_type == 0:
        writer = FFMpegWriter(fps=int(1 / dt), metadata=dict(artist='Trajectory Opt'), bitrate=1800)
        ani = FuncAnimation(fig1, update, frames=len(x_pend)+len(x_proj), interval=1000*dt, blit=True)
        ani.save("main.mp4", writer=writer)

        # Plot control inputs over time
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        labels = ["Left_Rotor", "Right_Rotor", "Torque_Arm"]
        for i in range(control_output.shape[1]):
            ax2.plot(time_output, control_output[:, i], label=labels[i])

        ax2.set_title("Control Inputs vs Time")
        ax2.set_xlabel("Time (s)")
        ax2.set_ylabel("Control Inputs")
        ax2.grid(True)
        ax2.legend()

        plt.figure(fig1.number)
        plt.show()
    elif config_controller_type == 1:
        ani = FuncAnimation(fig1, update, frames=len(output_states), interval=1000*dt, blit=True)
        plt.figure(fig1.number)
        plt.show()


if __name__=="__main__":
    visualize_simulation("config.yaml")