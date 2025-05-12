
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle

# physical parameters
g = 9.81  # acceleration due to gravity (m/s^2)
l = 2.0   # length of pendulum (m)
m = 100.0   # mass of pendulum bob (kg)
theta_0 = -3*np.pi/4  # initial angle (60 degrees)
theta_release = .7  # angle at which pendulum detaches (30 degrees)

# simulation parameters
fps = 60
dt = 1/fps
total_time = 3.0  # total simulation time (s)
frames = int(total_time / dt)

# calculate initial conditions
omega_0 = 0.0  # initial angular velocity

# arrays to store pendulum positions
pendulum_x = []
pendulum_y = []
projectile_x = []
projectile_y = []

# time of release (to be calculated)
t_release = None

# pendulum position function
def pendulum_position(theta):
    x = l * np.sin(theta)
    y = -l * np.cos(theta)
    return x, y

# initialize figure and axis
fig, ax = plt.subplots(figsize=(10, 8))
ax.set_xlim(-2.5, 10)
ax.set_ylim(-4, 6)
ax.set_aspect('equal')
ax.grid(True)
ax.set_title("detachable pendulum simulation", fontsize=14)
ax.set_xlabel("x position (m)")
ax.set_ylabel("y position (m)")

# plot pendulum components
pivot, = ax.plot(0, 0, 'ko', markersize=8)  # pivot point
rod, = ax.plot([], [], '-k', lw=2)  # pendulum rod
bob = Circle((0, 0), 0.1, fc='r', zorder=3)  # pendulum bob
ax.add_patch(bob)

# trajectory lines
pendulum_path, = ax.plot([], [], '--', color='gray', alpha=0.7)
projectile_path, = ax.plot([], [], '--', color='blue', alpha=0.7)

# text annotations
time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)
status_text = ax.text(0.02, 0.90, '', transform=ax.transAxes)

# numerical simulation for pendulum motion
def simulate_pendulum():
    global t_release
    
    # initial conditions
    theta = theta_0
    omega = omega_0
    
    # arrays for storage
    t_values = [0]
    theta_values = [theta]
    omega_values = [omega]
    
    t = 0
    released = False
    
    while t < total_time and not released:
        # simple euler integration for pendulum motion
        # d²θ/dt² + (g/l)sin(θ) = 0
        alpha = -g/l * np.sin(theta)  # angular acceleration
        
        # update angular velocity and position
        omega += alpha * dt
        theta += omega * dt
        
        # store values
        t += dt
        t_values.append(t)
        theta_values.append(theta)
        omega_values.append(omega)
        
        # check for release condition
        if theta_0 > 0:  # started with positive angle
            if theta < theta_release:
                released = True
                t_release = t
        else:  # started with negative angle
            if theta > theta_release:
                released = True
                t_release = t
    
    return t_values, theta_values, omega_values

# numerical simulation for projectile motion
def simulate_projectile(t_release, theta_release, omega_release):
    # pendulum velocity components at release
    v_tangential = l * omega_release
    v_x = v_tangential * np.cos(theta_release)
    v_y = v_tangential * np.sin(theta_release)
    
    # initial position at release
    x_release, y_release = pendulum_position(theta_release)
    
    # arrays for storage
    t_values = []
    x_values = []
    y_values = []
    
    t = t_release
    while t < total_time:
        # time since release
        dt_since_release = t - t_release
        
        # projectile equations
        x = x_release + v_x * dt_since_release
        y = y_release + v_y * dt_since_release - 0.5 * g * dt_since_release**2
        
        # store values
        t_values.append(t)
        x_values.append(x)
        y_values.append(y)
        
        t += dt
        
        # optional: stop if projectile hits ground
        if y < -1.5*l:
            break
    
    return t_values, x_values, y_values

# run simulations
t_pend, theta_vals, omega_vals = simulate_pendulum()
x_pend = [l * np.sin(theta) for theta in theta_vals]
y_pend = [-l * np.cos(theta) for theta in theta_vals]

if t_release is not None:
    release_index = t_pend.index(t_release)
    theta_release_actual = theta_vals[release_index]
    omega_release = omega_vals[release_index]
    t_proj, x_proj, y_proj = simulate_projectile(t_release, theta_release_actual, omega_release)
else:
    t_proj, x_proj, y_proj = [], [], []

# animation update function
def update(frame):
    if frame < len(x_pend):
        # update pendulum
        x = x_pend[frame]
        y = y_pend[frame]
        rod.set_data([0, x], [0, y])
        bob.center = (x, y)
        
        # update pendulum path
        pendulum_path.set_data(x_pend[:frame+1], y_pend[:frame+1])
        
        # update time text
        time_text.set_text(f'time: {frame*dt:.2f} s')
        status_text.set_text('status: attached')
        
    else:
        # after pendulum release
        proj_frame = frame - len(x_pend)
        if proj_frame < len(x_proj):
            # update projectile
            x = x_proj[proj_frame]
            y = y_proj[proj_frame]
            rod.set_data([], [])  # hide rod
            bob.center = (x, y)
            
            # update projectile path
            projectile_path.set_data(x_proj[:proj_frame+1], y_proj[:proj_frame+1])
            
            # update time text
            time_text.set_text(f'time: {t_proj[proj_frame]:.2f} s')
            status_text.set_text('status: detached')
    
    return rod, bob, pendulum_path, projectile_path, time_text, status_text

# create animation
ani = FuncAnimation(fig, update, frames=len(x_pend)+len(x_proj), 
                    interval=1000/fps, blit=True)

# display the initial frame
plt.tight_layout()
plt.show()

# to save the animation, uncomment the line below:
# ani.save('detachable_pendulum.mp4', fps=fps, dpi=100, writer='ffmpeg')