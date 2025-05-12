
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.animation import FuncAnimation
import matplotlib.transforms as transforms

# Physical parameters
g = 9.81             # gravity (m/s^2)
body_w = 0.5         # width of the rectangle (m)
body_h = 0.2         # height of the rectangle (m)
m = 1.0              # mass (kg)
I = (1/12) * m * (body_w**2 + body_h**2)  # moment of inertia
omega0 = 2.0         # initial angular velocity (rad/s)

# Initial state
x0, y0 = 0.0, 3.0             # initial COM position
vx0, vy0 = 1.0, 0.0           # initial COM velocity
theta0 = 0.0                  # initial orientation
state = np.array([x0, y0, vx0, vy0, theta0, omega0])

# Time parameters
dt = 0.02
T = 5
frames = int(T / dt)

# Dynamics update
def update_state(state, dt):
    x, y, vx, vy, theta, omega = state
    x += vx * dt
    y += vy * dt
    vx += 0
    vy += -g * dt
    theta += omega * dt
    return np.array([x, y, vx, vy, theta, omega])

# Setup figure
fig, ax = plt.subplots(figsize=(6, 6))
ax.set_xlim(-2, 5)
ax.set_ylim(-1, 5)
ax.set_aspect('equal')
ax.set_title("Free Falling Rotating Rectangle")

# Rectangle patch
rect = Rectangle((0, 0), body_w, body_h, color='skyblue')
ax.add_patch(rect)

# Update function
def animate(i):
    global state
    state = update_state(state, dt)
    x, y, _, _, theta, _ = state

    # Position the rectangle's lower-left corner before transforming
    rect.set_xy((x - body_w/2, y - body_h/2))

    # Apply rotation around center
    trans = transforms.Affine2D().rotate_around(x, y, theta) + ax.transData
    rect.set_transform(trans)

    return [rect]

# Animate
ani = FuncAnimation(fig, animate, frames=frames, interval=dt*1000, blit=True)
plt.tight_layout()
plt.show()
