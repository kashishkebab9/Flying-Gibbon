# ==========================================
# === Single Poster Figure Visualization ===
# ==========================================
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.transforms as tr

fig, ax = plt.subplots(figsize=(6,6))
ax.set_aspect('equal')
ax.set_xlim(-1.5, 1.5)
ax.set_ylim(-1.5, 1.5)
ax.set_title("Optimized Swing-Up Trajectory")

num_frames = 20  # number of faded snapshots to plot
indices = np.linspace(0, len(X_opt)-1, num_frames).astype(int)

for i, idx in enumerate(indices):
    theta, _, phi, _ = X_opt[idx]

    # Compute pendulum tip coordinates
    x_p = L * np.sin(theta)
    y_p = -L * np.cos(theta)

    # Rotation matrix for the body
    rot = np.array([
        [np.cos(theta+phi), -np.sin(theta+phi)],
        [np.sin(theta+phi),  np.cos(theta+phi)]
    ])

    # Rectangle corners in body frame
    corners = np.array([
        [-body_w/2, -body_h/2],
        [ body_w/2, -body_h/2],
        [ body_w/2,  body_h/2],
        [-body_w/2,  body_h/2]
    ])
    rotated = (rot @ corners.T).T + np.array([x_p, y_p])

    # Set alpha so later poses are more visible
    alpha = (i+1) / num_frames

    # Draw pendulum line
    ax.plot([0, x_p], [0, y_p], color='black', alpha=alpha)

    # Draw body polygon
    ax.fill(rotated[:,0], rotated[:,1], color='blue', alpha=0.15 + 0.8*alpha)

# Highlight final pose (thicker + higher opacity)
theta_f, _, phi_f, _ = X_opt[-1]
x_f = L*np.sin(theta_f)
y_f = -L*np.cos(theta_f)

ax.plot([0,x_f],[0,y_f],'k-',linewidth=3)
rot_f = np.array([
        [np.cos(theta_f+phi_f),-np.sin(theta_f+phi_f)],
        [np.sin(theta_f+phi_f), np.cos(theta_f+phi_f)]
])
corners_f = np.array([
    [-body_w/2, -body_h/2],
    [ body_w/2, -body_h/2],
    [ body_w/2,  body_h/2],
    [-body_w/2,  body_h/2]
])
rotated_f = (rot_f @ corners_f.T).T + np.array([x_f, y_f])
ax.fill(rotated_f[:,0], rotated_f[:,1], color='blue', alpha=1.0)

plt.xlabel("X (m)")
plt.ylabel("Y (m)")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()
