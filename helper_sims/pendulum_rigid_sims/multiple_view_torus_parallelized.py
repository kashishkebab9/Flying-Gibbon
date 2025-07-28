"""
Reachability‑on‑a‑Torus movie generator (Multi-View)
----------------------------------------------------
• Simulates M particles for N steps (RandUP‑style)
• Saves each frame to ./frames/frame_###.png with 4 views
• Combines frames into reachability.gif
"""

import os, shutil, imageio
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from mpl_toolkits.mplot3d import Axes3D        # registers 3‑D projection
from numba import jit

# ===== Physical parameters =====
g, L = 9.81, 1.0
m, body_w, I = 0.5, 0.30, 2.0
damping = 0.05

# ===== Simulation / RandUP =====
M, N, dt = 10000, 100, 0.05         # particles, steps, timestep
R, r     = 2.0, 1.0                # torus radii

def pendulum_dynamics(x, u):
    θ, θd, ϕ, ϕd = x[:4]
    f1, f2, τ   = u
    sin_diff    = np.sin(θ - ϕ)
    θdd = -g*np.sin(θ) + (1/m)*sin_diff*(f1+f2) - damping*θd
    ϕdd = -(body_w/2)*f1 + (body_w/2)*f2 + I*τ
    return np.array([θd, θdd, ϕd, ϕdd, 0, 0, 0, 0])

def rk4(x, u):
    k1 = pendulum_dynamics(x, u)
    k2 = pendulum_dynamics(x+0.5*dt*k1, u)
    k3 = pendulum_dynamics(x+0.5*dt*k2, u)
    k4 = pendulum_dynamics(x+    dt*k3, u)
    return x + (dt/6)*(k1+2*k2+2*k3+k4)

def torus_embed(theta, phi):
    x = (R + r*np.cos(phi))*np.cos(theta)
    y = (R + r*np.cos(phi))*np.sin(theta)
    z =  r*np.sin(phi)
    return np.column_stack((x, y, z))

# ===== Make / reset frame folder =====
if os.path.exists("frames"):
    shutil.rmtree("frames")
os.makedirs("frames")

# ===== Initialise particles and controls =====
X = np.zeros((M, 8))
X[:, 0] = np.random.uniform(-0.3, 0.3, M)      # θ
X[:, 2] = np.random.uniform(-0.3, 0.3, M)      # ϕ
F1 = np.random.uniform(0.0, 1.0, (M, N))
F2 = np.random.uniform(0.0, 1.0, (M, N))
TA = np.random.uniform(-10.0, 10.0, (M, N))

# ===== Prepare wireframe once (reuse each frame) =====
uu = np.linspace(-np.pi, np.pi, 40)
vv = np.linspace(-np.pi, np.pi, 20)
U, V = np.meshgrid(uu, vv)
W = torus_embed(U.ravel(), V.ravel())

# ===== Main loop – simulate & draw every step =====
for k in range(N):
    # propagate
    U_k = np.stack([F1[:, k], F2[:, k], TA[:, k]], axis=1)
    for j in range(M):
        X[j] = rk4(X[j], U_k[j])
        X[j, 0] = (X[j, 0]+np.pi) % (2*np.pi) - np.pi   # wrap θ
        X[j, 2] = (X[j, 2]+np.pi) % (2*np.pi) - np.pi   # wrap ϕ

    θ, ϕ = X[:, 0], X[:, 2]
    P    = torus_embed(θ, ϕ)

    # convex hull (skip until > 3 non‑collinear points)
    hull_pts = None
    if len(P) >= 4:
        hull = ConvexHull(P)
        hull_pts = P[hull.simplices]

    # ---- plotting from multiple views ----
    # fig = plt.figure(figsize=(12, 9)
    fig = plt.figure(figsize=(14, 10))

    views = [
        ("Isometric", (30, 45)),
        ("Top View (X-Y)", (90, 0)),
        ("Side View (X-Z)", (0, -90)),
        ("Side View (Y-Z)", (0, 0)),
    ]


    for i, (title, (elev, azim)) in enumerate(views):
        ax = fig.add_subplot(2, 2, i+1, projection='3d')
        # ax.set_title(title)

        # Torus wireframe
        ax.plot_wireframe(W[:,0].reshape(U.shape),
                          W[:,1].reshape(U.shape),
                          W[:,2].reshape(U.shape),
                          color='k', alpha=0.1)

        # Particles
        ax.scatter(P[:,0], P[:,1], P[:,2], s=10, c='darkturquoise', alpha=.6)

        # Convex hull
        if hull_pts is not None:
            for tri in hull_pts:
                ax.plot(tri[:,0], tri[:,1], tri[:,2], c='orange', lw=1)

        # Add red axes lines (theta=0 or phi=0)
        if title == "Top View (X-Y)":
            # phi varies, theta = 0
            phi_line = np.linspace(-np.pi, np.pi, 100)
            theta_zero = np.zeros_like(phi_line)
            line_pts = torus_embed(theta_zero, phi_line)
            ax.plot(line_pts[:,0], line_pts[:,1], line_pts[:,2], color='crimson', lw=2)

        elif title == "Side View (X-Z)":
            # theta varies, phi = 0
            theta_line = np.linspace(-np.pi, np.pi, 100)
            phi_zero = np.zeros_like(theta_line)
            line_pts = torus_embed(theta_line, phi_zero)
            ax.plot(line_pts[:,0], line_pts[:,1], line_pts[:,2], color='crimson', lw=2)

        elif title == "Side View (Y-Z)":
            # phi = 0, theta varies
            theta_line = np.linspace(-np.pi, np.pi, 100)
            phi_zero = np.zeros_like(theta_line)
            line_pts = torus_embed(theta_line, phi_zero)
            ax.plot(line_pts[:,0], line_pts[:,1], line_pts[:,2], color='crimson', lw=2)

        ax.set_axis_off()
        ax.view_init(elev=elev, azim=azim)

        # ax.set_axis_off()
        # ax.view_init(elev=elev, azim=azim)

    fig.suptitle(f"Time: {dt*k:.2f}", fontsize=14)
    plt.tight_layout(pad=0.2, h_pad=0.2, w_pad=0.2)
    # plt.subplots_adjust(wspace=0.01, hspace=0.01)
    # plt.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01, wspace=0.01, hspace=0.01)
    # plt.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
    plt.savefig(f"frames/frame_{k:03d}.png", dpi=300)
    plt.close(fig)

print("Frames saved → ./frames")

# ===== Make GIF =====
images = [imageio.imread(f"frames/frame_{k:03d}.png") for k in range(N)]
imageio.mimsave("reachability.gif", images, fps=10)
print("GIF saved as reachability.gif")
