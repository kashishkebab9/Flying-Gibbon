
"""
Reachability‑on‑a‑Torus movie generator
--------------------------------------
• Simulates M particles for N steps (RandUP‑style)
• Saves each frame to ./frames/frame_###.png
• Combines frames into reachability.gif
"""

# returns a convex hull approximation of the true reachable set

import os, shutil, imageio
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from mpl_toolkits.mplot3d import Axes3D        # registers 3‑D projection

# ===== Physical parameters =====
g, L = 9.81, 1.0
m, body_w, I = 0.5, 0.30, 2.0
damping = 0.05

# ===== Simulation / RandUP =====
M, N, dt = 10000, 50, 0.05         # particles, steps, timestep
R, r     = 2.0, 1.0              # torus radii

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
    hull_pts, hull_simp = None, []
    if len(P) >= 4:
        hull = ConvexHull(P)
        hull_pts  = P[hull.simplices]

    # ---- plotting ----
    fig = plt.figure(figsize=(6, 5))
    ax  = fig.add_subplot(111, projection='3d')

    # torus wireframe
    uu = np.linspace(-np.pi, np.pi, 40)
    vv = np.linspace(-np.pi, np.pi, 20)
    U, V = np.meshgrid(uu, vv)
    W = torus_embed(U.ravel(), V.ravel())
    ax.plot_wireframe(W[:,0].reshape(U.shape),
                      W[:,1].reshape(U.shape),
                      W[:,2].reshape(U.shape),
                      color='k', alpha=0.1)

    # particles
    ax.scatter(P[:,0], P[:,1], P[:,2], s=10, c='darkturquoise')

    # hull edges
    if hull_pts is not None:
        for tri in hull_pts:
            ax.plot(tri[:,0], tri[:,1], tri[:,2], c='orange', lw=1)

    ax.set_axis_off()
    ax.set_title(f"step {k+1}/{N}")
    plt.tight_layout()
    plt.savefig(f"frames/frame_{k:03d}.png", dpi=120)
    plt.close(fig)

print("Frames saved → ./frames")

# ===== Make GIF =====
images = [imageio.imread(f"frames/frame_{k:03d}.png") for k in range(N)]
imageio.mimsave("reachability.gif", images, fps=10)
print("GIF saved as reachability.gif")
