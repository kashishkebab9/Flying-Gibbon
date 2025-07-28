import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull, Delaunay
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

# ===== Physical parameters =====
g, L = 9.81, 1.0
m, body_w, I = 0.5, 0.30, 2.0
damping = 0.05

# ===== Simulation / RandUP =====
M, N, dt = 5000, 100, 0.05
R, r     = 2.0, 1.0

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

# ===== Precompute simulation =====
X = np.zeros((M, 8))
X[:, 0] = np.random.uniform(-0.3, 0.3, M)
X[:, 2] = np.random.uniform(-0.3, 0.3, M)
F1 = np.random.uniform(0.0, 2.4525, (M, N))
F2 = np.random.uniform(0.0, 2.4525, (M, N))
TA = np.random.uniform(-10.0, 10.0, (M, N))

particles_over_time = []
hulls_over_time = []

for k in range(N):
    U_k = np.stack([F1[:, k], F2[:, k], TA[:, k]], axis=1)
    for j in range(M):
        X[j] = rk4(X[j], U_k[j])
        X[j, 0] = (X[j, 0]+np.pi) % (2*np.pi) - np.pi
        X[j, 2] = (X[j, 2]+np.pi) % (2*np.pi) - np.pi

    θ, ϕ = X[:, 0], X[:, 2]
    P = torus_embed(θ, ϕ)
    particles_over_time.append(P)

    if len(P) >= 4:
        hull = ConvexHull(P)
        hull_pts = P[hull.simplices]
    else:
        hull_pts = np.empty((0, 3, 3))

    hulls_over_time.append(hull_pts)

# ===== Plot Setup =====
fig = plt.figure(figsize=(12, 6))
ax = fig.add_subplot(111, projection='3d')

uu = np.linspace(-np.pi, np.pi, 40)
vv = np.linspace(-np.pi, np.pi, 20)
U, V = np.meshgrid(uu, vv)
W = torus_embed(U.ravel(), V.ravel())
wire_x = W[:,0].reshape(U.shape)
wire_y = W[:,1].reshape(U.shape)
wire_z = W[:,2].reshape(U.shape)

scat = ax.scatter([], [], [], s=5, c='darkturquoise', alpha=0.6)
lines = []

def init():
    ax.plot_wireframe(wire_x, wire_y, wire_z, color='k', alpha=0.1)
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.set_zlim(-2, 2)
    ax.set_axis_off()
    ax.view_init(elev=30, azim=45)
    return scat,

def update(frame):
    global lines
    for line in lines:
        line.remove()
    lines.clear()

    P = particles_over_time[frame]
    scat._offsets3d = (P[:, 0], P[:, 1], P[:, 2])

    hull_pts = hulls_over_time[frame]
    for tri in hull_pts:
        line, = ax.plot(tri[:,0], tri[:,1], tri[:,2], color='orange', lw=1)
        lines.append(line)

    ax.set_title(f"Time: {dt*frame:.2f}s")
    return scat, *lines

ani = FuncAnimation(fig, update, frames=N, init_func=init, blit=False, interval=50)
plt.show()
