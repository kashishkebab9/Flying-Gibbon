
import casadi as ca
import numpy as np
from matplotlib.animation import FFMpegWriter

# --- System parameters ---
L = 1
body_w = .6
body_h = 0.2
g = 9.81
m = 3.0 # kg
I = 2.0 # need to figure this out
damping = 0.5

F_min, F_max = 0.0, 25     # Thrust bounds (N)
tau_min, tau_max = -2.0, 2.0 # Torque bounds (Nm)

# State:  x = [theta, theta_dot, phi, phi_dot]
# Control: u = [Fl, Fr, tau]

# --- CasADi symbols ---
x = ca.SX.sym("x", 4)   # [theta, theta_dot, phi, phi_dot]
u = ca.SX.sym("u", 3)   # [Fl, Fr, tau]

theta, theta_dot, phi, phi_dot = x[0], x[1], x[2], x[3]
Fl, Fr, tau = u[0], u[1], u[2]

# Dynamics (as you wrote them)
sin_diff = ca.sin(theta - phi)
theta_ddot = -g * ca.sin(theta) + (1.0 / m) * sin_diff * (Fl + Fr) - damping * theta_dot
phi_ddot   = -(body_w / 2.0) * Fl + (body_w / 2.0) * Fr + I * tau

f = ca.vertcat(theta_dot, theta_ddot, phi_dot, phi_ddot)

# Continuous-time dynamics function
f_fun = ca.Function("f_fun", [x, u], [f])

# --- Discretization (RK4) ---
def rk4_step(xk, uk, dt):
    k1 = f_fun(xk,             uk)
    k2 = f_fun(xk + dt/2 * k1, uk)
    k3 = f_fun(xk + dt/2 * k2, uk)
    k4 = f_fun(xk + dt   * k3, uk)
    x_next = xk + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
    return x_next

# --- OCP parameters ---
N = 200          # horizon steps
dt = 0.1        # time step (s)
T = N * dt

# Desired final state
theta_des = np.pi
phi_des   = 0.0

# Weights
Q_theta = 10.0
Q_phi   = 10.0
Q_vel   = 1.0
R_u     = 1.5
Qf_theta = 1000.0
Qf_phi   = 1000.0
Qf_vel   = 1000.0

# --- Decision variables and constraints ---
w   = []   # optimization variables
w_lb = []  # lower bounds
w_ub = []  # upper bounds
g_constr = []  # equality/inequality constraints
g_lb = []      # constraint lower bounds
g_ub = []      # constraint upper bounds

# Initial condition parameter
x0_param = ca.SX.sym("x0_param", 4)

# Collect states and controls for convenience
Xk_list = []
Uk_list = []

# --- Initial state decision variable ---
Xk = ca.SX.sym("X_0", 4)
w.append(Xk)
# bounds on theta, theta_dot, phi, phi_dot at k=0:
# here we don't restrict initial state besides phi range; you can tighten if you like
phi_max = np.pi/2
phi_min = -np.pi/2

w_lb += [-ca.inf, -ca.inf, phi_min, -ca.inf]
w_ub += [ ca.inf,  ca.inf, phi_max,  ca.inf]

# Initial condition constraint X0 == x0_param
g_constr.append(Xk - x0_param)
g_lb += [0.0]*4
g_ub += [0.0]*4

Xk_list.append(Xk)

J = 0  # cost

# --- Loop over horizon ---
for k in range(N):
    # Control at step k
    Uk = ca.SX.sym(f"U_{k}", 3)
    w.append(Uk)

    # Input bounds: Fl, Fr in [F_min,F_max], tau in [tau_min,tau_max]
    w_lb += [F_min, F_min, tau_min]
    w_ub += [F_max, F_max, tau_max]

    Uk_list.append(Uk)
    Fl_k = Uk[0]
    Fr_k = Uk[1]
    g_constr.append(Fl_k - Fr_k)  # enforce Fl - Fr = 0
    g_lb.append(0.0)
    g_ub.append(0.0)

    # Stage cost (tracking + control effort)
    theta_k = Xk[0]
    theta_dot_k = Xk[1]
    phi_k = Xk[2]
    phi_dot_k = Xk[3]

    J += Q_theta*(theta_k - theta_des)**2 \
       + Q_phi*(phi_k - phi_des)**2 \
       + Q_vel*(theta_dot_k**2 + phi_dot_k**2) \
       + R_u*ca.sumsqr(Uk)

    # Integrate dynamics
    Xk_next = ca.SX.sym(f"X_{k+1}", 4)
    w.append(Xk_next)

    # State bounds: phi in [-90deg, 90deg] for all k
    w_lb += [-ca.inf, -ca.inf, phi_min, -ca.inf]
    w_ub += [ ca.inf,  ca.inf, phi_max,  ca.inf]

    Xk_list.append(Xk_next)

    # Dynamics constraint: X_{k+1} - F(X_k, U_k) = 0
    x_next_pred = rk4_step(Xk, Uk, dt)
    g_constr.append(Xk_next - x_next_pred)
    g_lb += [0.0]*4
    g_ub += [0.0]*4

    Xk = Xk_next  # move to next step

# --- Terminal cost ---
theta_N = Xk[0]
theta_dot_N = Xk[1]
phi_N   = Xk[2]
phi_dot_N = Xk[3]

J += Qf_theta*(theta_N - theta_des)**2 \
   + Qf_phi*(phi_N - phi_des)**2 \
   + Qf_vel*(theta_dot_N**2 + phi_dot_N**2)

# --- Build NLP ---
w   = ca.vertcat(*w)
g_constr = ca.vertcat(*g_constr)

nlp = {"x": w, "f": J, "g": g_constr, "p": x0_param}

solver = ca.nlpsol(
    "solver", "ipopt", nlp,
    {
        "ipopt.print_level": 0,
        "print_time": 0,
        "ipopt.max_iter": 1000,
    }
)

# --- Solve for a given initial condition ---
x0_val = np.array([0.0, 0.0, 0.0, 0.0])  # example: upright-ish initial, phi=0

sol = solver(
    x0=np.zeros(w.shape[0]),
    lbx=w_lb,
    ubx=w_ub,
    lbg=g_lb,
    ubg=g_ub,
    p=x0_val
)

w_opt = sol["x"].full().flatten()

# Extract solution trajectories
X_opt = []
U_opt = []
idx = 0

# X_0
X_opt.append(w_opt[idx:idx+4])
idx += 4

for k in range(N):
    # U_k
    U_opt.append(w_opt[idx:idx+3])
    idx += 3
    # X_{k+1}
    X_opt.append(w_opt[idx:idx+4])
    idx += 4

X_opt = np.array(X_opt)  # shape (N+1, 4)
U_opt = np.array(U_opt)  # shape (N, 3)

print("Final state:", X_opt[-1])
print("Final theta (rad):", X_opt[-1,0], "target:", theta_des)
print("Final phi (rad):",   X_opt[-1,2], "target:", phi_des)

# ============================================================
# === Visualization: Animation + State / Control Plots    ===
# ============================================================
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.transforms as tr
from matplotlib.animation import FFMpegWriter
import numpy as np

# --- Animation ---
fig, ax = plt.subplots()
ax.set_aspect('equal')
ax.set_xlim(-1.5, 1.5)
ax.set_ylim(-1.5, 1.5)

# Pendulum line from origin to point mass
line, = ax.plot([], [], 'ko-', lw=2)

# Rectangular body attached at pendulum tip
body_rect = plt.Rectangle((0, 0), body_w, body_h, color='blue', alpha=0.5)
ax.add_patch(body_rect)

def init():
    line.set_data([], [])
    # Move body offscreen initially
    body_rect.set_xy((-10, -10))
    return line, body_rect

def update(frame):
    # X_opt has shape (N+1, 4): [theta, theta_dot, phi, phi_dot]
    theta, _, phi, _ = X_opt[frame]

    # Pendulum tip position
    x_p = L * np.sin(theta)
    y_p = -L * np.cos(theta)

    # Rotation matrix for body (theta + phi)
    rot = np.array([
        [np.cos(theta + phi), -np.sin(theta + phi)],
        [np.sin(theta + phi),  np.cos(theta + phi)]
    ])

    # Rectangle corners in body frame (centered at (0, 0))
    corners = np.array([
        [-body_w/2, -body_h/2],
        [ body_w/2, -body_h/2],
        [ body_w/2,  body_h/2],
        [-body_w/2,  body_h/2]
    ])
    rotated = (rot @ corners.T).T + np.array([x_p, y_p])

    # Update pendulum line
    line.set_data([0, x_p], [0, y_p])

    # Affine transform: rotate around the pendulum tip
    transform = (tr.Affine2D()
                 .rotate_around(x_p, y_p, theta + phi)
                 + ax.transData)

    # Set the rectangle's lower-left corner and transform
    body_rect.set_xy((x_p - body_w / 2, y_p - body_h / 2))
    body_rect.set_transform(transform)

    return line, body_rect

# Note: X_opt has N+1 states, so animate all of them
ani = FuncAnimation(
    fig,
    update,
    frames=len(X_opt),
    init_func=init,
    blit=True,
    interval=dt * 1000,
    repeat=False
)

plt.title("Optimized Trajectory: Pendulum with Actuated Body")
plt.show()

# Optional: save video
# writer = FFMpegWriter(fps=int(1 / dt),
#                       metadata=dict(artist='Trajectory Opt'),
#                       bitrate=1800)
# ani.save("optimized_pendulum_trajectory.mp4", writer=writer)
# print("Video saved as optimized_pendulum_trajectory.mp4")

# --- Time-series plots ---
time_states = np.linspace(0, T, N + 1)  # for X_opt
time_ctrls  = np.linspace(0, T - dt, N) # for U_opt

plt.figure(figsize=(10, 6))

# States: theta, phi
plt.subplot(2, 1, 1)
plt.plot(time_states, X_opt[:, 0], label='theta')
plt.plot(time_states, X_opt[:, 2], label='phi')
plt.legend()
plt.title('State Trajectories')
plt.ylabel('Angle (rad)')

# Controls: Fl, Fr, tau
plt.subplot(2, 1, 2)
plt.plot(time_ctrls, U_opt[:, 0], label='Fl')
plt.plot(time_ctrls, U_opt[:, 1], label='Fr')
plt.plot(time_ctrls, U_opt[:, 2], label='tau')
plt.legend()
plt.title('Control Inputs')
plt.xlabel('Time (s)')
plt.ylabel('Input')
plt.tight_layout()
plt.show()

# fps = int(1 / dt)  # or just set e.g. fps = 30
# writer = FFMpegWriter(
#     fps=fps,
#     metadata=dict(artist='Trajectory Opt'),
#     bitrate=1800
# )
# 
# ani.save("optimized_pendulum_trajectory.mp4", writer=writer)
# print("Video saved as optimized_pendulum_trajectory.mp4")
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

num_frames = 300  # number of faded snapshots to plot
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
    ax.fill(rotated[:,0], rotated[:,1], color='skyblue', alpha=0.15 + 0.8*alpha)

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
ax.fill(rotated_f[:,0], rotated_f[:,1], color='teal', alpha=1.0)

plt.xlabel("X (m)")
plt.ylabel("Y (m)")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()
