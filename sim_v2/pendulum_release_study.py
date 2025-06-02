import optuna
from projectile import simulate_projectile  # make sure this accepts release state
from pendulum import simulate_pendulum
import numpy as np

def objective(trial):
    # Sample all four release state variables
    theta = trial.suggest_float("theta", 0, np.pi)
    theta_dot = trial.suggest_float("theta_dot", 0, 10.0)
    phi = trial.suggest_float("phi", -np.pi / 2, np.pi / 2)
    phi_dot = trial.suggest_float("phi_dot", -10.0, 10.0)

    release_state = [theta, theta_dot, phi, phi_dot]
    t_values, theta_pend, omega_pend, phi_pend, phi_dot_pend, t_release, U_opt = simulate_pendulum(release_state=release_state)
    if len(t_values) == 0:
        return 1e6
    else:
        t_proj, x_vals, y_vals, theta_vals, alpha_vals, control_vals, T_opt = simulate_projectile(
            t_release, release_state=release_state
        )
        if len(x_vals) == 0:
            return 1e6
    

    # Evaluate cost
    goal_x = 10.0
    goal_y = 0.0
    print(len(x_vals))
    x_final = x_vals[-1]
    y_final = y_vals[-1]
    position_error = (x_final - goal_x)**2 + (y_final - goal_y)**2
    total_time = t_proj[-1]

    return position_error + 0.1 * total_time

# Set up and run the study
if __name__ == "__main__":
    study = optuna.create_study(
        storage="sqlite:///db.sqlite3",  # Specify the storage URL here.
        study_name="release_state",
        load_if_exists=True
    )
    study.optimize(objective, n_trials=100)

    print("Best Parameters:", study.best_params)
    print("Best Objective Value:", study.best_value)
