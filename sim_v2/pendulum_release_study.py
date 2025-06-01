import optuna
from projectile import simulate_projectile  # make sure this accepts release state
import numpy as np

def objective(trial):
    # Sample all four release state variables
    theta = trial.suggest_float("theta", -np.pi, np.pi)
    theta_dot = trial.suggest_float("theta_dot", -10.0, 10.0)
    phi = trial.suggest_float("phi", -np.pi / 2, np.pi / 2)
    phi_dot = trial.suggest_float("phi_dot", -10.0, 10.0)

    # Fixed release time (or make this a trial variable too)
    t_release = 1.0

    release_state = [theta, theta_dot, phi, phi_dot]

    try:
        # Run projectile phase only
        t_proj, state_proj, control_proj = simulate_projectile(
            t_release, theta, theta_dot, phi, phi_dot
        )
    except Exception as e:
        # If it fails (e.g., NaNs or out of bounds), assign large penalty
        return 1e6

    # Evaluate cost
    goal_x = 10.0
    goal_y = 0.0
    x_final = state_proj[-1][0]
    y_final = state_proj[-1][1]
    position_error = (x_final - goal_x)**2 + (y_final - goal_y)**2
    total_time = t_proj[-1]

    return position_error + 0.1 * total_time

# Set up and run the study
if __name__ == "__main__":
    study = optuna.create_study(
        storage="sqlite:///db.sqlite3",  # Specify the storage URL here.
        study_name="release_state"
    )
    study.optimize(objective, n_trials=100)

    print("Best Parameters:", study.best_params)
    print("Best Objective Value:", study.best_value)
