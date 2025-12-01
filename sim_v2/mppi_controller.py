import yaml
import numpy as np
from hybrid_dynamics import pendulum_dynamics, projectile_dynamics, rk4_step

class MPPI:
    def __init__(self, config_file="config.yaml", dt=0.01, config=None):
        with open(config_file, "r") as file:
            config = yaml.safe_load(file)

        self.num_samples = config["mppi"]["num_samples"]
        self.horizon_length = config["mppi"]["horizon_length"]
        self.temperature = config["mppi"]["temperature"]
        self.sigma = config["mppi"]["sigma"]
        self.dt = config["simulation_parameters"]["dt"]
        self.rotor_bound = config["mppi"]["control_bounds"]["force_rotor"]
        self.arm_torque_bound = config["mppi"]["control_bounds"]["torque"]

    # what is a good u_nominal?
    # so you can take the prior, "optimal" control sequence for a good u_nominal
    # And, with enough samples, we can sample from a uniform distribution
    def step(self, u_nominal):
        u_nominal = [5.0, 5.0, 1.0]
        output = mppi.sample_control_trajectories(u_nominal)
        pass

    def sample_control_trajectories(self, u_nominal):
        control_samples = []
        for i in range(self.num_samples):
            control_sequence = []
            for j in range(self.horizon_length):
                noise_rotor_1 = np.random.uniform(self.rotor_bound[0], self.rotor_bound[1])
                noise_rotor_2 = np.random.uniform(self.rotor_bound[0], self.rotor_bound[1])
                noise_torque = np.random.uniform(self.arm_torque_bound[0], self.arm_torque_bound[1])
                u = [0, 0, 0]
                u[0] = u_nominal[0] + noise_rotor_1
                u[1] = u_nominal[1] + noise_rotor_2
                u[2] = u_nominal[2] + noise_torque
                control_sequence.append(u)
            control_samples.append(control_sequence)
        return control_samples

    def forward_simulate_trajectories(self, current_x, control_sequence, mode_function, t):
        states = []
        x = current_x
        states.append(x)

        for i in range(control_sequence):
            x = rk4_step(mode_function, x, t, self.dt)
            states.append(x)

        return states


    def evaluate_trajectory(self, u_sequence, states, target_state, dynamics_mode):
        # if len(u_sequence) != len(states):
        #     print("WE GOT AN ERROR")
        # total_cost = 0
        # for i in range(u_sequence):
        #     total_cost += self.state_cost(states[i])
        #     total_cost += self.control_cost(u_sequence[i])
        pass

    def weigh_trajectories(self):
        pass

    def update_control_sequence(self):
        pass

    def state_cost(self, state, mode):
        pass

    def control_cost(self, u, mode):
        if mode == 0:
            rotor_weight = 100
            torque_weight = 10
        elif mode == 1:

        cost = rotor_weight * u[0] + rotor_weight * u[1] + torque_weight * u[2]
        return cost

if __name__=="__main__":
    mppi = MPPI()
    u_nominal = [5.0, 5.0, 1.0]
    output = mppi.sample_control_trajectories(u_nominal)
    print(output)

    mppi.step(output[0])
