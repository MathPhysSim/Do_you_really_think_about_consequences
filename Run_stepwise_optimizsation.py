import os
import pickle
from datetime import datetime
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import minimize

from awake_steering_simulated import AwakeSteering


def create_experiment_folder(experiment_name: str, algorithm: str, parameter_name: str) -> Path:
    """
    Creates a structured experiment folder based on the experiment name, algorithm, and parameter name.

    Args:
        experiment_name (str): Name of the experiment.
        algorithm (str): Name of the algorithm used.
        parameter_name (str): Parameter name associated with the experiment.

    Returns:
        Path: The path to the created experiment folder.
    """
    current_date = datetime.now().strftime('%Y-%m-%d')
    save_folder_results = Path("results") / experiment_name / f"Results_{current_date}" / algorithm / parameter_name

    try:
        save_folder_results.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        print(f"Error creating directory {save_folder_results}: {e}")
        raise

    return save_folder_results, current_date


class TrajectoryDataManager:

    def __init__(self, experiment_name, test_name):
        # Initialize storage
        self.experiment_name = experiment_name
        self.test_name = test_name

        self.state_history = []
        self.action_history = []
        self.reward_history = []

    def add_step_data(self, state, action ,reward):
        self.action_history.append(action)
        self.state_history.append(state)
        self.reward_history.append(reward)

    def clear_data(self):
        self.state_history = []
        self.action_history = []
        self.reward_history = []

    def get_data(self):

        return np.array(self.state_history), np.array(self.action_history), np.array(self.reward_history)

    def save_data(self, noise_sigma, seed):
        # Save results
        results_data = {
            'state': np.array(self.state_history),
            'action': np.array(self.action_history),
            'reward': np.array(self.reward_history)
        }
        save_path, current_date_pickle = create_experiment_folder(experiment_name=self.experiment_name,
                                                                  algorithm=self.test_name,
                                                                  parameter_name=f'noise_sigma_{noise_sigma}')
        save_file_name = os.path.join(save_path, f'{seed}.pkl')
        with open(save_file_name, 'wb') as f:
            pickle.dump(results_data, f)
        print(f"Results saved at {save_file_name}")

        # self.clear_data()

def get_bounds(current_state):
    # Example: dynamic bounds based on the current state.
    # Here we simply return fixed bounds, but you can customize this
    # to depend on current_state if needed.
    action_dim = len(current_state)  # or however you want to define it
    return [(-1, 1)] * action_dim

def iterative_optimization(env, trajectory_data_manager, max_steps=50):
    # Reset the environment
    obs, info = env.reset()
    episode_states = [obs]
    episode_actions = []
    episode_rewards = [env._get_reward(obs)]

    trajectory_data_manager.add_step_data(obs, [np.nan] * env.action_space.shape[-1], [env._get_reward(obs)])

    # Initialize action
    action = np.zeros(env.action_space.shape)
    stop_optimization = env.check_threshold_condition(obs)  # External flag to track termination

    for step in range(max_steps):
        if stop_optimization:  # Stop immediately if flagged
            print(f"Optimization stopped early at step {step} due to termination condition.")
            break

        # Determine bounds for this step
        bounds = get_bounds(env.state)

        def optimize_step(action):
            """Objective function: Runs the step and returns negative reward"""
            nonlocal obs, stop_optimization, step  # Ensure flag updates outside
            if not stop_optimization:
                episode_actions.append(action)
                next_obs, reward, done, truncated, info = env.step(action)
                step+=1
                obs = next_obs  # Update observation
                # Append to states and rewards
                episode_states.append(next_obs)
                episode_rewards.append(reward)

                trajectory_data_manager.add_step_data(obs, action, [reward])

                # If the episode is done, return a high penalty and force early exit
                if done or truncated:
                    print(f"Early stopping at step {step + 1}: done={done}, truncated={truncated}")
                    stop_optimization = True  # Signal to stop the main loop

                    return 1e6  # Large penalty to discourage further evaluation
            else:
                return 1e6
            return -reward  # Maximizing reward

        # print(f'stop optimization {stop_optimization}')
        # Enable absolute settings mode
        env.set_use_absolute_settings(True)

        # Optimize with a callback that stops the process naturally
        # minimize(optimize_step, x0=action, method='COBYLA', bounds=bounds, tol=1e-1)
        minimize(optimize_step, x0=action, method='COBYLA', bounds=bounds, tol=1e-1, options={'disp': False})

        # Disable absolute settings after optimization
        env.set_use_absolute_settings(False)

    # Plot the objective function evaluations over time
    plt.figure(figsize=(10, 6))
    plt.plot(episode_rewards, marker='o', linestyle='-', label='Reward per Step')
    plt.xlabel('Step')
    plt.ylabel('Reward')
    plt.title('Reward Evolution Over Steps')
    plt.legend()
    plt.grid(True)
    plt.show()

    return episode_states, episode_actions, episode_rewards


if __name__ == '__main__':

    num_steps = 50
    noise_sigma_list = [0, 0.001, 0.01, 0.025, 0.05, 0.1]
    seed_list = [1,2,3,4,5,6,7,8,9]
    experiment_name = 'noise_test'
    test_name = 'Classical'
    trajectory_data_manager = TrajectoryDataManager(experiment_name=experiment_name, test_name=test_name)



    # def optimize_standard(env):
    #     # Reset the environment and obtain the initial observation
    #     obs, info = env.reset()
    #
    #     # Temporary storage for episode data
    #     episode_states = [obs]
    #     episode_actions = []
    #     episode_rewards = [env._get_reward(obs)]
    #
    #     def optimize_env(action):
    #         episode_actions.append(action)
    #         next_obs, reward, done, truncated, info = env.step(action)
    #         episode_states.append(next_obs)
    #         episode_rewards.append(reward)
    #
    #         return -reward
    #
    #
    #     # Define the bounds for each action component (e.g., [-1, 1])
    #     initial_action = np.zeros(env.action_space.shape)
    #     bounds_actions = [(-1, 1)] * len(initial_action)
    #
    #     # Global optimization using Differential Evolution
    #     result_global = differential_evolution(optimize_env, bounds=bounds_actions, tol=1e-16, disp=True)
    #
    #     print("Global Optimization Result:", result_global)
    #
    #     # Plot the objective function evaluations over time
    #     plt.figure(figsize=(10, 6))
    #     plt.plot(episode_rewards, marker='o', linestyle='-', label='Objective Value')
    #     plt.xlabel('Evaluation Number')
    #     plt.ylabel('Objective Function Value')
    #     plt.title('Objective Function Evaluations Over Time')
    #     plt.legend()
    #     plt.grid(True)
    #     plt.show()
    #
    #     return episode_states, episode_actions, episode_rewards
    #
    #     # Set seed for reproducibility


    for noise_sigma in noise_sigma_list:
        for seed in seed_list:
            trajectory_data_manager.clear_data()
            env = AwakeSteering(seed=seed, noise_sigma=noise_sigma, use_absolute_settings=False)
            # Assuming 'env' is your AwakeSteering environment already instantiated:
            episode_states, episode_actions, episode_rewards = iterative_optimization(env,
                                                                                      trajectory_data_manager,
                                                                                      max_steps=num_steps)

            trajectory_data_manager.save_data(noise_sigma, seed)
            #
            # if episode_actions:
            #     plot_trajectories([episode_rewards], [episode_states], [episode_actions])

