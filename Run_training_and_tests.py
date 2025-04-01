import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from stable_baselines3 import PPO
from torch.ao.nn.quantized.functional import threshold

from awake_steering_simulated import AwakeSteering
from helper_scripts.MPC import model_predictive_control
from Run_stepwise_optimizsation import iterative_optimization

# seed = 10
test_name = 'Analytic'
# test_name = 'PPO'
# test_name = 'Random'
# test_name = 'MPC'
# test_name = 'MPC_short'
test_name = 'Classical'
experiment_name = 'noise_test'

test_names = [
    'PPO',
    'Analytic',
    'Random',
    'MPC',
    'MPC_short',
    'Classical'
]

# Number of steps for evaluation
num_steps = 100

import os
from datetime import datetime
from pathlib import Path

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



def plot_trajecory(state_history, action_history, reward_history, noise_sigma, seed):
    # Create and save plot
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True, gridspec_kw={'hspace': 0.4})

    for i in range(state_history.shape[1]):
        axes[0].plot(state_history[:, i], marker='o', linestyle='-', label=f'State {i + 1}')
    axes[0].set_title("State Evolution Over Time")
    axes[0].set_ylabel("State Value")
    axes[0].legend()
    axes[0].grid(True)

    for i in range(action_history.shape[1]):
        axes[1].plot(action_history[:, i], marker='s', linestyle='-', label=f'Action {i + 1}')
    axes[1].set_title("Action Evolution Over Time")
    axes[1].set_ylabel("Action Value")
    axes[1].legend()
    axes[1].grid(True)

    axes[2].plot(reward_history, color='red', marker='x', linestyle='-', label="Reward")
    axes[2].set_title("Reward Over Time")
    axes[2].set_xlabel("Time Step")
    axes[2].set_ylabel("Reward")
    axes[2].legend()
    axes[2].set_ylim(-1, 0)
    axes[2].axhline(y=env.threshold, linestyle='--', color='black')
    plt.suptitle(f'{test_name} Policy Evolution with noise $\sigma$: {noise_sigma}, seed: {seed}')

    plt.tight_layout()

    # Save the plot
    #output_folder = Path("Plots") / f"{test_name}_policy_animation_noise_{noise_sigma}_seed_{seed}"
    #output_folder.mkdir(parents=True, exist_ok=True)
    #fig.savefig(output_folder / "evolution_plot.pdf", format="pdf")
    #fig.savefig(output_folder / "evolution_plot.png", format="png")
    #print(f"Figure saved as {output_folder}.pdf")
    #print(f"Figure saved as {output_folder}.png")


    # combining pkl files into one dataframe
    # Define the output folder using Path
    current_date = datetime.now().strftime('%Y-%m-%d')
    output_folder = Path('results') / experiment_name / f'Figures_{current_date}' / test_name / f'{test_name}_noise_sigma_{noise_sigma}'
    # Create the directory
    output_folder.mkdir(parents=True, exist_ok=True)
    # Save the figures
    current_date = datetime.now().strftime('%Y-%m-%d-%H%M%S')
    fig.savefig(output_folder / f"evolution_plot_{current_date}_seed_{seed}.pdf", format="pdf")
    fig.savefig(output_folder / f"evolution_plot_{current_date}_seed_{seed}.png", format="png")
    print(f"Figure saved as {output_folder / f'evolution_plot_{current_date}_seed_{seed}.pdf'}")
    print(f"Figure saved as {output_folder / f'evolution_plot_{current_date}_seed_{seed}.png'}")
    plt.show()




create_data = True
noise_sigma_list = [0, 0.1, 0.01, 0.05, 0.025, 0.001]
seed_list = [1,2,3,4,5,6,7,8,9]


if create_data:
    for test_name in test_names:
        trajectory_data_manager = TrajectoryDataManager(experiment_name=experiment_name, test_name=test_name)
        for noise_sigma in noise_sigma_list:
            # Define the environment and parameters
            env = AwakeSteering(noise_sigma=noise_sigma)#, seed=seed)
            for seed in seed_list:
                trajectory_data_manager.clear_data()
                if not test_name == 'Classical':
                    # Select policy
                    if test_name == 'PPO':
                        # Load trained model
                        noise_sigma_ppo = 0.005
                        seed_ppo = 1
                        model_save_path = Path(
                            "PPO_policy") / f"ppo_awake_steering_noise_sigma_{noise_sigma_ppo}_seed_{seed_ppo}"
                        if model_save_path.exists():
                            loaded_model = PPO.load(model_save_path, env=env)
                        else:
                            print('run ppo')
                            os.system(f'python ppo_train.py --noise_sigma {noise_sigma_ppo} --seed {seed_ppo}')
                            loaded_model = PPO.load(model_save_path, env=env)
                        policy_ppo = lambda obs: loaded_model.predict(obs, deterministic=True)[0]
                        policy_used = policy_ppo
                    elif test_name == 'Analytic':
                        invrmatrix = np.linalg.inv(env.response)
                        def policy_analytical(state):
                            return -invrmatrix.dot(state)
                        policy_used = policy_analytical
                    elif test_name == 'Random':
                        random_policy = lambda obs: env.action_space.sample()
                        policy_used = random_policy
                    elif test_name == 'MPC':
                        # Define the policy for MPC
                        mpc_horizon = 5
                        action_matrix_scaled = env.response
                        threshold = -env.threshold
                        mpc_tol = 1e-10
                        mpc_disp = True
                        policy_used = lambda x: model_predictive_control(x, mpc_horizon, action_matrix_scaled, threshold,
                                                                        plot=False, tol=mpc_tol, disp=mpc_disp)
                    elif test_name == 'MPC_short':
                        # Define the policy for MPC
                        mpc_horizon = 5
                        action_matrix_scaled = env.response
                        threshold = -env.threshold
                        mpc_tol = 1e-10
                        mpc_disp = True
                        policy_used = lambda x: model_predictive_control(x, mpc_horizon, action_matrix_scaled, threshold,
                                                                        plot=False, tol=mpc_tol, disp=mpc_disp,
                                                                         discount_factor=0.0)

                    obs, _ = env.reset(seed=seed)
                    done = False

                    trajectory_data_manager.add_step_data(obs, [np.nan]*env.action_space.shape[-1], [env._get_reward(obs)])

                    for _ in range(num_steps):
                        action = policy_used(obs)
                        print(f'action {action}')
                        obs, reward, done, _, _ = env.step(action)
                        trajectory_data_manager.add_step_data(obs, action, [reward])
                        if done:
                            break
                    trajectory_data_manager.save_data(noise_sigma, seed)
                else:
                    # Assuming 'env' is your AwakeSteering environment already instantiated:
                    episode_states, episode_actions, episode_rewards = iterative_optimization(env, trajectory_data_manager,
                                                                                              max_steps=50)


            state_history, action_history, reward_history = trajectory_data_manager.get_data()

        # plot_trajecory(state_history, action_history, reward_history, noise_sigma, seed)





def load_experiment_data(base_dir_root):
    """
    Load experiment results from stored pickle files.

    Args:
        base_dir_root (Path): Base directory where results are stored.
        test_name (str): Name of the test (e.g., 'PPO', 'Analytic').
        noise_levels (list): List of noise sigma values.

    Returns:
        pd.DataFrame: Combined DataFrame with all loaded results.
    """
    data_all_noises = []
    # Auto-detect available noise levels by checking subdirectories
    noise_levels = sorted(
        [d.name.split("_")[-1] for d in base_dir_root.iterdir() if d.is_dir() and d.name.startswith("noise_sigma_")],
        key=float
    )

    if not noise_levels:
        print("No noise level directories found.")
        return None

    print(f"Detected noise levels: {noise_levels}")
    for noise_sigma in noise_levels:
        base_dir = base_dir_root / f"noise_sigma_{noise_sigma}"
        print(f"Checking directory: {base_dir}")


        if not base_dir.exists():
            print(f"Directory does not exist: {base_dir}")
            continue

        df_all_seeds = []

        for file in base_dir.glob("*.pkl"):  # Process .pkl files only
            with open(file, "rb") as f:
                data = pickle.load(f)
                print(f"Loaded {file.name}")

                dfs = []
                for key in data:
                    columns = [f"{key}_{i}" for i in range(data[key].shape[-1])]
                    df = pd.DataFrame(data[key], columns=columns).T
                    dfs.append(df)

                df_episode = pd.concat(dfs).T
                seed = file.stem  # Extract seed from filename
                df_episode["Seed"] = seed
                df_episode["Time Step"] = df_episode.index

            df_all_seeds.append(df_episode)

        if df_all_seeds:
            df_all_seeds = pd.concat(df_all_seeds, ignore_index=True)
            df_all_seeds["Noise Sigma"] = noise_sigma
            data_all_noises.append(df_all_seeds)

    if data_all_noises:
        return pd.concat(data_all_noises, ignore_index=True)
    else:
        print("No data found for the specified noise levels.")
        return None


def plot_experiment_results(df_combined):
    """
    Plot reward evolution over time for different noise levels.

    Args:
        df_combined (pd.DataFrame): Processed DataFrame with experiment data.
    """
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df_combined, x="Time Step", y="reward_0", hue="Noise Sigma", errorbar='sd')
    plt.title(f"Reward vs Time for Different Noise Levels for {test_name}")
    plt.xlabel("Time Step")
    plt.ylabel("Reward")
    plt.legend(title="Noise Levels")
    plt.grid(True)
    plt.show()

plot_results = False
if plot_results:

    # -------------------------- Execution --------------------------

    # base_dir_root = Path("results") / experiment_name / f"Results_{current_date_pickle}" / test_name
    base_dir_root = Path("results") / experiment_name / 'Results_2025-03-19' / test_name

    # Load and process data
    df_combined = load_experiment_data(base_dir_root)

    print(df_combined)

    # Plot results if data is available
    if df_combined is not None:
        plot_experiment_results(df_combined)


