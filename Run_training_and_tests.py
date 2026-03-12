import os
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
import seaborn as sns
from stable_baselines3 import PPO

from awake_steering_simulated import AwakeSteering
from helper_scripts.MPC import model_predictive_control
from helper_scripts.data_management import TrajectoryDataManager
from Run_stepwise_optimization import iterative_optimization

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


def plot_trajectory(state_history, action_history, reward_history, noise_sigma, seed, test_name, env):
    """Create and save trajectory evolution plots."""
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
    plt.suptitle(f'{test_name} Policy Evolution with noise $\\sigma$: {noise_sigma}, seed: {seed}')

    plt.tight_layout()

    current_date = datetime.now().strftime('%Y-%m-%d')
    output_folder = Path('results') / experiment_name / f'Figures_{current_date}' / test_name / f'{test_name}_noise_sigma_{noise_sigma}'
    output_folder.mkdir(parents=True, exist_ok=True)
    current_date = datetime.now().strftime('%Y-%m-%d-%H%M%S')
    fig.savefig(output_folder / f"evolution_plot_{current_date}_seed_{seed}.pdf", format="pdf")
    fig.savefig(output_folder / f"evolution_plot_{current_date}_seed_{seed}.png", format="png")
    print(f"Figure saved as {output_folder / f'evolution_plot_{current_date}_seed_{seed}.pdf'}")
    plt.show()


create_data = True
noise_sigma_list = [0, 0.1, 0.01, 0.05, 0.025, 0.001]
seed_list = [1, 2, 3, 4, 5, 6, 7, 8, 9]


if create_data:
    for test_name in test_names:
        trajectory_data_manager = TrajectoryDataManager(experiment_name=experiment_name, test_name=test_name)
        for noise_sigma in noise_sigma_list:
            env = AwakeSteering(noise_sigma=noise_sigma)
            for seed in seed_list:
                trajectory_data_manager.clear_data()
                if test_name != 'Classical':
                    # Select policy
                    if test_name == 'PPO':
                        noise_sigma_ppo = 0.005
                        seed_ppo = 1
                        model_save_path = Path("PPO_policy") / f"ppo_awake_steering_noise_sigma_{noise_sigma_ppo}_seed_{seed_ppo}"
                        if model_save_path.exists():
                            loaded_model = PPO.load(model_save_path, env=env)
                        else:
                            print('run ppo')
                            os.system(f'python ppo_train.py --noise_sigma {noise_sigma_ppo} --seed {seed_ppo}')
                            loaded_model = PPO.load(model_save_path, env=env)
                        policy_used = lambda obs: loaded_model.predict(obs, deterministic=True)[0]
                    elif test_name == 'Analytic':
                        invrmatrix = np.linalg.inv(env.response)
                        policy_used = lambda state: -invrmatrix.dot(state)
                    elif test_name == 'Random':
                        policy_used = lambda obs: env.action_space.sample()
                    elif test_name == 'MPC':
                        mpc_horizon = 5
                        action_matrix_scaled = env.response
                        threshold = -env.threshold
                        mpc_tol = 1e-10
                        mpc_disp = True
                        policy_used = lambda x: model_predictive_control(
                            x, mpc_horizon, action_matrix_scaled, threshold,
                            plot=False, tol=mpc_tol, disp=mpc_disp
                        )
                    elif test_name == 'MPC_short':
                        mpc_horizon = 5
                        action_matrix_scaled = env.response
                        threshold = -env.threshold
                        mpc_tol = 1e-10
                        mpc_disp = True
                        policy_used = lambda x: model_predictive_control(
                            x, mpc_horizon, action_matrix_scaled, threshold,
                            plot=False, tol=mpc_tol, disp=mpc_disp,
                            discount_factor=0.0
                        )

                    obs, _ = env.reset(seed=seed)
                    done = False

                    trajectory_data_manager.add_step_data(
                        obs, [np.nan] * env.action_space.shape[-1], [env._get_reward(obs)]
                    )

                    for _ in range(num_steps):
                        action = policy_used(obs)
                        print(f'action {action}')
                        obs, reward, done, _, _ = env.step(action)
                        trajectory_data_manager.add_step_data(obs, action, [reward])
                        if done:
                            break
                    trajectory_data_manager.save_data(noise_sigma, seed)
                else:
                    episode_states, episode_actions, episode_rewards = iterative_optimization(
                        env, trajectory_data_manager, max_steps=50
                    )

            state_history, action_history, reward_history = trajectory_data_manager.get_data()


def load_experiment_data(base_dir_root):
    """Load experiment results from stored pickle files."""
    data_all_noises = []
    noise_levels = sorted(
        [d.name.split("_")[-1] for d in base_dir_root.iterdir()
         if d.is_dir() and d.name.startswith("noise_sigma_")],
        key=float
    )

    if not noise_levels:
        print("No noise level directories found.")
        return None

    print(f"Detected noise levels: {noise_levels}")
    for noise_sigma in noise_levels:
        base_dir = base_dir_root / f"noise_sigma_{noise_sigma}"

        if not base_dir.exists():
            print(f"Directory does not exist: {base_dir}")
            continue

        df_all_seeds = []

        for file in base_dir.glob("*.pkl"):
            with open(file, "rb") as f:
                data = pickle.load(f)

            dfs = []
            for key in data:
                columns = [f"{key}_{i}" for i in range(data[key].shape[-1])]
                df = pd.DataFrame(data[key], columns=columns).T
                dfs.append(df)

            df_episode = pd.concat(dfs).T
            seed = file.stem
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


def plot_experiment_results(df_combined, test_name):
    """Plot reward evolution over time for different noise levels."""
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
    base_dir_root = Path("results") / experiment_name / 'Results_2025-03-19' / test_name
    df_combined = load_experiment_data(base_dir_root)
    print(df_combined)
    if df_combined is not None:
        plot_experiment_results(df_combined, test_name)
