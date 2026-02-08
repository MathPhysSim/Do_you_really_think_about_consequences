import os
import pickle
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Tuple, List, Optional

import numpy as np


class TrajectoryDataManager:
    """
    Manages trajectory data collection and storage for experiments.
    Provides a unified interface for all experiment types.
    """

    def __init__(self, experiment_name: str, test_name: str):
        """
        Initialize the trajectory data manager.

        Args:
            experiment_name: Name of the experiment
            test_name: Name of the test/algorithm
        """
        self.experiment_name = experiment_name
        self.test_name = test_name

        self.state_history = []
        self.action_history = []
        self.reward_history = []

    def add_step_data(self, state: np.ndarray, action: np.ndarray, reward: float) -> None:
        """
        Add data from a single environment step.

        Args:
            state: Environment state
            action: Action taken
            reward: Reward received
        """
        self.action_history.append(action)
        self.state_history.append(state)
        self.reward_history.append(reward if isinstance(reward, list) else [reward])

    def clear_data(self) -> None:
        """
        Clear all stored trajectory data.
        """
        self.state_history = []
        self.action_history = []
        self.reward_history = []

    def get_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get all stored trajectory data.

        Returns:
            Tuple of (states, actions, rewards) as numpy arrays
        """
        return np.array(self.state_history), np.array(self.action_history), np.array(self.reward_history)

    def save_data(self, noise_sigma: float, seed: int) -> str:
        """
        Save the collected trajectory data to a file.

        Args:
            noise_sigma: Noise level used in the experiment
            seed: Random seed used

        Returns:
            Path to the saved file
        """
        # Save results
        results_data = {
            'state': np.array(self.state_history),
            'action': np.array(self.action_history),
            'reward': np.array(self.reward_history)
        }
        save_path, _ = create_experiment_folder(
            experiment_name=self.experiment_name,
            algorithm=self.test_name,
            parameter_name=f'noise_sigma_{noise_sigma}'
        )
        save_file_name = os.path.join(save_path, f'{seed}.pkl')
        with open(save_file_name, 'wb') as f:
            pickle.dump(results_data, f)
        print(f"Results saved at {save_file_name}")

        return save_file_name


def create_experiment_folder(experiment_name: str, algorithm: str, parameter_name: str) -> Tuple[Path, str]:
    """
    Creates a structured experiment folder based on the experiment name, algorithm, and parameter name.

    Args:
        experiment_name: Name of the experiment
        algorithm: Name of the algorithm used
        parameter_name: Parameter name associated with the experiment

    Returns:
        Tuple containing the path to the created experiment folder and current date string
    """
    current_date = datetime.now().strftime('%Y-%m-%d')
    save_folder_results = Path("results") / experiment_name / f"Results_{current_date}" / algorithm / parameter_name

    try:
        save_folder_results.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        print(f"Error creating directory {save_folder_results}: {e}")
        raise

    return save_folder_results, current_date


def run_experiment(env_class: Any, 
                  controller: Any, 
                  noise_levels: List[float], 
                  seeds: List[int],
                  experiment_name: str,
                  test_name: str,
                  num_steps: int = 50,
                  **env_kwargs) -> None:
    """
    Runs a complete experiment across multiple noise levels and seeds.

    Args:
        env_class: Environment class to instantiate
        controller: Controller object with compute_action method
        noise_levels: List of noise levels to test
        seeds: List of random seeds to use
        experiment_name: Name of the experiment
        test_name: Name of the test/algorithm
        num_steps: Number of steps per episode
        env_kwargs: Additional keyword arguments for environment initialization
    """
    trajectory_manager = TrajectoryDataManager(experiment_name=experiment_name, test_name=test_name)

    for noise_sigma in noise_levels:
        print(f"\nRunning with noise level: {noise_sigma}")
        # Define the environment with current noise level
        env_kwargs['noise_sigma'] = noise_sigma

        for seed in seeds:
            print(f"  Seed: {seed}")
            trajectory_manager.clear_data()
            env = env_class(**env_kwargs)

            # Reset environment with current seed
            obs, _ = env.reset(seed=seed)

            # Record initial state
            trajectory_manager.add_step_data(
                obs, 
                [np.nan] * env.action_space.shape[-1], 
                env._get_reward(obs)
            )

            # Run episode
            for step in range(num_steps):
                action, info_dict = controller.compute_action(obs)
                obs, reward, done, _, _ = env.step(action)
                trajectory_manager.add_step_data(obs, action, reward)

                if done:
                    print(f"    Episode terminated early at step {step}")
                    break

            # Save trajectory data
            trajectory_manager.save_data(noise_sigma, seed)

    print("\nExperiment completed successfully.")
