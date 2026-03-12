"""
Shared data management utilities for experiment trajectory recording and persistence.
"""
import os
import pickle
from datetime import datetime
from pathlib import Path
from typing import Tuple

import numpy as np


def create_experiment_folder(experiment_name: str, algorithm: str, parameter_name: str) -> Tuple[Path, str]:
    """
    Creates a structured experiment folder based on the experiment name, algorithm, and parameter name.

    Args:
        experiment_name: Name of the experiment.
        algorithm: Name of the algorithm used.
        parameter_name: Parameter name associated with the experiment.

    Returns:
        Tuple of (Path to the created experiment folder, current date string).
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
    """Manages trajectory data collection (states, actions, rewards) and persistence."""

    def __init__(self, experiment_name: str, test_name: str):
        self.experiment_name = experiment_name
        self.test_name = test_name

        self.state_history: list = []
        self.action_history: list = []
        self.reward_history: list = []

    def add_step_data(self, state, action, reward):
        """Record a single step's state, action, and reward."""
        self.action_history.append(action)
        self.state_history.append(state)
        self.reward_history.append(reward)

    def clear_data(self):
        """Clear all recorded trajectory data."""
        self.state_history = []
        self.action_history = []
        self.reward_history = []

    def get_data(self):
        """Return collected data as numpy arrays."""
        return np.array(self.state_history), np.array(self.action_history), np.array(self.reward_history)

    def save_data(self, noise_sigma, seed):
        """Save recorded trajectory data to a pickle file."""
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
