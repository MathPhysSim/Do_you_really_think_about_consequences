from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple, Optional, Union

import numpy as np
from gymnasium import spaces


class BaseController(ABC):
    """
    Abstract base class for all controllers.
    Provides a common interface for different controller implementations.
    """

    def __init__(self, observation_space: spaces.Space, action_space: spaces.Space, params_dict: Dict[str, Any]):
        """
        Initialize the controller with spaces and parameters.

        Args:
            observation_space: Environment observation space
            action_space: Environment action space
            params_dict: Dictionary of controller parameters
        """
        self.observation_space = observation_space
        self.action_space = action_space
        self.params_dict = params_dict

        # Extract common parameters
        self.controller_params = params_dict.get("controller", {})
        self.DoF = self.controller_params.get("DoF", observation_space.shape[0])

        # Initialize memory for observations and actions
        self.observations = []
        self.actions = []
        self.rewards = []

    @abstractmethod
    def compute_action(self, obs: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Compute the next action given the current observation.

        Args:
            obs: Current environment observation

        Returns:
            Tuple of (action, info_dict)
        """
        pass

    @abstractmethod
    def compute_cost_unnormalized(self, obs: np.ndarray, action: np.ndarray) -> Tuple[float, float]:
        """
        Compute the unnormalized cost for an observation-action pair.

        Args:
            obs: Environment observation
            action: Action taken

        Returns:
            Tuple of (cost, cost_variance)
        """
        pass

    def add_memory(self, obs: np.ndarray, action: np.ndarray, obs_new: np.ndarray, reward: float, 
                  **kwargs) -> None:
        """
        Add a transition to memory.

        Args:
            obs: Previous observation
            action: Action taken
            obs_new: Resulting observation
            reward: Reward received
            **kwargs: Additional information to store
        """
        self.observations.append((obs, obs_new))
        self.actions.append(action)
        self.rewards.append(reward)

    def reset(self) -> None:
        """
        Reset the controller's internal state.
        """
        self.observations = []
        self.actions = []
        self.rewards = []
