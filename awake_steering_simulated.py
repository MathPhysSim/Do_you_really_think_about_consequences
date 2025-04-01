import logging.config
from typing import Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from matplotlib import pyplot as plt

from helper_scripts.helpers import MamlHelpers, Plane


class AwakeSteering(gym.Env):
    def __init__(self, task=None, train=False, use_absolute_settings: bool = False, **kwargs):
        self.init_kicks = None
        self.__version__ = "0.6"
        logging.info(f"e_trajectory_simENV - Version {self.__version__}")

        self.plane = Plane.horizontal
        self.maml_helper = MamlHelpers(plane=self.plane)

        # General environment settings
        self.MAX_TIME = 1000
        self.threshold = -0.1
        self.train = train
        self.use_absolute_settings = use_absolute_settings  # New flag for absolute mode

        if 'noise_sigma' in kwargs:
            self.noise_sigma = kwargs['noise_sigma']
        else:
            self.noise_sigma = 0.0
        self.current_episode = -1
        self.current_steps = 0
        self.nr_init_trials_to_find_a_good_setting = 10000



        # Initialize state and action spaces
        self.positions = np.zeros(len(self.maml_helper.twiss_bpms) - 1)
        self.settings = np.zeros(len(self.maml_helper.twiss_correctors) - 1)

        self.action_space = spaces.Box(low=-np.ones(len(self.settings)),
                                       high=np.ones(len(self.settings)),
                                       dtype=np.float32)

        self.observation_space = spaces.Box(low=-np.ones(len(self.positions)),
                                            high=np.ones(len(self.positions)),
                                            dtype=np.float32)
        # Initialize task
        self.reset_task(task if task else self.maml_helper.get_origin_task())

        # Set the seed for reproducibility
        if 'seed' in kwargs:
            print('seed', kwargs['seed'])
        self.seed(kwargs.get('seed'))

        self.DoF = None

    def add_noise(self, state_new):
        # 5) Optionally add noise
            # by default np.random.randn() is float64, you can convert to float32 if needed
        state_new += np.random.randn(*self.observation_space.shape) * self.noise_sigma
        np.clip(state_new, -1.0, 1.0, out=state_new)
        return state_new

    def set_use_absolute_settings(self, use_absolute_settings):
        self.use_absolute_settings = use_absolute_settings
        if use_absolute_settings:
            self.previous_action = np.zeros(self.action_space.shape)

    def step(self, action):
        """Executes a step in the environment given an action."""
        self.current_steps += 1

        # In absolute mode, convert the absolute state command into an incremental action.
        if self.use_absolute_settings:
            # Here, action is the desired absolute state.
            previous_action = self.previous_action
            incremental_action = action - previous_action
            self.previous_action = action
            action = incremental_action


        # 1) Normalize the action if max(|a|) > 1, then clip once in-place
        max_abs_action = np.max(np.abs(action))
        if max_abs_action > 1.0:
            action /= max_abs_action
        # np.clip(action, -1.0, 1.0, out=action)

        # 2) Take the step
            # 1) Compute next state
        state_new = self.response @ action + self.state
            # 2) Compute reward with optimized function
        reward = self._get_reward(state_new)
            # 3) Check if any coordinate violates ±1 in one pass
        abs_state = np.abs(state_new)
        max_abs_state = np.max(abs_state)
        if max_abs_state >= 1.0:
            violation_pos = np.argmax(abs_state >= 1.0)
            sign_val = np.sign(state_new[violation_pos])
            state_new[violation_pos:] = sign_val

            # 4) Update internal state
        self.state = state_new
        if self.noise_sigma != 0.0:
            state_new = self.add_noise(state_new)

        # Instead of np.any(np.abs(return_state) >= 1),
        # use np.max(np.abs(...)) once

        is_truncated = (self.current_steps >= self.MAX_TIME)
        is_finalized =  self.check_threshold_condition(self.state)
        if is_finalized:
            print(f'finalized')
        return (
            state_new,
            reward,
            is_finalized,
            is_truncated,
            {"task": self._id, "time": self.current_steps}
        )

    def _get_reward(self, state):
        # Same RMS logic, just a slight tweak in how we compute squares
        return -np.sqrt(np.mean(state * state))

    def check_threshold_condition(self, state):
        is_finalized = (
                self._get_reward(state) > self.threshold
                or self.current_steps >= self.MAX_TIME
                or np.max(np.abs(state)) >= 1.0
        )
        return is_finalized


    def find_good_initialisation(self):
        """Finds a good initial state within defined thresholds."""
        for _ in range(self.nr_init_trials_to_find_a_good_setting):
            self.init_kicks = self.action_space.sample() * 10
            init_state = self.response @ self.init_kicks
            # init_state = self.observation_space.sample()
            # Check constraints once
            if np.all(np.abs(init_state) <= 1.0):# and self._get_reward(init_state):# < 3.0 * self.threshold:
                break

        return init_state

    def reset(self, seed: Optional[int] = None, **kwargs):
        if seed is not None:
            self.seed(seed)

        self.current_episode += 1
        self.current_steps = 0

        if 'init_state' in kwargs:
            init_state = kwargs['init_state']
        else:
            init_state = self.find_good_initialisation()

        self.state = init_state

        if self.use_absolute_settings:
            self.initial_state = init_state
            self.previous_action = np.zeros(self.action_space.shape)

        if self.noise_sigma != 0.0:
            init_state = self.add_noise(init_state)
        return init_state, {}

    def seed(self, seed=None):
        print('seed function with seed: ', seed)
        # self.np_random, seed = seeding.np_random(seed)
        np.random.seed(seed)
        # random.seed(seed)
        self.action_space.seed(seed)
        self.observation_space.seed(seed)

    def sample_tasks(self, num_tasks):
        return self.maml_helper.sample_tasks(num_tasks)

    def reset_task(self, task):
        self._task = task
        self._goal = task["goal"]
        self._id = task["id"]
        self.response = self._goal

    def get_current_state(self):
        return self.state



# def plot_trajectories(episode_rewards, all_episodes_states, all_episodes_actions):
#     # ---------------------------
#     # Plotting
#     # ---------------------------
#     plt.figure(figsize=(12, 8))
#
#     # -------------------------------------------------------
#     # 1) Plot total reward per episode
#     # -------------------------------------------------------
#     plt.subplot(3, 1, 1)
#     plt.plot(episode_rewards, marker='o', linestyle='-', color='b')
#     plt.title("Total Reward per Episode")
#     plt.xlabel("Episode")
#     plt.ylabel("Total Reward")
#     plt.grid(True)
#
#     # -------------------------------------------------------
#     # 2) Plot the state trajectories for ALL episodes
#     #    (with gaps & vertical lines separating episodes)
#     # -------------------------------------------------------
#     plt.subplot(3, 1, 2)
#
#     # Build a single x-array and y-array for each state dimension
#     # so we can see them all in one plot (with breaks).
#     # We'll insert NaNs to create the "gap" in the line between episodes.
#     if len(all_episodes_states) > 0:
#         # Number of state dimensions
#         n_state_dims = len(all_episodes_states[0][0])  # based on the first state
#
#         # We loop over each dimension and build a single line
#         for dim in range(n_state_dims):
#             dim_x = []
#             dim_y = []
#             current_x = 0
#
#             # Go through each episode
#             for e, episode_states in enumerate(all_episodes_states):
#                 # Draw a vertical line to mark the start of each new episode (if you prefer a horizontal line, see note below)
#                 plt.axvline(current_x, color='gray', linestyle='--', alpha=0.5)
#
#                 for s in episode_states:
#                     dim_x.append(current_x)
#                     dim_y.append(s[dim])
#                     current_x += 1
#
#                 # Insert a gap (NaN) after each episode to break the line
#                 dim_x.append(current_x)
#                 dim_y.append(np.nan)
#                 current_x += 1  # Move x a bit further to create the gap
#
#             # Now plot the dimension
#             plt.plot(dim_x, dim_y, label=f"State dim {dim}", marker='o')
#
#         plt.title("States Over Steps (All Episodes)")
#         plt.xlabel("Step (with gaps between episodes)")
#         plt.ylabel("States (1)")
#         plt.legend()
#         plt.grid(True)
#
#         # If you actually want a HORIZONTAL line at the beginning of each new episode
#         # rather than a vertical line, you could do something like:
#         #
#         #   y_start = 0  # or some other reference
#         #   x_start = current_x_for_episode
#         #   x_end = current_x_for_episode + <some small offset>
#         #   plt.hlines(y_start, x_start, x_end, color='gray', linestyles='--')
#         #
#         # but typically a vertical line is used to show a new "time" boundary.
#
#     # -------------------------------------------------------
#     # 3) Plot the action trajectories for ALL episodes
#     #    (with gaps & vertical lines separating episodes)
#     # -------------------------------------------------------
#     plt.subplot(3, 1, 3)
#
#     if len(all_episodes_actions) > 0:
#         # Number of action dimensions
#         n_action_dims = len(all_episodes_actions[0][0])  # based on the first action
#
#         for dim in range(n_action_dims):
#             dim_x = []
#             dim_y = []
#             current_x = 0
#
#             for e, episode_actions in enumerate(all_episodes_actions):
#                 # Mark the start of the episode
#                 plt.axvline(current_x, color='gray', linestyle='--', alpha=0.5)
#
#                 for a in episode_actions:
#                     dim_x.append(current_x)
#                     dim_y.append(a[dim])
#                     current_x += 1
#
#                 # Insert gap
#                 dim_x.append(current_x)
#                 dim_y.append(np.nan)
#                 current_x += 1
#
#             plt.plot(dim_x, dim_y, label=f"Action dim {dim}", marker='o')
#
#         plt.title("Actions Over Steps (All Episodes)")
#         plt.xlabel("Step (with gaps between episodes)")
#         plt.ylabel("Actions (1)")
#         plt.legend()
#         plt.grid(True)
#
#     plt.tight_layout()
#     plt.show()
def plot_trajectories(episode_rewards_per_step, all_episodes_states, all_episodes_actions):
    # ---------------------------
    # Plotting
    # ---------------------------
    plt.figure(figsize=(12, 8))

    # -------------------------------------------------------
    # 1) Plot reward per step
    # -------------------------------------------------------
    plt.subplot(3, 1, 1)
    plt.plot(episode_rewards_per_step[0], marker='o', linestyle='-', color='b')
    plt.title("Reward per Step")
    plt.xlabel("Step")
    plt.ylabel("Reward")
    plt.grid(True)

    # -------------------------------------------------------
    # 2) Plot the state trajectories for ALL episodes
    #    (with gaps & vertical lines separating episodes)
    # -------------------------------------------------------
    plt.subplot(3, 1, 2)

    if len(all_episodes_states) > 0:
        n_state_dims = len(all_episodes_states[0][0])

        for dim in range(n_state_dims):
            dim_x = []
            dim_y = []
            current_x = 0

            for e, episode_states in enumerate(all_episodes_states):
                plt.axvline(current_x, color='gray', linestyle='--', alpha=0.5)
                for s in episode_states:
                    dim_x.append(current_x)
                    dim_y.append(s[dim])
                    current_x += 1
                dim_x.append(current_x)
                dim_y.append(np.nan)
                current_x += 1

            plt.plot(dim_x, dim_y, label=f"State dim {dim}", marker='o')

        plt.title("States Over Steps (All Episodes)")
        plt.xlabel("Step")
        plt.ylabel("States")
        plt.legend()
        plt.grid(True)

    # -------------------------------------------------------
    # 3) Plot the action trajectories for ALL episodes
    # -------------------------------------------------------
    plt.subplot(3, 1, 3)

    if len(all_episodes_actions) > 0:
        n_action_dims = len(all_episodes_actions[0][0])

        for dim in range(n_action_dims):
            dim_x = []
            dim_y = []
            current_x = 0

            for e, episode_actions in enumerate(all_episodes_actions):
                plt.axvline(current_x, color='gray', linestyle='--', alpha=0.5)
                for a in episode_actions:
                    dim_x.append(current_x)
                    dim_y.append(a[dim])
                    current_x += 1
                dim_x.append(current_x)
                dim_y.append(np.nan)
                current_x += 1

            plt.plot(dim_x, dim_y, label=f"Action dim {dim}", marker='o')

        plt.title("Actions Over Steps (All Episodes)")
        plt.xlabel("Step")
        plt.ylabel("Actions")
        plt.legend()
        plt.grid(True)

    plt.tight_layout()
    plt.show()
def run_policy(env, policy, num_episodes = 5, max_steps_per_episode = 50):
    # Lists to store episode-level info
    all_episodes_rewards = []
    all_episodes_states = []
    all_episodes_actions = []

    for episode in range(num_episodes):
        obs, info = env.reset()
        done = False
        truncated = False
        total_reward = 0.0
        step_count = 0

        # Temporary storage for this episode
        episode_states = []
        episode_actions = []

        while not done:# and not truncated and step_count < max_steps_per_episode:
            episode_states.append(obs)
            action = policy(obs)
            episode_actions.append(action)

            # Step the environment
            next_obs, reward, done, truncated, info = env.step(action)
            total_reward += reward

            obs = next_obs
            step_count += 1

            if step_count > max_steps_per_episode:
                break

        episode_states.append(obs)
        # Store results for this episode
        all_episodes_rewards.append(total_reward)
        all_episodes_states.append(episode_states)
        all_episodes_actions.append(episode_actions)

        print(f"[Episode {episode + 1}] Steps: {step_count}, Total reward: {total_reward:.3f}")

    return all_episodes_rewards, all_episodes_states, all_episodes_actions


# -----------------------------------------------------------
# Example script with plotting
# -----------------------------------------------------------
def main():
    # Create an instance of the environment
    seed = 1
    np.random.seed(seed)
    noise_sigma = 0.0
    env = AwakeSteering(seed=seed, noise_sigma=noise_sigma, use_absolute_settings=False)
    num_episodes = 5
    max_steps_per_episode = 50
    policy = lambda a: np.zeros(env.action_space.shape)+.1
    all_episodes_rewards, all_episodes_states, all_episodes_actions = run_policy(env, policy,
                                                                            num_episodes=num_episodes,
                                                                            max_steps_per_episode=max_steps_per_episode)

    plot_trajectories(all_episodes_rewards, all_episodes_states, all_episodes_actions)


if __name__ == "__main__":
    main()



