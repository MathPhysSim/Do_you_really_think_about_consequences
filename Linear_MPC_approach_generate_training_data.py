import os
import pickle
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np

from awake_steering_simulated import AwakeSteering
from environment.environment_helpers import read_experiment_config, load_env_config, \
    SmartEpisodeTrackerWithPlottingWrapper
from helper_scripts.gp_mpc_structured_clean import init_control
from helper_scripts.linear_Bayesian_mpc import LinearMPCController, init_visu_and_folders, logger, close_run
# from helper_scripts.linear_data_driven_mpc import LinearMPCController, init_visu_and_folders, logger, close_run

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

# Example main function (assuming environment wrappers and configuration are available)
def init_graphics_and_controller(env: Any, num_steps: int, params_controller_dict: Dict) -> Tuple[
    Any, LinearMPCController]:
    live_plot_obj = init_visu_and_folders(env, num_steps, params_controller_dict)
    ctrl_obj = LinearMPCController(
        observation_space=env.observation_space,
        action_space=env.action_space,
        params_dict=params_controller_dict,
        env=env
    )
    return live_plot_obj, ctrl_obj


def adjust_params_for_DoF(params_controller_dict: Dict, DoF: int) -> None:
    for var in ["gp_init"]:
        for key in params_controller_dict[var]:
            params_controller_dict[var][key] = params_controller_dict[var][key][:DoF]
    controller_keys = [
        "target_state_norm",
        "weight_state",
        "weight_state_terminal",
        "target_action_norm",
        "weight_action",
        "obs_var_norm",
    ]
    for key in controller_keys:
        params_controller_dict["controller"][key] = params_controller_dict["controller"][key][:DoF]
    params_controller_dict["controller"]["DoF"] = DoF




def main() -> None:
    params_controller_dict = read_experiment_config("config/data_driven_mpc_config.yaml")
    num_steps = 50
    training_steps = num_steps
    random_actions_init = params_controller_dict.get("random_actions_init", 0)
    # env = load_env_config(env_config="config/environment_setting.yaml")
    env = AwakeSteering()
    DoF = env.DoF
    adjust_params_for_DoF(params_controller_dict, DoF)
    env = SmartEpisodeTrackerWithPlottingWrapper(env)
    live_plot_obj, ctrl_obj = init_graphics_and_controller(env, num_steps, params_controller_dict)
    ctrl_obj, env, live_plot_obj, obs, action, cost, obs_prev_ctrl, obs_lst, actions_lst, rewards_lst = init_control(
        ctrl_obj, env, live_plot_obj, random_actions_init
    )

    info_dict = None
    done = False
    for step in range(random_actions_init, num_steps):
        t0 = time.time()
        if step % 1 == 0:
            if step < training_steps:
                ctrl_obj.add_memory(obs=obs_prev_ctrl, action=action, obs_new=obs, reward=-cost)
            if done:
                obs, _ = env.reset()
            action, info_dict = ctrl_obj.compute_action(obs)
            logger.info(f"Step {step}, computed action: {action}, info: {info_dict}")
        obs_new, reward, done, _, _ = env.step(action)
        cost, _ = ctrl_obj.compute_cost_unnormalized(obs, action)
        live_plot_obj.update(obs=obs, action=action, cost=cost, info_dict=info_dict)
        obs_prev_ctrl = obs
        obs = obs_new
        logger.debug(f"Loop time: {time.time()-t0:.4f} s")
    close_run(ctrl_obj=ctrl_obj, env=env)

# here we generate the data

    test_name = 'LinearMPC'
    experiment_name = 'noise_test'
    num_steps = 50
    noise_sigma_list = [0, 0.001, 0.01, 0.025, 0.05, 0.1]
    seed_list = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    trajectory_data_manager = TrajectoryDataManager(experiment_name=experiment_name, test_name=test_name)

    for noise_sigma in noise_sigma_list:
        # Define the environment and parameters
        env = AwakeSteering(noise_sigma=noise_sigma)#, seed=seed)
        for seed in seed_list:
            trajectory_data_manager.clear_data()
            obs, _ = env.reset(seed=seed)
            trajectory_data_manager.add_step_data(obs, [np.nan] * env.action_space.shape[-1], [env._get_reward(obs)])
            for _ in range(num_steps):
                action, info_dict = ctrl_obj.compute_action(obs)
                print(f'action {action}')
                obs, reward, done, _, _ = env.step(action)
                trajectory_data_manager.add_step_data(obs, action, [reward])
                if done:
                    break
            trajectory_data_manager.save_data(noise_sigma, seed)


if __name__ == "__main__":
    main()