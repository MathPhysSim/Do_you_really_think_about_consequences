import time
import logging
from typing import Any, Tuple, Dict

import matplotlib
import numpy as np

from Linear_MPC_approach_generate_training_data import TrajectoryDataManager
from awake_steering_simulated import AwakeSteering

matplotlib.use('TkAgg')  # Force the TkAgg backend for external windows

from helper_scripts.gp_mpc_structured_clean import GpMpcController, init_control, init_visu_and_folders, close_run
from environment.environment_helpers import (
    read_experiment_config,
    load_env_config,
    RewardScalingWrapper,
    SmartEpisodeTrackerWithPlottingWrapper,
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def init_graphics_and_controller(
    env: Any, num_steps: int, params_controller_dict: Dict
) -> Tuple[Any, GpMpcController]:
    """
    Initialize and return the live plot object and GP-MPC controller.

    Args:
        env: The environment instance.
        num_steps: Total number of environment steps.
        params_controller_dict: Dictionary of controller parameters.

    Returns:
        Tuple containing the live plot object and the controller.
    """
    live_plot_obj = init_visu_and_folders(env, num_steps, params_controller_dict)
    ctrl_obj = GpMpcController(
        observation_space=env.observation_space,
        action_space=env.action_space,
        params_dict=params_controller_dict,
    )
    return live_plot_obj, ctrl_obj


def adjust_params_for_DoF(params_controller_dict: Dict, DoF: int) -> None:
    """
    Adjust the controller parameters to match the environment's degrees of freedom (DoF).

    Args:
        params_controller_dict: Dictionary containing controller parameters.
        DoF: Degrees of freedom from the environment.
    """
    # Update GP initialization parameters
    for var in ["gp_init"]:
        for key in params_controller_dict[var]:
            params_controller_dict[var][key] = params_controller_dict[var][key][:DoF]
            logger.debug(f"{var} {key}: {params_controller_dict[var][key]}")

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
        logger.debug(f"controller {key}: {params_controller_dict['controller'][key]}")

    params_controller_dict["controller"]["DoF"] = DoF


def main() -> None:
    """
    Main function to run the GP-MPC controller with the environment.
    """
    params_controller_dict = read_experiment_config("config/data_driven_mpc_config.yaml")
    num_steps = 75
    step_training = num_steps
    num_repeat_actions = params_controller_dict["controller"].get("num_repeat_actions", 1)
    random_actions_init = params_controller_dict.get("random_actions_init", 0)

    # env = load_env_config(env_config="config/environment_setting.yaml")
    env = AwakeSteering(noise_sigma=0.01)
    DoF = 10#env.DoF

    adjust_params_for_DoF(params_controller_dict, DoF)

    # Optionally wrap the environment with additional trackers/wrappers
    env = SmartEpisodeTrackerWithPlottingWrapper(env)

    live_plot_obj, ctrl_obj = init_graphics_and_controller(env, num_steps, params_controller_dict)

    (
        ctrl_obj,
        env,
        live_plot_obj,
        obs,
        action,
        cost,
        obs_prev_ctrl,
        obs_lst,
        actions_lst,
        rewards_lst,
    ) = init_control(
        ctrl_obj=ctrl_obj,
        env=env,
        live_plot_obj=live_plot_obj,
        random_actions_init=random_actions_init,
        num_repeat_actions=num_repeat_actions,
    )

    info_dict = None
    done = False
    for step in range(random_actions_init, num_steps):
        start_time = time.time()

        if step % num_repeat_actions == 0:
            # Retrieve prediction info if available
            if info_dict is not None:
                predicted_state = info_dict.get("predicted states", [None])[0]
                predicted_state_std = info_dict.get("predicted states std", [None])[0]
            else:
                predicted_state = predicted_state_std = None

            if step < step_training:
                # Add memory before computing the next action
                ctrl_obj.add_memory(
                    obs=obs_prev_ctrl,
                    action=action,
                    obs_new=obs,
                    reward=-cost,
                    predicted_state=predicted_state,
                    predicted_state_std=predicted_state_std,
                )

            if done:
                obs, _ = env.reset()

            # Compute the next action
            action, info_dict = ctrl_obj.compute_action(obs_mu=obs)

            if params_controller_dict.get("verbose", False):
                for key, value in info_dict.items():
                    logger.info(f"{key}: {value}")

        # Execute action in the environment
        obs_new, reward, done, _, _ = env.step(action)
        cost, cost_var = ctrl_obj.compute_cost_unnormalized(obs, action)

        # Update visualization if enabled
        try:
            if live_plot_obj is not None:
                live_plot_obj.update(obs=obs, cost=cost, action=action, info_dict=info_dict)
        except Exception as e:
            logger.error(f"Error updating live plot: {e}")

        obs_prev_ctrl = obs
        obs = obs_new

        logger.debug(f"Time loop: {time.time() - start_time:.4f} s")

    # Clean up resources
    close_run(ctrl_obj=ctrl_obj, env=env)

# here we generate the data

    test_name = 'Structured_MPC_2'
    experiment_name = 'noise_test'
    num_steps = 50
    noise_sigma_list = [0, 0.001, 0.01, 0.025, 0.05, 0.1]
    # noise_sigma_list = [0.05]
    seed_list = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    trajectory_data_manager = TrajectoryDataManager(experiment_name=experiment_name, test_name=test_name)

    for noise_sigma in noise_sigma_list:
        # Define the environment and parameters
        env = AwakeSteering(noise_sigma=noise_sigma)  # , seed=seed)
        print(f'starting experiment with noise sigma {noise_sigma}')
        for seed in seed_list:
            trajectory_data_manager.clear_data()
            obs, _ = env.reset(seed=seed)
            trajectory_data_manager.add_step_data(obs, [np.nan] * env.action_space.shape[-1], [env._get_reward(obs)])
            for _ in range(num_steps):
                action, info_dict = ctrl_obj.compute_action(obs)
                obs, reward, done, _, _ = env.step(action)
                trajectory_data_manager.add_step_data(obs, action, [reward])
                if done:
                    break
            trajectory_data_manager.save_data(noise_sigma, seed)


if __name__ == "__main__":
    main()