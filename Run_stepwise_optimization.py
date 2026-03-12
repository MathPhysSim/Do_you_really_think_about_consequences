import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import minimize

from awake_steering_simulated import AwakeSteering
from helper_scripts.data_management import TrajectoryDataManager


def get_bounds(current_state):
    """Return action bounds based on the current state dimension."""
    action_dim = len(current_state)
    return [(-1, 1)] * action_dim


def iterative_optimization(env, trajectory_data_manager, max_steps=50):
    """Run iterative stepwise optimization using COBYLA."""
    obs, info = env.reset()
    episode_states = [obs]
    episode_actions = []
    episode_rewards = [env._get_reward(obs)]

    trajectory_data_manager.add_step_data(obs, [np.nan] * env.action_space.shape[-1], [env._get_reward(obs)])

    action = np.zeros(env.action_space.shape)
    stop_optimization = env.check_threshold_condition(obs)

    for step in range(max_steps):
        if stop_optimization:
            print(f"Optimization stopped early at step {step} due to termination condition.")
            break

        bounds = get_bounds(env.state)

        def optimize_step(action):
            """Objective function: runs the step and returns negative reward."""
            nonlocal obs, stop_optimization, step
            if not stop_optimization:
                episode_actions.append(action)
                next_obs, reward, done, truncated, info = env.step(action)
                step += 1
                obs = next_obs
                episode_states.append(next_obs)
                episode_rewards.append(reward)
                trajectory_data_manager.add_step_data(obs, action, [reward])

                if done or truncated:
                    print(f"Early stopping at step {step + 1}: done={done}, truncated={truncated}")
                    stop_optimization = True
                    return 1e6
            else:
                return 1e6
            return -reward

        env.set_use_absolute_settings(True)
        minimize(optimize_step, x0=action, method='COBYLA', bounds=bounds, tol=1e-1, options={'disp': False})
        env.set_use_absolute_settings(False)

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
    seed_list = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    experiment_name = 'noise_test'
    test_name = 'Classical'
    trajectory_data_manager = TrajectoryDataManager(experiment_name=experiment_name, test_name=test_name)

    for noise_sigma in noise_sigma_list:
        for seed in seed_list:
            trajectory_data_manager.clear_data()
            env = AwakeSteering(seed=seed, noise_sigma=noise_sigma, use_absolute_settings=False)
            episode_states, episode_actions, episode_rewards = iterative_optimization(
                env, trajectory_data_manager, max_steps=num_steps
            )
            trajectory_data_manager.save_data(noise_sigma, seed)
