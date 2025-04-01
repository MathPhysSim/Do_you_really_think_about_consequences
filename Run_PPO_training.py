import argparse
from pathlib import Path
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

from awake_steering_simulated import AwakeSteering

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Train PPO with AwakeSteering environment.")
parser.add_argument("--noise_sigma", type=float, default=0.0, help="Noise sigma value for environment (default: 0.0).")
parser.add_argument("--seed", type=int, default=42, help="Random seed for environment (default: 42).")
args = parser.parse_args()

# Create the environment
env = AwakeSteering(noise_sigma=args.noise_sigma, seed=args.seed)

# Define PPO model parameters
ppo_kwargs = {
    "policy": "MlpPolicy",  # Use a multilayer perceptron (MLP) policy
    "env": env,  # Pass the custom environment
    "learning_rate": 3e-4,  # Learning rate for the optimizer
    "n_steps": 2048,  # Number of steps to run for each environment per update
    "batch_size": 64,  # Minibatch size for each update
    "n_epochs": 10,  # Number of epochs to train each update
    "gamma": 0.99,  # Discount factor
    "gae_lambda": 0.95,  # Factor for GAE (Generalized Advantage Estimation)
    "clip_range": 0.2,  # PPO clipping parameter
    "verbose": 1  # Print training info
}

if __name__ == '__main__':
    # Instantiate the PPO model
    model = PPO(**ppo_kwargs)

    # Train the agent
    timesteps = 100000  # Set number of training timesteps
    model.learn(total_timesteps=timesteps)

    # Save the trained model
    model_save_path = Path("PPO_policy") / f"ppo_awake_steering_noise_sigma_{args.noise_sigma}_seed_{args.seed}"
    model.save(model_save_path)
    print(f"Model saved to {model_save_path}")

    # Load and test the trained model
    env = make_vec_env(lambda: AwakeSteering(noise_sigma=args.noise_sigma), n_envs=1)
    loaded_model = PPO.load(model_save_path, env=env)

    # Run a test episode
    obs = env.reset()
    done = False
    while not done:
        action, _states = loaded_model.predict(obs, deterministic=True)
        obs, reward, done, _ = env.step(action)
        print(f"Step Reward: {reward}")

    print("Test episode completed.")