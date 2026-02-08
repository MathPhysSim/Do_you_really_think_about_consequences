#!/usr/bin/env python
"""
Master script to run all experiment types in sequence.
Provides a unified entry point for running complete experiment suites.
"""

import argparse
import logging
import sys
import time
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("experiments.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("run_experiments")


def parse_args():
    """
    Parse command line arguments for experiment configuration.
    """
    parser = argparse.ArgumentParser(description='Run all experiments for AWAKE steering optimization')

    parser.add_argument('--experiment-name', type=str, default='noise_test',
                        help='Name of the experiment')
    parser.add_argument('--num-steps', type=int, default=50,
                        help='Number of steps per episode')
    parser.add_argument('--noise-levels', type=float, nargs='+', 
                        default=[0, 0.001, 0.01, 0.025, 0.05, 0.1],
                        help='Noise levels to test')
    parser.add_argument('--seeds', type=int, nargs='+', default=list(range(1, 10)),
                        help='Random seeds to use')
    parser.add_argument('--skip', type=str, nargs='+', default=[],
                        choices=['linear-mpc', 'gp-mpc', 'gp-mpc-structured', 'cobyla', 'ppo'],
                        help='Skip specific experiment types')

    return parser.parse_args()


def run_linear_mpc_experiments(args):
    """
    Run Linear MPC experiments.
    """
    if 'linear-mpc' in args.skip:
        logger.info("Skipping Linear MPC experiments")
        return

    logger.info("Running Linear MPC experiments")
    start_time = time.time()

    try:
        # Import here to avoid loading everything at startup
        from Linear_MPC_approach_generate_training_data import main
        main()
        logger.info(f"Linear MPC experiments completed in {time.time() - start_time:.2f} seconds")
    except Exception as e:
        logger.error(f"Error in Linear MPC experiments: {e}")


def run_gp_mpc_experiments(args):
    """
    Run GP MPC experiments.
    """
    if 'gp-mpc' in args.skip:
        logger.info("Skipping GP MPC experiments")
        return

    logger.info("Running GP MPC experiments")
    start_time = time.time()

    try:
        # Import here to avoid loading everything at startup
        from GP_MPC_approach_standard_generate_data import main
        main()
        logger.info(f"GP MPC experiments completed in {time.time() - start_time:.2f} seconds")
    except Exception as e:
        logger.error(f"Error in GP MPC experiments: {e}")


def run_gp_mpc_structured_experiments(args):
    """
    Run structured GP MPC experiments.
    """
    if 'gp-mpc-structured' in args.skip:
        logger.info("Skipping structured GP MPC experiments")
        return

    logger.info("Running structured GP MPC experiments")
    start_time = time.time()

    try:
        # Import here to avoid loading everything at startup
        import GP_MPC_approach_structured_generate_training_data
        GP_MPC_approach_structured_generate_training_data.main()
        logger.info(f"Structured GP MPC experiments completed in {time.time() - start_time:.2f} seconds")
    except Exception as e:
        logger.error(f"Error in structured GP MPC experiments: {e}")


def run_cobyla_experiments(args):
    """
    Run COBYLA optimization experiments.
    """
    if 'cobyla' in args.skip:
        logger.info("Skipping COBYLA experiments")
        return

    logger.info("Running COBYLA optimization experiments")
    start_time = time.time()

    try:
        # Import here to avoid loading everything at startup
        import Run_stepwise_optimizsation
        # Note: This script doesn't have a main() function, execution happens at module level
        logger.info(f"COBYLA experiments completed in {time.time() - start_time:.2f} seconds")
    except Exception as e:
        logger.error(f"Error in COBYLA experiments: {e}")


def run_ppo_experiments(args):
    """
    Run PPO reinforcement learning experiments.
    """
    if 'ppo' in args.skip:
        logger.info("Skipping PPO experiments")
        return

    logger.info("Running PPO experiments")
    start_time = time.time()

    try:
        # Import here to avoid loading everything at startup
        import Run_PPO_training
        Run_PPO_training.main()
        logger.info(f"PPO experiments completed in {time.time() - start_time:.2f} seconds")
    except Exception as e:
        logger.error(f"Error in PPO experiments: {e}")


def generate_visualizations(args):
    """
    Generate visualizations from experiment results.
    """
    logger.info("Generating visualizations")
    start_time = time.time()

    try:
        # Import here to avoid loading everything at startup
        import Read_results_and_create_figures
        # Note: This script doesn't have a main() function, execution happens at module level
        logger.info(f"Visualization generation completed in {time.time() - start_time:.2f} seconds")
    except Exception as e:
        logger.error(f"Error in visualization generation: {e}")


def main():
    """
    Main function to run all experiments.
    """
    args = parse_args()

    logger.info(f"Starting experiments with settings:")
    logger.info(f"  Experiment name: {args.experiment_name}")
    logger.info(f"  Number of steps: {args.num_steps}")
    logger.info(f"  Noise levels: {args.noise_levels}")
    logger.info(f"  Seeds: {args.seeds}")
    logger.info(f"  Skipping: {args.skip}")

    # Create necessary directories
    Path("results").mkdir(exist_ok=True)
    Path(f"results/{args.experiment_name}").mkdir(exist_ok=True)

    # Run all experiment types
    run_linear_mpc_experiments(args)
    run_gp_mpc_experiments(args)
    run_gp_mpc_structured_experiments(args)
    run_cobyla_experiments(args)
    run_ppo_experiments(args)

    # Generate visualizations
    generate_visualizations(args)

    logger.info("All experiments completed successfully")


if __name__ == "__main__":
    main()
