import pickle
from pathlib import Path
from typing import List, Dict, Optional, Union, Tuple

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def load_experiment_data(base_dir_root: Path) -> Optional[pd.DataFrame]:
    """
    Load experiment results from stored pickle files.

    Args:
        base_dir_root: Base directory where results are stored.

    Returns:
        Combined DataFrame with all loaded results, or None if no data found.
    """
    data_all_noises = []

    # Auto-detect available noise levels by checking subdirectories
    noise_levels = sorted(
        [d.name.split("_")[-1] for d in base_dir_root.iterdir() if d.is_dir() and d.name.startswith("noise_sigma_")],
        key=float
    )
    print(f'Found noise levels: {noise_levels}')

    if not noise_levels:
        print("No noise level directories found.")
        return None

    for noise_sigma in noise_levels:
        base_dir = base_dir_root / f"noise_sigma_{noise_sigma}"

        if not base_dir.exists():
            print(f"Directory does not exist: {base_dir}")
            continue

        df_all_seeds = []

        for file in base_dir.glob("*.pkl"):  # Process .pkl files only
            with open(file, "rb") as f:
                data = pickle.load(f)

                dfs = []
                for key in data:
                    columns = [f"{key}_{i}" for i in range(data[key].shape[-1])]
                    df = pd.DataFrame(data[key], columns=columns).T
                    dfs.append(df)

                df_episode = pd.concat(dfs).T
                seed = file.stem  # Extract seed from filename
                df_episode["Seed"] = seed
                df_episode["Time Step"] = df_episode.index

            df_all_seeds.append(df_episode)

        if df_all_seeds:
            df_all_seeds = pd.concat(df_all_seeds, ignore_index=True)
            df_all_seeds["Noise Sigma"] = float(noise_sigma)
            data_all_noises.append(df_all_seeds)

    if data_all_noises:
        return pd.concat(data_all_noises, ignore_index=True)
    else:
        print("No data found for the specified noise levels.")
        return None


def load_all_tests(base_dir: Path, results_date: str, experiment_name: str, 
                   exclude_tests: Optional[List[str]] = None) -> Optional[pd.DataFrame]:
    """
    Automatically detects and loads data for all tests (methods) in the results directory.

    Args:
        base_dir: Base directory containing results
        results_date: Date string for the results folder
        experiment_name: Name of the experiment
        exclude_tests: List of test names to exclude

    Returns:
        Combined DataFrame with data from all tests, or None if no data found
    """
    base_dir_root = base_dir / experiment_name / f'Results_{results_date}'
    all_tests_data = []

    # Default empty list if None provided
    exclude_tests = exclude_tests or []

    # Automatically detect test names by checking subdirectories
    test_names = [d.name for d in base_dir_root.iterdir() if d.is_dir()]
    print(f"Detected test names: {test_names}")

    # Filter out excluded tests
    test_names = [test_name for test_name in test_names if test_name not in exclude_tests]

    for test_name in test_names:
        test_dir = base_dir_root / test_name
        if test_dir.exists():
            data = load_experiment_data(test_dir)
            if data is not None:
                data['algorithm'] = map_test_name(test_name)
                all_tests_data.append(data)

    if all_tests_data:
        return pd.concat(all_tests_data, ignore_index=True)
    else:
        print("No data found for any tests.")
        return None


def map_test_name(test_name: str) -> str:
    """
    Maps internal test names to more descriptive names for visualization.

    Args:
        test_name: Internal test name

    Returns:
        User-friendly test name
    """
    name_mapping = {
        'Classical': 'Model-free stepwise optimisation (COBYLA)',
        'GP_MPC_1': 'Data-driven GP based MPC - generic',
        'GP_MPC_2': 'Data-driven GP based MPC - enhanced',
        'GP_MPC_3': 'Data-driven GP based MPC - v3',
        'LinearMPC': 'Data-driven Linear Bayesian based MPC',
        'MPC': 'Model-based MPC - perfect model',
        'MPC_short': 'Model-based stepwise optimisation - perfect model',
        'Structured_MPC': 'Data-driven GP based MPC respecting causality',
        'PPO': 'PPO'
    }

    return name_mapping.get(test_name, test_name)


def filter_after_threshold(df: pd.DataFrame, threshold: float = -0.1) -> pd.DataFrame:
    """
    For each trajectory (grouped by 'Seed'), remove the data point that occurs
    one time step after the reward first falls below the given threshold.

    Args:
        df: DataFrame containing the trajectory data.
        threshold: The threshold value for reward_0.

    Returns:
        The filtered DataFrame.
    """
    def filter_trajectory(traj):
        # Sort trajectory by time step
        traj = traj.sort_values("Time Step")
        # Find time steps where the threshold is surpassed (reward goes below threshold)
        crossing_times = traj[traj["reward_0"] < threshold]["Time Step"].unique()
        # Identify time steps to remove: each crossing time plus one
        times_to_remove = set(t + 1 for t in crossing_times)
        # Filter out the rows with those time steps
        return traj[~traj["Time Step"].isin(times_to_remove)]

    # Apply the filtering function to each trajectory (grouped by Seed)
    return df.groupby("Seed", group_keys=False).apply(filter_trajectory)


def plot_reward_evolution_all_noises_combined(df: pd.DataFrame, 
                                            noise_sigmas: List[float],
                                            figure_name: str = 'combined_plot.pdf',
                                            apply_filter: bool = False,
                                            threshold: float = -0.1,
                                            selected_algorithms: Optional[List[str]] = None,
                                            xlim_max: int = 80,
                                            **kwargs) -> None:
    """
    Plot reward evolution over time for all noise sigma levels in one combined figure with subplots.

    Args:
        df: The combined DataFrame containing experiment data
        noise_sigmas: List of unique noise sigma levels to plot
        figure_name: Output figure filename
        apply_filter: Whether to apply reward threshold filtering
        threshold: Reward threshold for filtering
        selected_algorithms: List of algorithms to include (if None, show all)
        xlim_max: Maximum x-axis limit
        **kwargs: Additional keyword arguments
    """
    # Create a figure with 2x2 subplots (adjust if the number of noise levels differs)
    n_plots = len(noise_sigmas)
    nrows = 2
    ncols = (n_plots + 1) // 2  # Ceiling division

    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 10))
    axes = axes.flatten()  # Flatten to easily iterate

    if apply_filter:
        df = filter_after_threshold(df, threshold)

    for i, noise_level in enumerate(sorted(noise_sigmas, key=lambda x: float(x) if isinstance(x, str) else x)):
        if i >= len(axes):
            print(f"Warning: More noise levels than subplot slots. Skipping noise level {noise_level}.")
            continue

        df_filtered = df[df["Noise Sigma"] == noise_level]
        if selected_algorithms is not None:
            df_filtered = df_filtered[df_filtered['algorithm'].isin(selected_algorithms)]

        sns.lineplot(
            data=df_filtered,
            x="Time Step",
            y="reward_0",
            hue="algorithm",
            style="algorithm",
            markers=True,
            dashes=True,
            linewidth=2,
            markersize=8,
            errorbar='sd',
            ax=axes[i]
        )

        axes[i].axhline(y=threshold, color='red', linestyle='--', label="Target rms threshold")
        axes[i].set_title(f"Noise Sigma = {noise_level}")
        axes[i].set_xlabel("Time Step")
        axes[i].set_ylabel("Reward")
        axes[i].grid(True)

        # Set x-axis limit using the minimum of xlim_max and the maximum 'Time Step' in the filtered data
        max_time_step = df_filtered['Time Step'].max() if not df_filtered.empty else xlim_max
        axes[i].set_xlim(0, min(xlim_max, max_time_step))

        # Remove individual legends for subplots except the first one
        if i != 0:
            legend = axes[i].get_legend()
            if legend is not None:
                legend.remove()

    # Remove any unused subplots
    for i in range(len(noise_sigmas), len(axes)):
        fig.delaxes(axes[i])

    # Extract legend handles and labels from the first subplot and remove its legend
    if len(axes) > 0 and df.shape[0] > 0:
        handles, labels = axes[0].get_legend_handles_labels()
        if axes[0].get_legend() is not None:
            axes[0].get_legend().remove()
        # Add a common legend to the figure, placed on top
        fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.0), ncol=2, title='Algorithm')

    plt.tight_layout(rect=[0, 0, 1, 0.85])
    plt.savefig(figure_name)
    plt.show()
