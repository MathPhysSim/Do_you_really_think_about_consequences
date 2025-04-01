import pickle
from pathlib import Path
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

def load_experiment_data(base_dir_root):
    """
    Load experiment results from stored pickle files.

    Args:
        base_dir_root (Path): Base directory where results are stored.

    Returns:
        pd.DataFrame: Combined DataFrame with all loaded results.
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

    # print(f"Detected noise levels: {noise_levels}")
    for noise_sigma in noise_levels:
        base_dir = base_dir_root / f"noise_sigma_{noise_sigma}"
        # print(f"Checking directory: {base_dir}")

        if not base_dir.exists():
            print(f"Directory does not exist: {base_dir}")
            continue

        df_all_seeds = []

        for file in base_dir.glob("*.pkl"):  # Process .pkl files only
            with open(file, "rb") as f:
                data = pickle.load(f)
                # print(f"Loaded {file.name}")

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


def plot_experiment_results(df_combined, test_name):
    """
    Plot reward evolution over time for different noise levels.

    Args:
        df_combined (pd.DataFrame): Processed DataFrame with experiment data.
    """
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df_combined, x="Time Step", y="reward_0", hue="Noise Sigma", errorbar='sd')
    plt.title(f"Reward vs Time for Different Noise Levels for {test_name}")
    plt.xlabel("Time Step")
    plt.ylabel("Reward")
    plt.legend(title="Noise Levels")
    plt.grid(True)
    plt.show()


experiment_name = 'noise_test'
# results_date = '2025-03-19'  # Replace with the actual date of your results
results_date = '2025-03-28'  # Replace with the actual date of your results

def map_test_name(test_name):
    if test_name == 'Classical':
        test_name='Model-free stepwise optimisation (COBYLA)'
    elif test_name == 'GP_MPC_1':
        test_name = 'Data-driven GP based MPC - generic'
    elif test_name == 'LinearMPC':
        test_name='Data-driven Linear Bayesian based MPC'
    elif test_name == 'MPC':
        test_name = 'Model-based MPC - perfect model'
    elif test_name == 'MPC_short':
        test_name = 'Model-based stepwise optimisation - perfect model'
    elif test_name == 'Structured_MPC':
        test_name = 'Data-driven GP based MPC respecting causality'
    else:
        test_name = test_name
    return test_name

def load_all_tests():
    """
    Automatically detects and loads data for all tests (methods) in the results directory.

    Returns:
        pd.DataFrame: Combined DataFrame with data from all tests.
    """
    base_dir_root = Path("results") / experiment_name / f'Results_{results_date}'
    all_tests_data = []

    # Automatically detect test names by checking subdirectories
    test_names = [d.name for d in base_dir_root.iterdir() if d.is_dir()]
    print(f"Detected test names: {test_names}")

    test_names = [test_name for test_name in test_names if test_name != 'Random']
    # test_names = [test_name for test_name in test_names if test_name != 'Classical']
    # test_names = [test_name for test_name in test_names if test_name != 'MPC_short']
    # test_names = [test_name for test_name in test_names if test_name != 'Analytic']
    test_names = [test_name for test_name in test_names if test_name != 'Structured_MPC_1']
    test_names = [test_name for test_name in test_names if test_name != 'Structured_MPC_2']
    test_names = [test_name for test_name in test_names if test_name != 'GP_MPC']

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



def filter_after_threshold(df, threshold=-0.1):
    """
    For each trajectory (grouped by 'Seed'), remove the data point that occurs
    one time step after the reward first falls below the given threshold.

    Args:
        df (pd.DataFrame): DataFrame containing the trajectory data.
        threshold (float): The threshold value for reward_0.

    Returns:
        pd.DataFrame: The filtered DataFrame.
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

# Load data for all tests
data_all = load_all_tests()
data_all.sort_values(by='algorithm', inplace=True)

if data_all is not None:
    print(data_all.columns)
    print(data_all['Noise Sigma'])

    # Plot the reward evolution for each noise sigma level by Olga
    def plot_reward_evolution_all_noises(df, noise_sigmas):
        """
        Plot reward evolution over time for all noise sigma levels, with different algorithms.

        Args:
            df (pd.DataFrame): The combined DataFrame containing experiment data.
            noise_sigmas (list): List of unique noise sigma levels to plot.
        """
        for noise_level in noise_sigmas:
            # Filter the DataFrame for the current noise level
            # df_filtered = df[df["Noise Sigma"] == noise_level]
            df_filtered = df[df["Noise Sigma"].astype(str) == str(noise_level)]

            # Create the plot
            plt.figure(figsize=(10, 6))
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
                errorbar='sd'
            )

            # Add title and labels
            plt.title(f"Reward Evolution Over Time (Noise Sigma = {noise_level})")
            plt.xlabel("Time Step")
            plt.ylabel("Reward")
            plt.legend(title="Algorithm")
            plt.grid(True)
            plt.tight_layout()
            # Add a horizontal threshold line
            plt.axhline(y=-0.1, color='red', linestyle='--', label="Threshold (-0.1)")

            # Adjust layout
            plt.tight_layout()

            # Show the plot
            plt.show()

    # Get unique noise sigma levels from the DataFrame
    noise_sigmas = data_all["Noise Sigma"].unique()

    # # Plot the reward evolution for each noise sigma level
    # plot_reward_evolution_all_noises(data_all, noise_sigmas)

def plot_reward_evolution_selected_algorithms(df):
    """
    Plot reward evolution over time for four specific algorithms:
    'Model-based MPC - perfect model', 'PPO',
    'Data-driven Linear Bayesian based MPC', and
    'Data-driven GP based MPC respecting causality'.

    Args:
        df (pd.DataFrame): The combined DataFrame containing experiment data.
    """
    # Define the list of algorithms to plot
    algorithms_to_plot = [
        "Model-based MPC - perfect model",
        "PPO",
        "Data-driven Linear Bayesian based MPC",
        "Data-driven GP based MPC respecting causality"
    ]

    # Filter the DataFrame for the selected algorithms
    df_filtered = df[df["algorithm"].isin(algorithms_to_plot)]

    # Create the plot
    plt.figure(figsize=(12, 8))
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
        errorbar='sd'
    )
    plt.title("Reward Evolution for Selected Algorithms")
    plt.xlabel("Time Step")
    plt.ylabel("Reward")
    plt.legend(title="Algorithm")
    plt.grid(True)
    plt.tight_layout()
    plt.show()



def plot_reward_evolution_all_noises_combined(df, noise_sigmas,
                                              figure_name='combined_plot.pdf',
                                              apply_filter=False,
                                              threshold=-0.1,
                                              selected_algorithms=None, **kwargs):
    """
    Plot reward evolution over time for all noise sigma levels in one combined figure with subplots.

    Args:
        df (pd.DataFrame): The combined DataFrame containing experiment data.
        noise_sigmas (list): List of unique noise sigma levels to plot.
    """
    # Create a figure with 2x2 subplots (adjust if the number of noise levels differs)
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()  # Flatten to easily iterate

    if apply_filter:
        df = filter_after_threshold(df, threshold)


    print(df['algorithm'].unique())
    for i, noise_level in enumerate(sorted(noise_sigmas, key=lambda x: float(x))):
        df_filtered = df[df["Noise Sigma"] == noise_level]
        if selected_algorithms is not None:
            df_filtered = df_filtered[df_filtered['algorithm'].isin(selected_algorithms)]

        print(df_filtered.columns)
        # df_filtered = df[df["Noise Sigma"].astype(str) == str(noise_level)]
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

        axes[i].axhline(y=-0.1, color='red', linestyle='--', label="Target rms threshold (-0.1)")
        axes[i].set_title(f"Noise Sigma = {noise_level}")
        axes[i].set_xlabel("Time Step")
        axes[i].set_ylabel("Reward")
        axes[i].grid(True)
        xlim_max = kwargs.get('xlim_max',80)
        # Set x-axis limit using the minimum of axlim and the maximum 'Time Step' in the filtered data
        axes[i].set_xlim(0, min(xlim_max, df_filtered['Time Step'].max()))

        # # Highlight lines that include 'causality' (case-insensitive) in their label
        # for line in axes[i].lines:
        #     if 'causality' in line.get_label().lower():
        #         line.set_linewidth(4)
        #         line.set_markersize(50)

        # Remove individual legends for subplots except the first one
        if i != 0:
            legend = axes[i].get_legend()
            if legend is not None:
                legend.remove()

    # Extract legend handles and labels from the first subplot and remove its legend
    handles, labels = axes[0].get_legend_handles_labels()
    axes[0].get_legend().remove()
    # Add a common legend to the figure, placed on top
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.0), ncol=2, title='Algorithm')
    plt.tight_layout(rect=[0, 0, 1, 0.85])
    plt.savefig(figure_name)
    plt.show()

noise_sigmas = [
    0,
    # 0.1,
    # 0.05,
    0.01,
    0.001,
    0.025
]
# Uncomment the following line to generate the combined plot for the four methods
plot_reward_evolution_all_noises_combined(df=data_all, noise_sigmas=noise_sigmas, figure_name='combined_plot_all.pdf')




noise_sigmas = [
    # '0',
    # 0.1,
    0.05,
    0.01,
    0.001,
    0.025
]

plot_reward_evolution_all_noises_combined(
    df=data_all,
    noise_sigmas=noise_sigmas,
    figure_name='combined_plot_selected_algorithms.pdf',
    xlim_max=15,
    selected_algorithms=[
        'Data-driven GP based MPC respecting causality',
        'Data-driven GP based MPC - generic',
        'Data-driven Linear Bayesian based MPC',
        # 'GP_MPC_1'
        'Model-based MPC - perfect model',
        'PPO'
    ]
)

# plot_reward_evolution_all_noises_combined(df=data_all[data_all['algorithm'] !=
#                                                       'Model-free stepwise optimisation (COBYLA)'],
#                                           noise_sigmas=noise_sigmas,
#                                           figure_name='combined_plot_all_without_COBYLA.pdf',
#                                           xlim_max=5)
