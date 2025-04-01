import pickle
import shutil
from pathlib import Path

import yaml


def manage_directory(path, clear=False):
    """
    Ensures a directory exists. Optionally clears its contents.
    """
    path = Path(path)
    if clear and path.exists():
        shutil.rmtree(path, ignore_errors=True)
    path.mkdir(parents=True, exist_ok=True)


def remove_file(path):
    """
    Removes a file if it exists.
    """
    path = Path(path)
    if path.exists():
        path.unlink()


def create_experiment_setup_train(config_dir="configs/awake.yaml"):
    """
    Sets up the necessary folder structure for training experiments.
    """
    with open(config_dir, "r") as f:
        config = yaml.safe_load(f)

    base_folder = config["env-name"]
    experiment_folder = Path(base_folder) / config["experiment-config"]["experiment-name"]
    experiment_type = 'train'
    training_data_location = experiment_folder / experiment_type
    save_progress_data_dir = training_data_location / 'progress'

    meta_policy_location = experiment_folder / config["experiment-config"]["meta-policy-name"]
    policy_logging_dir = experiment_folder / "meta_policy_logging"

    # Ensure the policy logging directory exists
    policy_logging_dir.mkdir(parents=True, exist_ok=True)
    time_logging_location = Path("maml_rl/env/Tasks_data/total_time.csv")

    # Manage directories and files
    manage_directory(training_data_location, clear=True)
    manage_directory(save_progress_data_dir)
    manage_directory(policy_logging_dir)
    remove_file(meta_policy_location)
    remove_file(time_logging_location)

    print("Experiment setup created", experiment_folder)

    return config, base_folder, experiment_folder, save_progress_data_dir, meta_policy_location, policy_logging_dir


def create_experiment_setup_test(
    config_dir="configs/awake.yaml",
    experiment_name=None,
    # start_from_meta_policy=False,
    clear=True,
):
    """
    Sets up the necessary folder structure and tasks for testing experiments.
    """
    with open(config_dir, "r") as f:
        config = yaml.safe_load(f)

    base_folder = config["env-name"]
    experiment_type = 'test'
    # if not start_from_meta_policy:
    #     experiment_name += '_standard_training'
    if experiment_name is None:
        experiment_name = config["experiment-config"]["experiment-name"]
    experiment_folder = Path(base_folder) / experiment_name
    logging_path = experiment_folder / experiment_type
    save_progress_data_dir = logging_path / 'progress'

    # Manage directories
    manage_directory(logging_path, clear=clear)
    manage_directory(save_progress_data_dir, clear=clear)

    # Load verification tasks
    verification_tasks_loc = Path(config['env-kwargs']["verification_tasks_loc"])
    filename = config['env-kwargs']["verification_tasks_file"]
    full_path = verification_tasks_loc / filename
    with open(full_path, "rb") as input_file:
        tasks = pickle.load(input_file)
    # Save tasks from training for completeness for tests
    tasks_dir = experiment_folder / 'tasks'
    manage_directory(tasks_dir,clear)
    with open(tasks_dir / 'tasks.pkl', "wb") as fp:
        pickle.dump(tasks, fp)

    return config, base_folder, experiment_folder, save_progress_data_dir, tasks


def save_progress(file_name, save_progress_data_dir, data):
    """
    Saves progress data to the specified directory.
    """
    full_path = Path(save_progress_data_dir) / file_name
    with open(full_path, "wb") as file:
        pickle.dump(data, file)

def get_folder_names_for_experiment(experiment_name, experiment_type, config_dir="configs/awake.yaml"):
    with open(config_dir, "r") as f:
        config = yaml.safe_load(f)

    base_folder = config["env-name"]
    experiment_folder = Path(base_folder) / experiment_name
    training_data_location = experiment_folder / experiment_type
    save_progress_data_dir = training_data_location / 'progress'

    return base_folder, experiment_folder, save_progress_data_dir, config