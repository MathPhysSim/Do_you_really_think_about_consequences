import multiprocessing
import time
from abc import ABC, abstractmethod
from typing import Tuple, Union, List, Optional, Dict, Any

import gpytorch
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import cm
from scipy.optimize import minimize
from torch import Tensor

matplotlib.rc("font", size="6")
# np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)
ALPHA_CONFIDENCE_BOUNDS = 0.3
FONTSIZE = 6

# Values used for graph visualizations only
MEAN_PRED_COST_INIT = 0
STD_MEAN_PRED_COST_INIT = 1

N = 10

colors_map = cm.rainbow(np.linspace(0, 1, N))


def init_visu_and_folders(env, num_steps, params_controller_dict):
    """
    Create and return the objects used for visualisation in real-time.
    Also create the folder where to save the figures.
    Args:
            env (gym env): gym environment
            num_steps (int): number of steps during which the action is maintained constant
            env_str (str): string that describes the env name
            params_general (dict): general parameters (see parameters.md for more info)
            params_controller_dict (dict): controller parameters (see parameters.md for more info)

        Returns:
                live_plot_obj (object): object used to visualize the control and observation evolution in a 2d graph
    """
    live_plot_obj = LivePlotSequential(
        num_steps,
        env.observation_space,
        env.action_space,
        # step_pred=params_controller_dict["controller"]["num_repeat_actions"],
        step_pred=1,
    )

    return live_plot_obj


def close_run(ctrl_obj, env):
    """
    Close all visualisations and parallel processes that are still running.
    Save all visualisations one last time if save args set to True
    Args:
            ctrl_obj:
            env (gym env): gym environment
    """

    env.__exit__()
    ctrl_obj.check_and_close_processes()

class LivePlotSequential:
    def __init__(
        self,
        num_steps_total,
        obs_space,
        action_space,
        step_pred,
        mul_std_bounds=3,
        fontsize=6,
    ):
        self.fig, self.axes = plt.subplots(nrows=3, figsize=(6, 5), sharex=True)
        self.axes[0].set_title("Normed states and predictions")
        self.axes[1].set_title("Normed actions")
        self.axes[2].set_title("Reward and horizon reward")
        plt.xlabel("Env steps")
        self.min_state = -0.03
        self.max_state = 1.03
        self.axes[0].set_ylim(self.min_state, self.max_state)
        self.axes[1].set_ylim(-0.03, 1.03)
        self.axes[2].set_xlim(0, num_steps_total)
        plt.tight_layout()

        self.step_pred = step_pred
        self.mul_std_bounds = mul_std_bounds

        self.states = np.empty((num_steps_total, obs_space.shape[0]))
        self.actions = np.empty((num_steps_total, action_space.shape[0]))
        self.costs = np.empty(num_steps_total)

        self.mean_costs_pred = np.empty_like(self.costs)
        self.mean_costs_std_pred = np.empty_like(self.costs)

        self.min_obs = obs_space.low
        self.max_obs = obs_space.high
        self.min_action = action_space.low
        self.max_action = action_space.high

        self.num_points_show = 0
        self.lines_states = [
            self.axes[0].plot(
                [], [], label="state" + str(state_idx), color=colors_map[state_idx]#cmap.colors[2 * state_idx]
            )
            for state_idx in range(obs_space.shape[0])
        ]
        self.line_cost = self.axes[2].plot([], [], label="reward", color="k")

        self.lines_actions = [
            self.axes[1].step(
                [],
                [],
                label="action" + str(action_idx),
                color=colors_map[action_idx],
            )
            for action_idx in range(action_space.shape[0])
        ]

        self.line_mean_costs_pred = self.axes[2].plot(
            [], [], label="mean predicted reward", color="orange"
        )
        self.lines_states_pred = [
            self.axes[0].plot(
                [],
                [],
                # [],
                label="predicted_states" + str(state_idx),
                color=colors_map[state_idx],
                linestyle="dashed",
            )
            for state_idx in range(obs_space.shape[0])
        ]
        self.lines_actions_pred = [
            self.axes[1].step(
                [],
                [],
                label="predicted_action" + str(action_idx),
                color=colors_map[action_idx],
                linestyle="dashed",
            )
            for action_idx in range(action_space.shape[0])
        ]
        self.line_costs_pred = self.axes[2].plot(
            [], [], label="predicted reward", color="k", linestyle="dashed"
        )

        self.axes[0].legend(fontsize=fontsize)
        self.axes[0].grid()
        self.axes[1].legend(fontsize=fontsize)
        self.axes[1].grid()
        self.axes[2].legend(fontsize=fontsize)
        self.axes[2].grid()
        plt.show(block=False)

    def update(self, obs, action, cost, info_dict=None):
        obs_norm = (obs - self.min_obs) / (self.max_obs - self.min_obs)
        action_norm = (action - self.min_action) / (self.max_action - self.min_action)
        self.states[self.num_points_show] = obs_norm
        self.costs[self.num_points_show] = -cost

        update_limits = False
        min_state_actual = np.min(obs_norm)
        if min_state_actual < self.min_state:
            self.min_state = min_state_actual
            update_limits = True

        max_state_actual = np.max(obs_norm)
        if max_state_actual > self.max_state:
            self.max_state = max_state_actual
            update_limits = True

        if update_limits:
            self.axes[0].set_ylim(self.min_state, self.max_state)

        idxs = np.arange(0, (self.num_points_show + 1))
        for idx_axes in range(len(self.axes)):
            # self.axes[idx_axes].collections.clear()
            # print(self.axes[idx_axes])
            # self.axes[idx_axes].clear()
            for collection in self.axes[idx_axes].collections:
                collection.remove()

        for idx_state in range(len(obs_norm)):
            self.lines_states[idx_state][0].set_data(
                idxs, self.states[: (self.num_points_show + 1), idx_state]
            )

        self.actions[self.num_points_show] = action_norm
        for idx_action in range(len(action_norm)):
            self.lines_actions[idx_action][0].set_data(
                idxs, self.actions[: (self.num_points_show + 1), idx_action]
            )

        self.line_cost[0].set_data(idxs, self.costs[: (self.num_points_show + 1)])

        if info_dict is not None:
            mean_costs_pred = info_dict["mean predicted cost"]
            mean_costs_pred *= -1
            mean_costs_std_pred = info_dict["mean predicted cost std"]
            states_pred = info_dict["predicted states"]
            states_std_pred = info_dict["predicted states std"]
            actions_pred = info_dict["predicted actions"]
            costs_pred = info_dict["predicted costs"]
            costs_pred *= -1
            costs_std_pred = info_dict["predicted costs std"]
            np.nan_to_num(mean_costs_pred, copy=False, nan=-1, posinf=-1, neginf=-1)
            np.nan_to_num(mean_costs_std_pred, copy=False, nan=99, posinf=99, neginf=99)
            np.nan_to_num(states_pred, copy=False, nan=-1, posinf=-1, neginf=-1)
            np.nan_to_num(states_std_pred, copy=False, nan=99, posinf=99, neginf=0)
            np.nan_to_num(costs_pred, copy=False, nan=-1, posinf=-1, neginf=-1)
            np.nan_to_num(costs_std_pred, copy=False, nan=99, posinf=99, neginf=0)
            # if num_repeat_action is not 1, the control do not happen at each iteration,
            # we must select the last index where the control happened as the start of the prediction horizon
            idx_prev_control = (idxs[-1] // self.step_pred) * self.step_pred
            idxs_future = np.arange(
                idx_prev_control,
                idx_prev_control + self.step_pred + len(states_pred) * self.step_pred,
                self.step_pred,
            )

            self.mean_costs_pred[self.num_points_show] = mean_costs_pred
            self.mean_costs_std_pred[self.num_points_show] = mean_costs_std_pred

            for idx_state in range(len(obs_norm)):
                future_states_show = np.concatenate(
                    (
                        [self.states[idx_prev_control, idx_state]],
                        states_pred[:, idx_state],
                    )
                )
                self.lines_states_pred[idx_state][0].set_data(
                    idxs_future, future_states_show
                )
                future_states_std_show = np.concatenate(
                    ([0], states_std_pred[:, idx_state])
                )
                self.axes[0].fill_between(
                    idxs_future,
                    future_states_show - future_states_std_show * self.mul_std_bounds,
                    future_states_show + future_states_std_show * self.mul_std_bounds,
                    facecolor=colors_map[idx_state],
                    alpha=ALPHA_CONFIDENCE_BOUNDS,
                    label="predicted "
                    + str(self.mul_std_bounds)
                    + " std bounds state "
                    + str(idx_state),
                )
            for idx_action in range(len(action_norm)):
                self.lines_actions_pred[idx_action][0].set_data(
                    idxs_future,
                    np.concatenate(
                        (
                            [self.actions[idx_prev_control, idx_action]],
                            actions_pred[:, idx_action],
                        )
                    ),
                )

            future_costs_show = np.concatenate(
                ([self.costs[idx_prev_control]], costs_pred)
            )
            self.line_costs_pred[0].set_data(idxs_future, future_costs_show)

            future_cost_std_show = np.concatenate(([0], costs_std_pred))
            self.axes[2].fill_between(
                idxs_future,
                future_costs_show - future_cost_std_show * self.mul_std_bounds,
                future_costs_show + future_cost_std_show * self.mul_std_bounds,
                facecolor="black",
                alpha=ALPHA_CONFIDENCE_BOUNDS,
                label="predicted " + str(self.mul_std_bounds) + " std cost bounds",
            )
        else:
            if self.num_points_show == 0:
                self.mean_costs_pred[self.num_points_show] = MEAN_PRED_COST_INIT
                self.mean_costs_std_pred[self.num_points_show] = STD_MEAN_PRED_COST_INIT
            else:
                self.mean_costs_pred[self.num_points_show] = self.mean_costs_pred[
                    self.num_points_show - 1
                ]
                self.mean_costs_std_pred[
                    self.num_points_show
                ] = self.mean_costs_std_pred[self.num_points_show - 1]

        self.line_mean_costs_pred[0].set_data(
            idxs, self.mean_costs_pred[: (self.num_points_show + 1)]
        )
        self.axes[2].set_ylim(

            np.min(
                [
                    np.min(self.mean_costs_pred[: (self.num_points_show + 1)]),
                    np.min(self.costs[: (self.num_points_show + 1)]),
                ]
            )
            * 1.1,
            0.5
        )
        self.axes[2].fill_between(
            idxs,
            self.mean_costs_pred[: (self.num_points_show + 1)]
            - self.mean_costs_std_pred[: (self.num_points_show + 1)]
            * self.mul_std_bounds,
            self.mean_costs_pred[: (self.num_points_show + 1)]
            + self.mean_costs_std_pred[: (self.num_points_show + 1)]
            * self.mul_std_bounds,
            facecolor="orange",
            alpha=ALPHA_CONFIDENCE_BOUNDS,
            label="mean predicted " + str(self.mul_std_bounds) + " std cost bounds",
        )

        # Limit the third subplot's y-axis to -1 and 0
        self.axes[2].set_ylim(-3, 0.1)
        self.fig.canvas.draw()
        plt.pause(0.01)
        self.num_points_show += 1


class ExactGPModelMonoTask(gpytorch.models.ExactGP):
    """
    A single-task Exact Gaussian Process Model using GPyTorch.

    Args:
        train_x (Tensor): Training input data.
        train_y (Tensor): Training target data.
        dim_input (int): Dimensionality of the input features.
        likelihood (Optional[gpytorch.likelihoods.Likelihood]): Likelihood function. Defaults to GaussianLikelihood.
        kernel (Optional[gpytorch.kernels.Kernel]): Covariance kernel. Defaults to RBFKernel with ARD.
    """

    def __init__(
        self,
        train_x: Optional[Tensor],
        train_y: Optional[Tensor],
        dim_input: int,
        likelihood: Optional[gpytorch.likelihoods.Likelihood] = None,
        kernel: Optional[gpytorch.kernels.Kernel] = None
    ):
        likelihood = likelihood or gpytorch.likelihoods.GaussianLikelihood()
        super().__init__(train_x, train_y, likelihood)

        self.mean_module = gpytorch.means.ZeroMean()
        self.covar_module = (
            kernel or gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.RBFKernel(ard_num_dims=dim_input)
            )
        )

    def forward(self, x: Tensor) -> gpytorch.distributions.MultivariateNormal:
        """
        Forward pass through the model.

        Args:
            x (Tensor): Input data for prediction.

        Returns:
            gpytorch.distributions.MultivariateNormal: Predictive distribution.
        """
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


def create_models(
    train_inputs: Optional[Tensor],
    train_targets: Optional[Tensor],
    params: Union[Dict[str, Any], List[Dict[str, Any]]],
    constraints_gp: Optional[Dict[str, Any]],
    dof: int = 10
) -> List[ExactGPModelMonoTask]:
    """
    Create and configure Gaussian process models for state transition predictions.

    Args:
        train_inputs (Optional[Tensor]): Input data for the GPs.
        train_targets (Optional[Tensor]): Target data (state changes).
        params (dict or list): Hyper-parameters for GP initialization.
        constraints_gp (dict): Constraints for GP hyperparameters.
        dof (int, optional): Degrees of freedom / number of models. Defaults to 10.

    Returns:
        List[ExactGPModelMonoTask]: List of GP models.
    """
    num_models = dof

    # For each model, define the corresponding indices:
    indices_list = [list(range(pos)) + list(range(dof, pos + dof)) for pos in range(1, dof + 1)]
    num_inputs_model = [len(indices) for indices in indices_list]

    if train_inputs is not None and train_targets is not None:
        models = [
            ExactGPModelMonoTask(
                train_inputs[:, indices_list[idx_model]],
                train_targets[:, idx_model],
                dim_input=len(train_inputs[0, indices_list[idx_model]])
            )
            for idx_model in range(num_models)
        ]
    else:
        models = [
            ExactGPModelMonoTask(None, None, dim_input=num_inputs_model[idx_model])
            for idx_model in range(num_models)
        ]

    # Apply hyperparameter constraints if provided.
    for idx_model, model in enumerate(models):
        if constraints_gp:
            # Noise constraints
            if "min_std_noise" in constraints_gp:
                min_noise = constraints_gp["min_std_noise"]
                max_noise = constraints_gp["max_std_noise"]
                min_var_noise = np.power(min_noise[idx_model] if isinstance(min_noise, (list, np.ndarray, Tensor)) else min_noise, 2)
                max_var_noise = np.power(max_noise[idx_model] if isinstance(max_noise, (list, np.ndarray, Tensor)) else max_noise, 2)
                model.likelihood.noise_covar.register_constraint(
                    "raw_noise",
                    gpytorch.constraints.Interval(lower_bound=min_var_noise, upper_bound=max_var_noise)
                )
            # Outputscale constraints
            if "min_outputscale" in constraints_gp:
                min_out = constraints_gp["min_outputscale"]
                max_out = constraints_gp["max_outputscale"]
                min_outputscale = min_out[idx_model] if isinstance(min_out, (list, np.ndarray, Tensor)) else min_out
                max_outputscale = max_out[idx_model] if isinstance(max_out, (list, np.ndarray, Tensor)) else max_out
                model.covar_module.register_constraint(
                    "raw_outputscale",
                    gpytorch.constraints.Interval(lower_bound=min_outputscale, upper_bound=max_outputscale)
                )
            # Lengthscale constraints
            if "min_lengthscale" in constraints_gp:
                min_ls = constraints_gp["min_lengthscale"]
                max_ls = constraints_gp["max_lengthscale"]
                min_lengthscale = min_ls if isinstance(min_ls, (float, int)) else min_ls[idx_model]
                max_lengthscale = max_ls if isinstance(max_ls, (float, int)) else max_ls[idx_model]
                model.covar_module.base_kernel.register_constraint(
                    "raw_lengthscale",
                    gpytorch.constraints.Interval(lower_bound=min_lengthscale, upper_bound=max_lengthscale)
                )

        # Initialize hyperparameters
        if isinstance(params, dict):
            hypers = {
                "base_kernel.lengthscale": params["base_kernel.lengthscale"][idx_model],
                "outputscale": params["outputscale"][idx_model],
            }
            hypers_likelihood = {
                "noise_covar.noise": params["noise_covar.noise"][idx_model]
            }
            model.likelihood.initialize(**hypers_likelihood)
            model.covar_module.initialize(**hypers)
        elif isinstance(params, list):
            model.load_state_dict(params[idx_model])
    return models

class ActionExplorer:
    def __init__(self, action_space: Any, unit_vectors: Optional[List[np.ndarray]] = None):
        """
        Initialize the ActionExplorer with a given action space.

        Args:
            action_space: The action space of the environment. Must have attributes 'low' and 'high'.
            unit_vectors: Optional list of unit vectors to explore. If None, defaults to the standard basis.
        """
        self.action_space = action_space
        self.dim = action_space.shape[0]
        self.scaling_factor = 0.1
        self.n_actions_controlled = 50
        # Default to standard basis unit vectors if none provided
        if unit_vectors is None:
            self.unit_vectors = [self.scaling_factor * np.eye(self.dim)[i] + 0.5 for i in range(self.dim)] + \
                                [- self.scaling_factor * np.eye(self.dim)[i] + 0.5 for i in range(self.dim)]
        else:
            self.unit_vectors = unit_vectors
        self.current_index = 0

    def get_next_action(self) -> np.ndarray:
        """
        Return the next action.

        Initially returns actions corresponding to unit vectors scaled to the action space.
        After the unit vectors are exhausted, returns a random action.
        """
        if self.current_index < self.n_actions_controlled:#len(self.unit_vectors):
            unit_vector = self.unit_vectors[self.current_index % len(self.unit_vectors)] + np.random.uniform(self.action_space.low,
                                                                                    self.action_space.high,
                                                                                    self.dim) * self.scaling_factor
            self.current_index += 1
            # Scale the unit vector to span the action space range.
            return self.action_space.low + (self.action_space.high - self.action_space.low) * unit_vector
        else:
            return np.random.uniform(self.action_space.low, self.action_space.high)


def init_control(
        ctrl_obj: Any, env: Any, live_plot_obj: Any, random_actions_init: int, num_repeat_actions: int = 1
) -> Tuple[Any, Any, Any, Any, Any, Any, Any, List[Any], List[Any], List[Any]]:
    """
    Initialize control with a sequence of exploratory actions followed by random actions and update visualization/memory.

    Args:
        ctrl_obj: Controller object (e.g. GpMpcController).
        env: Gym environment.
        live_plot_obj: Visualization object with an `update` method.
        random_actions_init (int): Number of initial actions.
        num_repeat_actions (int, optional): Number of consecutive actions (affects memory). Defaults to 1.

    Returns:
        Tuple containing updated control object, environment, live plot object,
        last observation, last action, last cost, previous observation, and lists of observations,
        actions, and rewards.
    """
    import logging
    logger = logging.getLogger(__name__)

    obs_lst, actions_lst, rewards_lst = [], [], []
    obs, _ = env.reset()
    action = None
    cost = None
    obs_prev_ctrl = None
    done = False
    idx_action = 0

    # Create an instance of ActionExplorer for exploring all DoF equally.
    action_explorer = ActionExplorer(env.action_space)

    while idx_action < random_actions_init:
        if idx_action % num_repeat_actions == 0 or action is None:
            # Get the next exploratory action.
            action = action_explorer.get_next_action()
            logger.debug(f"Exploratory action chosen: {action}")
            idx_action += 1
            if obs_prev_ctrl is not None and cost is not None:
                ctrl_obj.add_memory(obs=obs_prev_ctrl, action=action, obs_new=obs, reward=cost)
        if done:
            obs, _ = env.reset()
            idx_action += 1
        obs_new, reward, done, _, _ = env.step(action)
        obs_prev_ctrl = obs
        obs = obs_new
        cost, _ = ctrl_obj.compute_cost_unnormalized(obs_new, action)

        obs_lst.append(obs)
        actions_lst.append(action)
        rewards_lst.append(cost)

        live_plot_obj.update(obs=obs, action=action, cost=cost, info_dict=None)

        ctrl_obj.action_previous_iter = action

    return ctrl_obj, env, live_plot_obj, obs, action, cost, obs_prev_ctrl, obs_lst, actions_lst, rewards_lst


class BaseControllerObject(ABC):
    def __init__(self, observation_space: Any, action_space: Any, n_points_init_memory: int = 1000):
        self.action_space = action_space
        self.obs_space = observation_space
        self.num_states = self.obs_space.shape[0]
        self.num_actions = action_space.shape[0]
        self.num_inputs = self.num_states + self.num_actions
        self.points_add_mem_when_full = n_points_init_memory
        self.x = torch.empty(n_points_init_memory, self.num_inputs)
        self.y = torch.empty(n_points_init_memory, self.num_states)
        self.rewards = torch.empty(n_points_init_memory)
        self.len_mem = 0

    @abstractmethod
    def add_memory(self, observation, action, new_observation, reward, **kwargs) -> None:
        pass

    @abstractmethod
    def compute_action(self, observation, s_observation) -> Any:
        pass

    @abstractmethod
    def train(self, *args, **kwargs) -> None:
        pass


# Set default tensor type to double for higher precision.
torch.set_default_dtype(torch.double)


class GpMpcController(BaseControllerObject):
    def __init__(self, observation_space: Any, action_space: Any, params_dict: Dict[str, Any]):
        params_dict = self.preprocess_params(params_dict)
        self.weight_matrix_cost = torch.block_diag(
            torch.diag(params_dict['controller']['weight_state']),
            torch.diag(params_dict['controller']['weight_action'])
        )
        self.weight_matrix_cost_terminal = torch.diag(params_dict['controller']['weight_state_terminal'])
        self.target_state_norm = params_dict['controller']['target_state_norm']
        self.target_state_action_norm = torch.cat((self.target_state_norm, params_dict['controller']['target_action_norm']))
        self.len_horizon = params_dict['controller']['len_horizon']
        self.exploration_factor = params_dict['controller']['exploration_factor']
        self.obs_var_norm = torch.diag(params_dict['controller']['obs_var_norm'])
        self.barrier_weight = params_dict['controller']['barrier_weight']

        self.lr_train = params_dict['train']['lr_train']
        self.iter_train = params_dict['train']['iter_train']
        self.clip_grad_value = params_dict['train']['clip_grad_value']
        self.training_frequency = params_dict['train']['training_frequency']
        self.print_train = params_dict['train']['print_train']
        self.step_print_train = params_dict['train']['step_print_train']
        self.DoF = params_dict['controller']['DoF']

        super().__init__(observation_space, action_space)

        self.gp_constraints = params_dict['gp_constraints']
        self.state_min = None
        self.state_max = None
        self.params_actions_optimizer = params_dict['actions_optimizer']

        # Use bounds (for normalized actions) and initialize previous action prediction.
        self.bounds = [(0, 1)] * (self.num_actions * self.len_horizon)
        self.actions_pred_previous_iter = np.random.uniform(low=0, high=1, size=(self.len_horizon, self.num_actions))

        self.models = create_models(
            train_inputs=None,
            train_targets=None,
            params=params_dict['gp_init'],
            constraints_gp=self.gp_constraints,
            dof=self.DoF
        )
        for model in self.models:
            model.eval()

        self.num_cores_main = multiprocessing.cpu_count()
        self.ctx = multiprocessing.get_context('spawn')
        self.queue_train = self.ctx.Queue()

        self.n_iter_ctrl = 0
        self.n_iter_obs = 0
        self.info_iters = {}
        self.idxs_mem_gp: List[int] = []

    @staticmethod
    def preprocess_params(params_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert list parameters into torch.Tensors.
        """
        controller = params_dict['controller']
        controller['target_state_norm'] = torch.Tensor(controller['target_state_norm'])
        controller['weight_state'] = torch.Tensor(controller['weight_state'])
        controller['weight_state_terminal'] = torch.Tensor(controller['weight_state_terminal'])
        controller['target_action_norm'] = torch.Tensor(controller['target_action_norm'])
        controller['weight_action'] = torch.Tensor(controller['weight_action'])

        for key in params_dict['gp_init']:
            if not isinstance(params_dict['gp_init'][key], (float, int)):
                params_dict['gp_init'][key] = torch.Tensor(params_dict['gp_init'][key])

        for key in params_dict['gp_constraints']:
            if not isinstance(params_dict['gp_constraints'][key], (float, int)):
                params_dict['gp_constraints'][key] = torch.Tensor(params_dict['gp_constraints'][key])
        controller['obs_var_norm'] = torch.Tensor(controller['obs_var_norm'])
        return params_dict

    def to_normed_obs_tensor(self, obs: np.ndarray) -> Tensor:
        """
        Normalize observation using gym environment bounds.
        """
        return torch.Tensor((obs - self.obs_space.low) / (self.obs_space.high - self.obs_space.low))

    def to_normed_var_tensor(self, obs_var: np.ndarray) -> Tensor:
        """
        Normalize observation variance using the squared range of the observation space.
        """
        range_val = self.obs_space.high - self.obs_space.low
        return torch.Tensor(obs_var / (range_val ** 2))

    def to_normed_action_tensor(self, action: np.ndarray) -> Tensor:
        """
        Normalize action using gym environment bounds.
        """
        return torch.Tensor((action - self.action_space.low) / (self.action_space.high - self.action_space.low))

    def denorm_action(self, action_norm: Union[np.ndarray, Tensor]) -> np.ndarray:
        """
        Denormalize action to apply to the environment.
        """
        return (action_norm * (self.action_space.high - self.action_space.low) + self.action_space.low)

    def barrier_cost(self, state: Tensor, alpha: float) -> Tensor:
        """
        Compute a barrier cost to penalize state values near the boundaries.
        """
        epsilon = 1e-6
        state = torch.clamp(state, epsilon, 1 - epsilon)
        return -alpha * (torch.log(state) + torch.log(1 - state)).sum(-1)

    def compute_cost(self, state_mu: Tensor, state_var: Tensor, action: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Compute mean and variance of the quadratic cost function for a given state-action distribution.
        """
        single_timestep = (state_var.ndim == 2)
        if single_timestep:
            state_mu = state_mu.unsqueeze(0)
            state_var = state_var.unsqueeze(0)
            action = action.unsqueeze(0)

        Np, Ns = state_mu.shape[0], state_mu.shape[1]
        Na = action.shape[1]
        x_mu = torch.cat((state_mu, action), dim=1)
        x_target = self.target_state_action_norm  # shape: (Ns + Na,)
        error = x_mu - x_target

        # Build state-action covariance (actions are assumed deterministic).
        state_action_var = torch.zeros((Np, Ns + Na, Ns + Na), device=state_var.device, dtype=state_var.dtype)
        state_action_var[:, :Ns, :Ns] = state_var

        W = self.weight_matrix_cost
        trace_term = torch.einsum('bii->b', W @ state_action_var)
        error = error.unsqueeze(-1)
        quadratic_term = (error.transpose(-2, -1) @ W @ error).squeeze(-1).squeeze(-1)
        cost_mu = trace_term + quadratic_term

        barrier = self.barrier_cost(state_mu, self.barrier_weight)
        cost_mu += barrier

        TS = W @ state_action_var
        trace_var_term = 2 * (TS ** 2).sum(dim=(-2, -1))
        var_quadratic_term = 4 * (error.transpose(-2, -1) @ W @ state_action_var @ W @ error).squeeze(-1).squeeze(-1)
        cost_var = trace_var_term + var_quadratic_term

        if single_timestep:
            cost_mu = cost_mu.squeeze(0)
            cost_var = cost_var.squeeze(0)
        return cost_mu, cost_var

    def compute_cost_terminal(self, state_mu: Tensor, state_var: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Compute mean and variance of the terminal cost.
        """
        state_mu, state_var = torch.as_tensor(state_mu), torch.as_tensor(state_var)
        single_timestep = (state_mu.ndim == 1)
        if single_timestep:
            state_mu = state_mu.unsqueeze(0)
            state_var = state_var.unsqueeze(0)

        Np, Ns = state_mu.shape[0], state_mu.shape[1]
        x_target = self.target_state_norm.unsqueeze(0)
        error = state_mu - x_target

        W = self.weight_matrix_cost_terminal
        TS = W @ state_var
        trace_term = torch.einsum('bii->b', TS)
        error = error.unsqueeze(-1)
        quadratic_term = (error.transpose(-2, -1) @ W @ error).squeeze(-1).squeeze(-1)
        cost_mu = trace_term + quadratic_term

        barrier = self.barrier_cost(state_mu, self.barrier_weight)
        cost_mu += barrier

        var_term1 = 2 * torch.einsum('bij,bji->b', TS, TS)
        W_error = W @ error
        temp = W @ state_var @ W_error
        var_term2 = 4 * (error.transpose(-2, -1) @ temp).squeeze(-1).squeeze(-1)
        cost_var = var_term1 + var_term2

        if single_timestep:
            cost_mu = cost_mu.squeeze(0)
            cost_var = cost_var.squeeze(0)
        return cost_mu, cost_var

    def compute_cost_unnormalized(self, obs: np.ndarray, action: np.ndarray, obs_var: Optional[np.ndarray] = None) -> Tuple[float, float]:
        """
        Compute cost on un-normalized state and action.
        """
        obs_norm = self.to_normed_obs_tensor(obs)
        action_norm = self.to_normed_action_tensor(action)
        obs_var_norm = self.obs_var_norm if obs_var is None else self.to_normed_var_tensor(obs_var)
        cost_mu, cost_var = self.compute_cost(obs_norm, obs_var_norm, action_norm)
        return cost_mu.item(), cost_var.item()

    @staticmethod
    def calculate_factorizations(x: Tensor, y: Tensor, models: List[ExactGPModelMonoTask], dof: int) -> Tuple[Tensor, Tensor]:
        """
        Compute intermediate factors (iK and beta) for GP predictions.
        """
        indices_list = [list(range(i)) + list(range(dof, i + dof)) for i in range(1, dof + 1)]
        K = torch.stack([
            model.covar_module(x[:, idxs]).evaluate()
            for model, idxs in zip(models, indices_list)
        ])
        batched_eye = torch.eye(K.shape[1]).repeat(K.shape[0], 1, 1)
        L = torch.linalg.cholesky(K + torch.stack([model.likelihood.noise for model in models])[:, None] * batched_eye)
        iK = torch.cholesky_solve(batched_eye, L)
        Y_ = (y.t())[:, :, None]
        beta = torch.cholesky_solve(Y_, L)[:, :, 0]
        return iK, beta

    def predict_next_state_change(self, state_mu: Tensor, state_var: Tensor, iK: Tensor, beta: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Predict the next state change, its variance, and the derivative-like term for each GP model.
        """
        M_list, V_list, S_diag_list = [], [], []
        indices_list = [list(range(i)) + list(range(self.DoF, i + self.DoF)) for i in range(1, self.DoF + 1)]
        for i, model in enumerate(self.models):
            idxs = indices_list[i]
            x_i = self.x[self.idxs_mem_gp][:, idxs]
            inp = x_i - state_mu[idxs]
            lengthscale = model.covar_module.base_kernel.lengthscale[0]
            variance = model.covar_module.outputscale
            iL = torch.diag(1 / lengthscale)
            iN = inp @ iL

            Sx = state_var[np.ix_(idxs, idxs)]
            epsilon = 1e-6
            B = iL @ Sx @ iL + torch.eye(len(idxs), dtype=Sx.dtype, device=Sx.device) + epsilon * torch.eye(len(idxs), dtype=Sx.dtype, device=Sx.device)
            t = torch.linalg.solve(B, iN.T).T
            lb = torch.exp(-0.5 * torch.sum(iN * t, dim=1)) * beta[i]
            det_B = torch.clamp(torch.det(B), min=1e-6)
            c = variance / torch.sqrt(det_B)
            M_i = c * torch.sum(lb)
            tiL = t @ iL
            V_i = (tiL.T @ lb[:, None])[:, 0] * c
            k_star = c * lb
            quad_term = k_star @ (iK[i] @ k_star)
            S_i_diag = torch.clamp(variance - quad_term, min=0.0)

            M_list.append(M_i)
            V_pad = torch.zeros(self.num_inputs, dtype=V_i.dtype, device=V_i.device)
            V_pad[idxs] = V_i
            V_list.append(V_pad)
            S_diag_list.append(S_i_diag)
        M = torch.stack(M_list)
        V = torch.stack(V_list)
        S = torch.diag(torch.tensor(S_diag_list, dtype=M.dtype, device=M.device))
        return M, S, V

    def predict_trajectory(self, actions: Tensor, obs_mu: Tensor, obs_var: Tensor, iK: Tensor, beta: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """
        Predict the trajectory (states and costs) given a sequence of actions.
        """
        state_dim = obs_mu.shape[0]
        mu_list = [obs_mu]
        var_list = [obs_var]
        for t in range(1, self.len_horizon + 1):
            input_var = torch.zeros((self.num_inputs, self.num_inputs), dtype=obs_var.dtype, device=obs_var.device)
            input_var[:state_dim, :state_dim] = var_list[-1]
            input_mean = torch.empty((self.num_inputs,), dtype=obs_mu.dtype, device=obs_mu.device)
            input_mean[:self.num_states] = mu_list[-1]
            input_mean[self.num_states:(self.num_states + self.num_actions)] = actions[t - 1]
            state_change, state_change_var, v = self.predict_next_state_change(input_mean, input_var, iK, beta)
            new_mu = mu_list[-1] + state_change
            v_state = v[:, :self.num_states]
            new_var = state_change_var + v_state @ var_list[-1] @ v_state.T
            mu_list.append(new_mu)
            var_list.append(new_var)
        states_mu_pred = torch.stack(mu_list)
        states_var_pred = torch.stack(var_list)
        costs_traj, costs_traj_var = self.compute_cost(states_mu_pred[:-1], states_var_pred[:-1], actions)
        cost_traj_final, costs_traj_var_final = self.compute_cost_terminal(states_mu_pred[-1], states_var_pred[-1])
        costs_traj = torch.cat((costs_traj, cost_traj_final.unsqueeze(0)), 0)
        costs_traj_var = torch.cat((costs_traj_var, costs_traj_var_final.unsqueeze(0)), 0)
        costs_traj_lcb = costs_traj - self.exploration_factor * torch.sqrt(costs_traj_var)
        return states_mu_pred, states_var_pred, costs_traj, costs_traj_var, costs_traj_lcb

    def compute_mean_lcb_trajectory(self, actions: np.ndarray, obs_mu: Tensor, obs_var: Tensor, iK: Tensor, beta: Tensor) -> Tuple[float, np.ndarray]:
        """
        Compute the mean lower confidence bound (LCB) of the trajectory cost and its gradients.
        """
        actions_tensor = torch.Tensor(actions.reshape(self.len_horizon, -1))
        actions_tensor.requires_grad = True
        mean_states_pred, s_states_pred, costs_traj, costs_traj_var, costs_traj_lcb = \
            self.predict_trajectory(actions_tensor, obs_mu, obs_var, iK, beta)
        mean_cost_traj_lcb = costs_traj_lcb.mean()
        gradients = torch.autograd.grad(mean_cost_traj_lcb, actions_tensor, retain_graph=False)[0]
        self.cost_traj_mean_lcb = mean_cost_traj_lcb.detach()
        self.mu_states_pred = mean_states_pred.detach()
        self.costs_trajectory = costs_traj.detach()
        self.states_var_pred = s_states_pred.detach()
        self.costs_traj_var = costs_traj_var.detach()
        return mean_cost_traj_lcb.item(), gradients.flatten().detach().numpy()

    def compute_action(self, obs_mu: np.ndarray, obs_var: Optional[np.ndarray] = None) -> Tuple[
        np.ndarray, Dict[str, Any]]:
        """
        Optimize the action trajectory to minimize the mean lower bound of the predicted cost.
        Returns the first denormalized action along with iteration info.
        """
        self.check_and_close_processes()
        torch.set_num_threads(self.num_cores_main)
        with torch.no_grad():
            obs_mu_norm = self.to_normed_obs_tensor(obs_mu)
            obs_var_norm = self.obs_var_norm if obs_var is None else self.to_normed_var_tensor(obs_var)
            self.iK, self.beta = self.calculate_factorizations(self.x[self.idxs_mem_gp],
                                                               self.y[self.idxs_mem_gp],
                                                               self.models, self.DoF)
            init_actions = np.concatenate((self.actions_pred_previous_iter[1:],
                                           self.actions_pred_previous_iter[-1:]), axis=0).flatten()

        time_start = time.time()

        # Separate the optimizer method from its options.
        optim_options = self.params_actions_optimizer.copy()
        optim_method = optim_options.pop("action_optim_method", "L-BFGS-B")

        res = minimize(
            fun=self.compute_mean_lcb_trajectory,
            x0=init_actions,
            jac=True,
            args=(obs_mu_norm, obs_var_norm, self.iK, self.beta),
            method=optim_method,
            bounds=self.bounds,
            options=optim_options
        )
        time_end = time.time()
        print(f"Optimisation time for iteration: {time_end - time_start:.3f} s")
        actions_norm = res.x.reshape(self.len_horizon, -1)
        self.actions_pred_previous_iter = actions_norm.copy()
        with torch.no_grad():
            action_next = actions_norm[0]
            actions_norm_tensor = torch.Tensor(actions_norm)
            action_denorm = self.denorm_action(action_next)
            cost, cost_var = self.compute_cost(obs_mu_norm, obs_var_norm, actions_norm_tensor[0])
            states_std_pred = torch.diagonal(self.states_var_pred, dim1=-2, dim2=-1).sqrt()
            info_dict = {
                'iteration': self.n_iter_ctrl,
                'state': self.mu_states_pred[0],
                'predicted states': self.mu_states_pred[1:],
                'predicted states std': states_std_pred[1:],
                'predicted actions': actions_norm_tensor,
                'cost': cost.item(),
                'cost std': cost_var.sqrt().item(),
                'predicted costs': self.costs_trajectory[1:],
                'predicted costs std': self.costs_traj_var[1:].sqrt(),
                'mean predicted cost': self.costs_trajectory[1:].mean().item(),
                'mean predicted cost std': self.costs_traj_var[1:].sqrt().mean().item(),
                'lower bound mean predicted cost': self.cost_traj_mean_lcb.item()
            }
            for key, value in info_dict.items():
                self.info_iters.setdefault(key, []).append(value)
            self.n_iter_ctrl += 1
            return action_denorm, info_dict

    def add_memory(self, obs: np.ndarray, action: np.ndarray, obs_new: np.ndarray, reward: float,
                   predicted_state: Optional[Any] = None, predicted_state_std: Optional[Any] = None) -> None:
        """
        Add a new transition to memory. Expand storage if necessary.
        """
        if obs is None:
            return
        obs_norm = self.to_normed_obs_tensor(obs)
        action_norm = self.to_normed_action_tensor(action)
        obs_new_norm = self.to_normed_obs_tensor(obs_new)

        if self.len_mem >= self.x.shape[0]:
            self.x = torch.cat([self.x, torch.empty(self.points_add_mem_when_full, self.x.shape[1])])
            self.y = torch.cat([self.y, torch.empty(self.points_add_mem_when_full, self.y.shape[1])])
            self.rewards = torch.cat([self.rewards, torch.empty(self.points_add_mem_when_full)])

        self.x[self.len_mem, :obs_norm.shape[0] + action_norm.shape[0]] = torch.cat((obs_norm, action_norm))[None]
        self.y[self.len_mem] = obs_new_norm - obs_norm
        self.rewards[self.len_mem] = reward

        # For now, store every memory point.
        self.idxs_mem_gp.append(self.len_mem)
        self.len_mem += 1
        self.n_iter_obs += 1

        if self.len_mem % self.training_frequency == 0 and ('p_train' not in self.__dict__ or self.p_train._closed):
            self.p_train = self.ctx.Process(
                target=self.train,
                args=(
                    self.queue_train,
                    self.x[self.idxs_mem_gp],
                    self.y[self.idxs_mem_gp],
                    [model.state_dict() for model in self.models],
                    self.gp_constraints,
                    self.lr_train,
                    self.iter_train,
                    self.clip_grad_value,
                    self.print_train,
                    self.step_print_train,
                    self.DoF
                )
            )
            self.p_train.start()
            self.num_cores_main -= 1

    @staticmethod
    def train(queue: multiprocessing.Queue, train_inputs: Tensor, train_targets: Tensor,
              parameters: List[Dict[str, Any]], constraints_hyperparams: Dict[str, Any],
              lr_train: float, num_iter_train: int, clip_grad_value: float,
              print_train: bool = False, step_print_train: int = 25, dof: int = 0) -> None:
        import logging
        logger = logging.getLogger(__name__)

        torch.set_num_threads(1)
        start_time = time.time()

        models = create_models(train_inputs, train_targets, parameters, constraints_hyperparams, dof=dof)
        logger.info(f"Starting training for {len(models)} models on {train_inputs.shape[0]} training points.")

        best_outputscales = [model.covar_module.outputscale.detach() for model in models]
        best_noises = [model.likelihood.noise.detach() for model in models]
        best_lengthscales = [model.covar_module.base_kernel.lengthscale.detach() for model in models]
        previous_losses = torch.empty(len(models))

        for idx, model in enumerate(models):
            output = model(model.train_inputs[0])
            mll = gpytorch.mlls.ExactMarginalLogLikelihood(model.likelihood, model)
            previous_losses[idx] = -mll(output, model.train_targets)

        best_losses = previous_losses.clone()

        for idx, model in enumerate(models):
            logger.info(f"Training model {idx} started.")
            ls_bounds = model.covar_module.base_kernel.raw_lengthscale_constraint
            model.covar_module.base_kernel.lengthscale = (
                    0.5 * (ls_bounds.lower_bound + ls_bounds.upper_bound) +
                    0.1 * (ls_bounds.upper_bound - ls_bounds.lower_bound) * torch.rand_like(
                model.covar_module.base_kernel.lengthscale)
            )
            os_bounds = model.covar_module.raw_outputscale_constraint
            model.covar_module.outputscale = (
                    0.5 * (os_bounds.lower_bound + os_bounds.upper_bound) +
                    0.1 * (os_bounds.upper_bound - os_bounds.lower_bound) * torch.rand_like(
                model.covar_module.outputscale)
            )
            noise_bounds = model.likelihood.noise_covar.raw_noise_constraint
            model.likelihood.noise = (
                    0.5 * (noise_bounds.lower_bound + noise_bounds.upper_bound) +
                    0.1 * (noise_bounds.upper_bound - noise_bounds.lower_bound) * torch.rand_like(
                model.likelihood.noise)
            )
            mll = gpytorch.mlls.ExactMarginalLogLikelihood(model.likelihood, model)
            model.train()
            model.likelihood.train()
            optimizer = torch.optim.LBFGS(model.parameters(), lr=lr_train, line_search_fn='strong_wolfe')

            try:
                for i in range(num_iter_train):
                    def closure():
                        optimizer.zero_grad()
                        output = model(model.train_inputs[0])
                        loss = -mll(output, model.train_targets)
                        loss.backward()
                        torch.nn.utils.clip_grad_value_(model.parameters(), clip_grad_value)
                        return loss

                    loss = optimizer.step(closure)
                    if print_train and i % step_print_train == 0:
                        ls = model.covar_module.base_kernel.lengthscale.detach().numpy()
                        logger.info(f"Model {idx}, Iteration {i + 1}/{num_iter_train} - Loss: {loss.item():.5f}, "
                                    f"output_scale: {model.covar_module.outputscale.item():.5f}, "
                                    f"lengthscale: {ls}, noise: {model.likelihood.noise.item() ** 0.5:.5f}")
                    if loss < best_losses[idx]:
                        best_losses[idx] = loss.item()
                        best_lengthscales[idx] = model.covar_module.base_kernel.lengthscale.clone()
                        best_noises[idx] = model.likelihood.noise.clone()
                        best_outputscales[idx] = model.covar_module.outputscale.clone()
            except Exception as e:
                logger.exception(f"[ERROR] Exception in training model {idx}: {e}")
            model_time = time.time() - start_time
            logger.info(f"Model {idx} training complete in {model_time:.6f} seconds. "
                        f"Final output_scale: {best_outputscales[idx].detach().cpu().numpy()}, "
                        f"lengthscales: {best_lengthscales[idx].detach().cpu().numpy()}, "
                        f"noise: {best_noises[idx].detach().cpu().numpy()}")

        logger.info(f"Overall training completed in {time.time() - start_time:.6f} seconds.")
        logger.info(
            f"Previous MLL: {previous_losses.detach().cpu().numpy()}, New MLL: {best_losses.detach().cpu().numpy()}")

        params_dict_list = []
        for idx in range(len(models)):
            params_dict_list.append({
                'covar_module.base_kernel.lengthscale': best_lengthscales[idx].detach().cpu().numpy(),
                'covar_module.outputscale': best_outputscales[idx].detach().cpu().numpy(),
                'likelihood.noise': best_noises[idx].detach().cpu().numpy()
            })
        queue.put(params_dict_list)

    def check_and_close_processes(self) -> None:
        """
        Check and close inactive parallel training processes.
        """
        if 'p_train' in self.__dict__ and not self.p_train._closed and (not self.p_train.is_alive()):
            params_dict_list = self.queue_train.get()
            self.p_train.join()
            for idx, model in enumerate(self.models):
                model.initialize(**params_dict_list[idx])
            self.p_train.close()
            self.iK, self.beta = self.calculate_factorizations(self.x[self.idxs_mem_gp],
                                                               self.y[self.idxs_mem_gp],
                                                               self.models, self.DoF)
            self.num_cores_main += 1