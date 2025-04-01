import time
from typing import Any, Dict, Tuple, List, Optional, Union

import numpy as np
import torch
from torch import Tensor
from scipy.optimize import minimize
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm

# Use TkAgg for external windows
matplotlib.use('TkAgg')
import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Visualization constants
matplotlib.rc("font", size="6")
ALPHA_CONFIDENCE_BOUNDS = 0.3
FONTSIZE = 6
MEAN_PRED_COST_INIT = 0
STD_MEAN_PRED_COST_INIT = 1
N_COLORS = 10
colors_map = cm.rainbow(np.linspace(0, 1, N_COLORS))


def init_visu_and_folders(env, num_steps, params_controller_dict):
    """
    Create and return a live plot object for real-time visualization.
    """
    live_plot_obj = LivePlotSequential(
        num_steps,
        env.observation_space,
        env.action_space,
        step_pred=1,
    )
    return live_plot_obj


def close_run(ctrl_obj, env):
    """
    Close environment and stop any active processes.
    """
    env.__exit__()
    ctrl_obj.check_and_close_processes()



def adjust_params_for_DoF(params_controller_dict: Dict, DoF: int) -> None:
    for var in ["gp_init"]:
        for key in params_controller_dict[var]:
            params_controller_dict[var][key] = params_controller_dict[var][key][:DoF]
    controller_keys = ["target_state_norm", "weight_state", "weight_state_terminal",
                       "target_action_norm", "weight_action", "obs_var_norm"]
    for key in controller_keys:
        params_controller_dict["controller"][key] = params_controller_dict["controller"][key][:DoF]
    params_controller_dict["controller"]["DoF"] = DoF

class LivePlotSequential:
    def __init__(self, num_steps_total, obs_space, action_space, step_pred, mul_std_bounds=3, fontsize=6):
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
        self.action_space = action_space

        self.num_points_show = 0
        self.lines_states = [self.axes[0].plot([], [], label=f"state{idx}", color=colors_map[idx])[0]
                             for idx in range(obs_space.shape[0])]
        self.line_cost = self.axes[2].plot([], [], label="reward", color="k")[0]
        self.lines_actions = [self.axes[1].step([], [], label=f"action{idx}", color=colors_map[idx])[0]
                              for idx in range(action_space.shape[0])]
        self.line_mean_costs_pred = self.axes[2].plot([], [], label="mean predicted reward", color="orange")[0]
        self.lines_states_pred = [self.axes[0].plot([], [], label=f"predicted_state{idx}",
                                                    color=colors_map[idx], linestyle="dashed")[0]
                                  for idx in range(obs_space.shape[0])]
        self.lines_actions_pred = [self.axes[1].step([], [], label=f"predicted_action{idx}",
                                                      color=colors_map[idx], linestyle="dashed")[0]
                                   for idx in range(action_space.shape[0])]
        self.line_costs_pred = self.axes[2].plot([], [], label="predicted reward", color="k", linestyle="dashed")[0]

        for ax in self.axes:
            ax.legend(fontsize=fontsize)
            ax.grid()
        plt.show(block=False)

    def update(self, obs, action, cost, info_dict=None):
        obs_norm = (obs - self.min_obs) / (self.max_obs - self.min_obs)
        action_norm = (action - self.min_action) / (self.action_space.high - self.action_space.low)
        self.states[self.num_points_show] = obs_norm
        self.costs[self.num_points_show] = -cost

        update_limits = False
        if np.min(obs_norm) < self.min_state:
            self.min_state = np.min(obs_norm)
            update_limits = True
        if np.max(obs_norm) > self.max_state:
            self.max_state = np.max(obs_norm)
            update_limits = True
        if update_limits:
            self.axes[0].set_ylim(self.min_state, self.max_state)

        idxs = np.arange(self.num_points_show + 1)
        for ax in self.axes:
            for coll in ax.collections:
                coll.remove()
        for idx in range(len(obs_norm)):
            self.lines_states[idx].set_data(idxs, self.states[:self.num_points_show + 1, idx])
        self.actions[self.num_points_show] = action_norm
        for idx in range(len(action_norm)):
            self.lines_actions[idx].set_data(idxs, self.actions[:self.num_points_show + 1, idx])
        self.line_cost.set_data(idxs, self.costs[:self.num_points_show + 1])

        if info_dict is not None:
            if "mean predicted cost" in info_dict:
                mean_costs_pred = -info_dict["mean predicted cost"]
                mean_costs_std_pred = info_dict["mean predicted cost std"]
                states_pred = info_dict["predicted states"]
                states_std_pred = info_dict["predicted states std"]
                actions_pred = info_dict["predicted actions"]
                costs_pred = -info_dict["predicted costs"]
                costs_std_pred = info_dict["predicted costs std"]
                idx_prev = (idxs[-1] // self.step_pred) * self.step_pred
                idxs_future = np.arange(idx_prev, idx_prev + self.step_pred * (len(states_pred) + 1), self.step_pred)
                self.mean_costs_pred[self.num_points_show] = mean_costs_pred
                self.mean_costs_std_pred[self.num_points_show] = mean_costs_std_pred
                for idx in range(len(obs_norm)):
                    future_states = np.concatenate(([self.states[idx_prev, idx]], states_pred[:, idx]))
                    self.lines_states_pred[idx].set_data(idxs_future, future_states)
                    future_std = np.concatenate(([0], states_std_pred[:, idx]))
                    self.axes[0].fill_between(idxs_future,
                                              future_states - future_std * self.mul_std_bounds,
                                              future_states + future_std * self.mul_std_bounds,
                                              facecolor=colors_map[idx],
                                              alpha=ALPHA_CONFIDENCE_BOUNDS)
                for idx in range(len(action_norm)):
                    future_actions = np.concatenate(([self.actions[idx_prev, idx]], actions_pred[:, idx]))
                    self.lines_actions_pred[idx].set_data(idxs_future, future_actions)
                future_costs = np.concatenate(([self.costs[idx_prev]], costs_pred))
                self.line_costs_pred.set_data(idxs_future, future_costs)
                future_cost_std = np.concatenate(([0], costs_std_pred))
                self.axes[2].fill_between(idxs_future,
                                          future_costs - future_cost_std * self.mul_std_bounds,
                                          future_costs + future_cost_std * self.mul_std_bounds,
                                          facecolor="black", alpha=ALPHA_CONFIDENCE_BOUNDS)
            elif "predicted total cost" in info_dict:
                mean_costs_pred = info_dict["predicted total cost"]
                self.mean_costs_pred[self.num_points_show] = mean_costs_pred
                self.mean_costs_std_pred[self.num_points_show] = 0
        else:
            if self.num_points_show == 0:
                self.mean_costs_pred[0] = MEAN_PRED_COST_INIT
                self.mean_costs_std_pred[0] = STD_MEAN_PRED_COST_INIT
            else:
                self.mean_costs_pred[self.num_points_show] = self.mean_costs_pred[self.num_points_show - 1]
                self.mean_costs_std_pred[self.num_points_show] = self.mean_costs_std_pred[self.num_points_show - 1]

        self.line_mean_costs_pred.set_data(idxs, self.mean_costs_pred[:self.num_points_show + 1])
        self.axes[2].set_ylim(np.min([np.min(self.mean_costs_pred[:self.num_points_show + 1]),
                                      np.min(self.costs[:self.num_points_show + 1])]) * 1.1, 0.5)
        self.axes[2].fill_between(
            idxs,
            self.mean_costs_pred[:self.num_points_show + 1] - self.mean_costs_std_pred[:self.num_points_show + 1] * self.mul_std_bounds,
            self.mean_costs_pred[:self.num_points_show + 1] + self.mean_costs_std_pred[:self.num_points_show + 1] * self.mul_std_bounds,
            facecolor="orange", alpha=ALPHA_CONFIDENCE_BOUNDS
        )
        self.axes[2].set_ylim(-3, 0.1)
        self.fig.canvas.draw()
        plt.pause(0.01)
        self.num_points_show += 1


class LinearDynamicsModel:
    def __init__(self):
        # Linear model: x_next = A x + B u
        self.A = None  # (state_dim, state_dim)
        self.B = None  # (state_dim, action_dim)
        self.residual_cov = None  # Residual covariance matrix

    def fit(self, X: np.ndarray, U: np.ndarray, X_next: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Fit the linear dynamics model using least squares regression.
        Args:
            X: Current states (N, state_dim)
            U: Actions (N, action_dim)
            X_next: Next states (N, state_dim)
        Returns:
            A, B, and the residual covariance matrix.
        """
        N, state_dim = X.shape
        action_dim = U.shape[1]
        Z = np.hstack([X, U])
        Theta, residuals, rank, s = np.linalg.lstsq(Z, X_next, rcond=None)
        self.A = Theta[:state_dim, :].T
        self.B = Theta[state_dim:, :].T
        X_next_pred = Z @ Theta
        errors = X_next - X_next_pred
        self.residual_cov = np.cov(errors.T)
        return self.A, self.B, self.residual_cov

    def predict(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """
        Predict next state given current state and action.
        """
        return x @ self.A.T + u @ self.B.T

    def predict_with_uncertainty(self, x: np.ndarray, u: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        pred = self.predict(x, u)
        return pred, self.residual_cov

class BaseControllerObject:
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

    def add_memory(self, observation, action, new_observation, reward, **kwargs) -> None:
        raise NotImplementedError()

    def compute_action(self, observation, s_observation) -> Any:
        raise NotImplementedError()

    def train(self, *args, **kwargs) -> None:
        raise NotImplementedError()

torch.set_default_dtype(torch.double)

class LinearMPCController(BaseControllerObject):
    def __init__(self, observation_space: Any, action_space: Any, params_dict: Dict[str, Any]):
        self.params = params_dict
        self.target_state = torch.Tensor(params_dict['controller']['target_state_norm'])
        self.weight_state = torch.diag(torch.Tensor(params_dict['controller']['weight_state']))
        self.weight_state_terminal = torch.diag(torch.Tensor(params_dict['controller']['weight_state_terminal']))
        self.len_horizon = params_dict['controller']['len_horizon']
        self.exploration_factor = params_dict['controller']['exploration_factor']
        self.error_cov_prior = torch.eye(observation_space.shape[0]) * 0.01
        self.barrier_weight = params_dict['controller']['barrier_weight']
        self.DoF = params_dict['controller']['DoF']

        self.lr_train = params_dict['train']['lr_train']
        self.iter_train = params_dict['train']['iter_train']
        self.training_frequency = params_dict['train']['training_frequency']

        super().__init__(observation_space, action_space)

        self.memory_states: List[np.ndarray] = []
        self.memory_actions: List[np.ndarray] = []
        self.memory_next_states: List[np.ndarray] = []

        self.params_actions_optimizer = params_dict['actions_optimizer']
        self.bounds = [(0, 1)] * (self.num_actions * self.len_horizon)
        self.actions_pred_previous_iter = np.random.uniform(0, 1, size=(self.len_horizon, self.num_actions))
        self.n_iter_ctrl = 0
        self.info_iters = {}

        self.model = LinearDynamicsModel()
        self.state_cost_weight = self.weight_state

    @staticmethod
    def preprocess_params(params_dict: Dict[str, Any]) -> Dict[str, Any]:
        controller = params_dict['controller']
        for key in ['target_state_norm', 'weight_state', 'weight_state_terminal', 'target_action_norm', 'weight_action', 'obs_var_norm']:
            controller[key] = torch.Tensor(controller[key])
        return params_dict

    def to_normed_obs_tensor(self, obs: np.ndarray) -> Tensor:
        return torch.Tensor((obs - self.obs_space.low) / (self.obs_space.high - self.obs_space.low))

    def to_normed_action_tensor(self, action: np.ndarray) -> Tensor:
        return torch.Tensor((action - self.action_space.low) / (self.action_space.high - self.action_space.low))

    def denorm_action(self, action_norm: Union[np.ndarray, Tensor]) -> np.ndarray:
        return (action_norm * (self.action_space.high - self.action_space.low) + self.action_space.low)

    def barrier_cost(self, state: Tensor, alpha: float) -> Tensor:
        epsilon = 1e-6
        state = torch.clamp(state, epsilon, 1 - epsilon)
        return -alpha * (torch.log(state) + torch.log(1 - state)).sum(-1)

    def compute_cost_unnormalized(self, obs: np.ndarray, action: np.ndarray, obs_var: Optional[np.ndarray] = None) -> Tuple[float, float]:
        obs_norm = self.to_normed_obs_tensor(obs)
        error = obs_norm - self.target_state
        cost = (error.unsqueeze(0) @ self.weight_state @ error.unsqueeze(1)).squeeze()
        cost += self.barrier_cost(obs_norm, self.barrier_weight)
        return cost.item(), 0.0

    def train(self) -> None:
        if len(self.memory_states) < 2:
            logger.info("Not enough data to train the linear dynamics model.")
            return
        X = np.array(self.memory_states)
        U = np.array(self.memory_actions)
        X_next = np.array(self.memory_next_states)
        A, B, residual_cov = self.model.fit(X, U, X_next)
        logger.info(f"Linear model trained. A: {A}, B: {B}, residual covariance: {residual_cov}")

    def add_memory(self, obs: np.ndarray, action: np.ndarray, obs_new: np.ndarray, reward: float, **kwargs) -> None:
        self.memory_states.append(obs)
        self.memory_actions.append(action)
        self.memory_next_states.append(obs_new)

    def predict_trajectory(self, actions: Tensor, x0: Tensor, P0: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        state_dim = x0.shape[0]
        mu_list = [x0]
        P_list = [P0]
        costs = []
        # Convert learned A and B to torch tensors (no gradients).
        A_tensor = torch.tensor(self.model.A, dtype=torch.double, requires_grad=False)
        B_tensor = torch.tensor(self.model.B, dtype=torch.double, requires_grad=False)
        Q = torch.tensor(self.model.residual_cov, dtype=torch.double, requires_grad=False)
        for t in range(self.len_horizon):
            u = actions[t]
            # Compute next state as a differentiable operation.
            x_next = mu_list[-1] @ A_tensor.T + u @ B_tensor.T
            P_next = A_tensor @ P_list[-1] @ A_tensor.T + Q
            mu_list.append(x_next)
            P_list.append(P_next)
            error = x_next - self.target_state
            step_cost = error.unsqueeze(0) @ self.weight_state @ error.unsqueeze(1)
            costs.append(step_cost.squeeze())
        states_mu_pred = torch.stack(mu_list)
        states_cov_pred = torch.stack(P_list)
        costs = torch.stack(costs)
        return states_mu_pred, states_cov_pred, costs

    def compute_mean_lcb_trajectory(self, actions: np.ndarray, x0: Tensor, P0: Tensor) -> Tuple[float, np.ndarray]:
        actions_tensor = torch.Tensor(actions.reshape(self.len_horizon, -1))
        actions_tensor.requires_grad = True
        states_mu, states_cov, costs = self.predict_trajectory(actions_tensor, x0, P0)
        total_cost = torch.sum(costs)
        uncertainty = sum(torch.trace(P) for P in states_cov[1:])
        total_cost_lcb = total_cost - self.exploration_factor * uncertainty
        gradients = torch.autograd.grad(total_cost_lcb, actions_tensor, retain_graph=False)[0]
        return total_cost_lcb.item(), gradients.flatten().detach().numpy()

    def compute_action(self, obs: np.ndarray, obs_var: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        # Train the dynamics model if sufficient data exists.
        self.train()
        x0 = self.to_normed_obs_tensor(obs)
        P0 = self.error_cov_prior
        init_actions = self.actions_pred_previous_iter.flatten()
        t_start = time.time()
        optim_options = self.params_actions_optimizer.copy()
        # Remove unsupported keys for trust-constr optimizer.
        for key in ["action_optim_method", "maxcor", "iprint"]:
            optim_options.pop(key, None)
        res = minimize(
            fun=self.compute_mean_lcb_trajectory,
            x0=init_actions,
            jac=True,
            args=(x0, P0),
            method='trust-constr',
            bounds=self.bounds,
            options=optim_options
        )
        t_end = time.time()
        print(f"Optimisation time for iteration: {t_end - t_start:.3f} s")
        actions_norm = res.x.reshape(self.len_horizon, -1)
        self.actions_pred_previous_iter = actions_norm.copy()
        action_next = actions_norm[0]
        action_denorm = self.denorm_action(action_next)
        states_mu_pred, states_cov_pred, costs_pred = self.predict_trajectory(torch.Tensor(actions_norm), x0, P0)
        # Compute predicted standard deviation for each state along the horizon.
        predicted_states = states_mu_pred[1:].detach().numpy()
        predicted_states_std = np.array([np.sqrt(np.diag(P.detach().numpy())) for P in states_cov_pred[1:]])
        predicted_actions = actions_norm  # already normalized; plotting will scale using the action space
        predicted_costs = costs_pred.detach().numpy()
        # For simplicity, we use zeros as the cost std estimates.
        predicted_costs_std = np.zeros_like(predicted_costs)
        total_cost_pred = costs_pred.sum().item()
        info_dict = {
            'iteration': self.n_iter_ctrl,
            'predicted total cost': total_cost_pred,
            'predicted states': predicted_states,
            'predicted states std': predicted_states_std,
            'predicted actions': predicted_actions,
            'predicted costs': predicted_costs,
            'predicted costs std': predicted_costs_std,
            'mean predicted cost': total_cost_pred,
            'mean predicted cost std': 0.0   # <-- New key added here
        }
        self.n_iter_ctrl += 1
        return action_denorm, info_dict

    def check_and_close_processes(self) -> None:
        pass

def init_graphics_and_controller(env: Any, num_steps: int, params_controller_dict: Dict) -> Tuple[Any, LinearMPCController]:
    live_plot_obj = init_visu_and_folders(env, num_steps, params_controller_dict)
    ctrl_obj = LinearMPCController(
        observation_space=env.observation_space,
        action_space=env.action_space,
        params_dict=params_controller_dict,
    )
    return live_plot_obj, ctrl_obj

