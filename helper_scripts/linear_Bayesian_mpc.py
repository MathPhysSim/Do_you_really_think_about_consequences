import time
from typing import Any, Dict, Tuple, List, Optional, Union
import cvxpy as cp

import numpy as np
import torch
from torch import Tensor
from scipy.optimize import minimize
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm

from environment.environment_awake_steering import AwakeSteering

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
    live_plot_obj = LivePlotSequential(
        num_steps,
        env.observation_space,
        env.action_space,
        step_pred=1,
    )
    return live_plot_obj


def close_run(ctrl_obj, env):
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
        # [unchanged plotting code...]
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
        # [unchanged plotting update code]
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
                self.line_costs_pred.set_data(idxs_future, np.full_like(idxs_future, mean_costs_pred))
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
            self.mean_costs_pred[:self.num_points_show + 1] - self.mean_costs_std_pred[
                                                              :self.num_points_show + 1] * self.mul_std_bounds,
            self.mean_costs_pred[:self.num_points_show + 1] + self.mean_costs_std_pred[
                                                              :self.num_points_show + 1] * self.mul_std_bounds,
            facecolor="orange", alpha=ALPHA_CONFIDENCE_BOUNDS
        )
        self.axes[2].set_ylim(-3, 0.1)
        self.fig.canvas.draw()
        plt.pause(0.01)
        self.num_points_show += 1

env = AwakeSteering()

class BayesianLinearDynamicsModel:
    def __init__(self, state_dim: int, action_dim: int, noise_var: float = 1e-2,
                 prior_mean: Optional[np.ndarray] = None, prior_cov: Optional[np.ndarray] = None,
                 regularization: float = 1e-4):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.input_dim = state_dim + action_dim
        self.noise_var = noise_var
        self.regularization = regularization  # Added regularization term
        if prior_mean is None:
            self.prior_mean = np.zeros((self.input_dim, state_dim))
        else:
            self.prior_mean = prior_mean
        if prior_cov is None:
            self.prior_cov = np.eye(self.input_dim) * 1e2
        else:
            self.prior_cov = prior_cov
        self.posterior_mean = None
        self.posterior_cov = None
        self.A = None
        self.B = None
        self.residual_cov = np.eye(state_dim) * self.noise_var

    def fit(self, X: np.ndarray, U: np.ndarray, X_next: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        N, state_dim = X.shape
        Z = np.hstack([X, U])
        Y = X_next
        Sigma0_inv = np.linalg.inv(self.prior_cov + self.regularization * np.eye(self.input_dim))
        Sigma_N_inv = Sigma0_inv + (1.0 / self.noise_var) * (Z.T @ Z)
        self.posterior_cov = np.linalg.inv(Sigma_N_inv)
        self.posterior_mean = self.posterior_cov @ (Sigma0_inv @ self.prior_mean + (1.0 / self.noise_var) * Z.T @ Y)
        self.A = self.posterior_mean[:state_dim, :].T
        print(np.linalg.norm(np.eye(state_dim)-self.A))
        self.B = self.posterior_mean[state_dim:, :].T
        print(np.linalg.norm(env.rmatrix-self.B))
        self.residual_cov = np.eye(state_dim) * self.noise_var
        return self.A, self.B, self.posterior_cov

    def predict(self, x: np.ndarray, u: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        z = np.hstack([x, u])
        mean = z @ self.posterior_mean
        var = z.T @ self.posterior_cov @ z + self.noise_var
        return mean, var


class BaseControllerObject:
    def __init__(self, observation_space: Any, action_space: Any, n_points_init_memory: int = 1000, env=None):
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
        self.env = env

    def add_memory(self, observation, action, new_observation, reward, **kwargs) -> None:
        raise NotImplementedError()

    def compute_action(self, observation, s_observation) -> Any:
        raise NotImplementedError()

    def train(self, *args, **kwargs) -> None:
        raise NotImplementedError()


torch.set_default_dtype(torch.double)


class LinearMPCController(BaseControllerObject):
    def __init__(self, observation_space: Any, action_space: Any, params_dict: Dict[str, Any], env=None):
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
        super().__init__(observation_space, action_space, env=env)
        self.memory_states: List[np.ndarray] = []
        self.memory_actions: List[np.ndarray] = []
        self.memory_next_states: List[np.ndarray] = []
        self.params_actions_optimizer = params_dict['actions_optimizer']
        self.bounds = [(0, 1)] * (self.num_actions * self.len_horizon)
        self.actions_pred_previous_iter = np.random.uniform(0, 1, size=(self.len_horizon, self.num_actions))
        self.n_iter_ctrl = 0
        self.info_iters = {}
        self.model = BayesianLinearDynamicsModel(state_dim=observation_space.shape[0],
                                                 action_dim=action_space.shape[0],
                                                 noise_var=1e-2)
        self.state_cost_weight = self.weight_state
        self.target_state = torch.Tensor(params_dict['controller']['target_state_norm'])

    def to_normed_obs_tensor(self, obs: np.ndarray) -> Tensor:
        return torch.Tensor((obs - self.obs_space.low) / (self.obs_space.high - self.obs_space.low))

    def to_normed_action_tensor(self, action: np.ndarray) -> Tensor:
        return torch.Tensor((action - self.action_space.low) / (self.action_space.high - self.action_space.low))

    def denorm_action(self, action_norm: Union[np.ndarray, Tensor]) -> np.ndarray:
        return (action_norm * (self.action_space.high - self.action_space.low) + self.action_space.low)

    def compute_cost_unnormalized(self, obs: np.ndarray, action: np.ndarray, obs_var: Optional[np.ndarray] = None) -> \
    Tuple[float, float]:
        obs_norm = self.to_normed_obs_tensor(obs)
        error = obs_norm - self.target_state
        cost = (error.unsqueeze(0) @ self.weight_state @ error.unsqueeze(1)).squeeze()
        return cost.item(), 0.0

    def train(self) -> None:
        if len(self.memory_states) < 2:
            logger.info("Not enough data to train the Bayesian dynamics model.")
            return
        X = np.array(self.memory_states)
        U = np.array(self.memory_actions)
        X_next = np.array(self.memory_next_states)
        A, B, post_cov = self.model.fit(X, U, X_next)
        # # Override the estimated matrices with the known physical dynamics
        # A = np.eye(self.DoF)  # A_phys
        # B = self.env.rmatrix  # B_phys
        # post_cov = np.zeros_like(post_cov)
        # logger.info(f"Bayesian linear model trained. A: {A}")
        # logger.info(f"Bayesian linear model trained. B: {B}")
        # logger.info(f"Bayesian linear model trained. post_cov: {post_cov}")

    def add_memory(self, obs: np.ndarray, action: np.ndarray, obs_new: np.ndarray, reward: float, **kwargs) -> None:

        obs_norm = self.to_normed_obs_tensor(obs)
        action_norm = self.to_normed_action_tensor(action)
        obs_new_norm = self.to_normed_obs_tensor(obs_new)

        self.memory_states.append(obs_norm)
        self.memory_actions.append(action_norm)
        self.memory_next_states.append(obs_new_norm)

    def compute_expected_cost(self, mu: Tensor, P: Tensor, target: Tensor, W: Tensor) -> Tuple[Tensor, Tensor]:
        n = mu.shape[0]
        cost_quad = (mu - target).unsqueeze(0) @ W @ (mu - target).unsqueeze(1)
        cost_quad = cost_quad.squeeze() + torch.trace(W @ P)
        cost_mean = cost_quad / n
        grad = 2 * (W @ (mu - target)) / n
        cost_var = grad @ P @ grad
        return cost_mean, cost_var

    def predict_trajectory(self, actions: Tensor, x0: Tensor, P0: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Propagate the state distribution over the horizon and compute stage costs.
        A terminal cost is added at the final state using the terminal weight.
        NOTE: This prediction is performed in the normalized space.
        """
        state_dim = x0.shape[0]
        mu_list = [x0]
        P_list = [P0]
        costs_mean = []
        costs_var = []
        A_tensor = torch.tensor(self.model.A, dtype=torch.double, requires_grad=False)  # A_phys (I)
        B_tensor = torch.tensor(self.model.B, dtype=torch.double, requires_grad=False)  # B_phys (rmatrix)
        Q = torch.tensor(self.model.residual_cov, dtype=torch.double, requires_grad=False)

        # --- Adjust dynamics to normalized coordinates ---
        # Given physical dynamics: x_phys^{+} = A_phys * x_phys + B_phys * u_phys,
        # and the normalization: x_norm = (x_phys + 1) / 2, u_norm = (u_phys + 1) / 2,
        # we have: x_norm^{+} = A_phys * x_norm + B_phys * u_norm + c,
        # with c = (1 - (A_phys+B_phys)*1) / 2, where 1 is a vector of ones.
        ones_x = np.ones(self.obs_space.shape[0])
        ones_u = np.ones(self.action_space.shape[0])
        # c = (ones_x - ((np.eye(self.DoF) + self.env.rmatrix) @ ones_x)) / 2.0
        c = (ones_x - (self.model.A + self.model.B) @ ones_x) / 2.0
        # Convert c to torch tensor:
        c_tensor = torch.tensor(c, dtype=torch.double)
        # -----------------------------------------------------

        # Propagate for the horizon:
        for t in range(self.len_horizon):
            u = actions[t]
            x_next = mu_list[-1] @ A_tensor.T + u @ B_tensor.T + c_tensor
            P_next = A_tensor @ P_list[-1] @ A_tensor.T + Q
            mu_list.append(x_next)
            P_list.append(P_next)
            step_mean, step_var = self.compute_expected_cost(x_next, P_next, self.target_state, self.weight_state)
            costs_mean.append(step_mean)
            costs_var.append(step_var)

        states_mu_pred = torch.stack(mu_list)
        states_cov_pred = torch.stack(P_list)
        total_stage_cost_mean = torch.stack(costs_mean).sum()
        total_stage_cost_var = torch.stack(costs_var).sum()
        terminal_cost_mean, terminal_cost_var = self.compute_expected_cost(
            states_mu_pred[-1], states_cov_pred[-1], self.target_state, self.weight_state_terminal
        )
        total_cost_mean = total_stage_cost_mean + terminal_cost_mean
        total_cost_var = total_stage_cost_var + terminal_cost_var
        return states_mu_pred, states_cov_pred, total_cost_mean, total_cost_var

    def compute_mean_lcb_trajectory(self, actions: np.ndarray, x0: Tensor, P0: Tensor) -> Tuple[float, np.ndarray]:
        actions_tensor = torch.Tensor(actions.reshape(self.len_horizon, -1))
        actions_tensor.requires_grad = True
        states_mu, states_cov, total_cost_mean, total_cost_var = self.predict_trajectory(actions_tensor, x0, P0)
        total_cost_lcb = total_cost_mean - self.exploration_factor * torch.sqrt(total_cost_var + 1e-8)
        gradients = torch.autograd.grad(total_cost_lcb, actions_tensor, retain_graph=False)[0]
        return total_cost_lcb.item(), gradients.flatten().detach().numpy()

    def compute_action(self, obs: np.ndarray, obs_var: Optional[np.ndarray] = None) -> Tuple[
        np.ndarray, Dict[str, Any]]:
        self.train()
        # Get current state in normalized space (assumed to be in [0,1])
        x0 = self.to_normed_obs_tensor(obs).numpy()
        P0 = self.error_cov_prior.numpy()  # Not used in QP formulation here

        T = self.len_horizon
        n = self.obs_space.shape[0]
        m = self.action_space.shape[0]

        # Use the physical dynamics matrices (in our case A_phys = I, B_phys = self.env.rmatrix)
        # A_phys = np.eye(self.DoF)
        A_phys = self.model.A
        # B_phys = self.env.rmatrix
        B_phys = self.model.B

        # Convert them to normalized dynamics.
        A_norm = A_phys  # remains identity.
        # The bias term:
        ones_x = np.ones(self.obs_space.shape[0])
        ones_u = np.ones(self.action_space.shape[0])
        c = (ones_x - ((A_phys + B_phys) @ ones_x)) / 2.0  # c = (1 - (I+B_phys)*ones)/2

        # Here we assume that the cost is defined in normalized space.
        Q = self.weight_state.numpy()
        Q_T = self.weight_state_terminal.numpy()
        target = self.target_state.numpy()

        # Create CVXPY variables for state and control trajectories in normalized space.
        x = cp.Variable((T + 1, n))
        u = cp.Variable((T, m))

        # Define the QP cost and constraints.
        cost = 0
        constraints = [x[0] == x0]
        for t in range(T):
            constraints += [x[t + 1] == A_norm @ x[t] + B_phys @ u[t] + c]
            cost += cp.quad_form(x[t] - target, Q)
        cost += cp.quad_form(x[T] - target, Q_T)
        constraints += [u >= 0, u <= 1]

        prob = cp.Problem(cp.Minimize(cost), constraints)
        prob.solve(solver=cp.OSQP)
        if prob.status not in ["optimal", "optimal_inaccurate"]:
            raise ValueError("The QP solver did not find an optimal solution.")

        action_norm = u.value[0]
        action_denorm = self.denorm_action(action_norm)
        info_dict = {'qp_cost': prob.value}
        return action_denorm, info_dict

    def check_and_close_processes(self) -> None:
        pass


def init_graphics_and_controller(env: Any, num_steps: int, params_controller_dict: Dict) -> Tuple[
    Any, LinearMPCController]:
    live_plot_obj = init_visu_and_folders(env, num_steps, params_controller_dict)
    ctrl_obj = LinearMPCController(
        observation_space=env.observation_space,
        action_space=env.action_space,
        params_dict=params_controller_dict,
    )
    return live_plot_obj, ctrl_obj