import multiprocessing
import time
from typing import Tuple

import numpy as np
import torch
from scipy.optimize import minimize


import gpytorch


class ExactGPModelMonoTask(gpytorch.models.ExactGP):
    """
    A single-task Exact Gaussian Process Model using GPyTorch.

    Parameters:
    - train_x (Tensor): Training input data.
    - train_y (Tensor): Training target data.
    - dim_input (int): Dimensionality of the input features.
    - likelihood (gpytorch.likelihoods.Likelihood, optional): Gaussian likelihood function. Defaults to GaussianLikelihood.
    - kernel (gpytorch.kernels.Kernel, optional): Covariance kernel. Defaults to RBFKernel with ARD.
    """

    def __init__(self, train_x, train_y, dim_input, likelihood=None, kernel=None):
        likelihood = (
            likelihood
            if likelihood is not None
            else gpytorch.likelihoods.GaussianLikelihood()
        )
        super().__init__(train_x, train_y, likelihood)

        self.mean_module = gpytorch.means.ZeroMean()
        self.covar_module = (
            kernel
            if kernel is not None
            else gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.RBFKernel(ard_num_dims=dim_input)
            )
        )

    def forward(self, x):
        """
        Performs a forward pass through the model.

        Parameters:
        - x (Tensor): Input data for prediction.

        Returns:
        - MultivariateNormal: Predictive distribution.
        """
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

def create_models(
    train_inputs,
    train_targets,
    params,
    constraints_gp,
    dof =10
):
    """
    Define gaussian process models used for predicting state transition,
    using constraints and init values for (outputscale, noise, lengthscale).

    Args:
            train_inputs (torch.Tensor or None): Input values in the memory of the gps
            train_targets (torch.Tensor or None): target values in the memory of the gps.
                                                                                            Represent the change in state values
            params (dict or list of dict): Value of the hyper-parameters of the gaussian processes.
                                                                            Dict type is used to create the models with init values from the json file.
                                                                            List of dict type is used to create the models
                                                                                                    with exported parameters, such as in the parallel training process.
            constraints_gp (dict): See the ReadMe about parameters for information about keys
            num_models (int or None): Must be provided when train_inputs or train_targets are None.
                                                                    The number of models should be equal to the dimension of state,
                                                                    so that the transition for each state can be predicted with a different gp.
                                                                    Default=None
            num_inputs (int or None): Must be provided when train_inputs or train_targets are None.
                                                                    The number of inputs should be equal to the sum of the dimension of state
                                                                    and dimension of action. Default=None
            include_time (bool): If True, gp will have one additional input corresponding to the time of the observation.
                                                            This is usefull if the env change with time,
                                                            as more recent points will be trusted more than past points
                                                            (time closer to the point to make inference at).
                                                            It is to be specified only if

    Returns:
            models (list of gpytorch.models.ExactGP): models containing the parameters, memory,
                                                                                                    constraints of the gps and functions for exact predictions
                                                                                                    :param dof:
    """

    num_models = dof

    train_inputs_indicies = [list(range(pos)) + list(range(dof, pos + dof)) for pos in range(1, dof + 1)]

    # print(f'train_inputs_indicies: {train_inputs_indicies}')
    num_inputs_model = [len(train_inputs_indicies[pos]) for pos in range(len(train_inputs_indicies))]
    # print(f'num_inputs_model: {num_inputs_model}')

    if train_inputs is not None and train_targets is not None:
        # we deduce the length from the training data
        models = [
            ExactGPModelMonoTask(
                train_inputs[:,train_inputs_indicies[idx_model]], train_targets[:, idx_model],
                len(train_inputs[:,train_inputs_indicies[idx_model]][0])
            )
            for idx_model in range(num_models)
        ]
        # for idx_model in range(num_models):
        #     print(train_inputs[:,train_inputs_indicies[idx_model]])
        #     print(train_targets[:, idx_model])
        #     print(len(train_inputs[0]))
    else:
        models = [
            ExactGPModelMonoTask(None, None, num_inputs_model[idx_model]) for idx_model in range(num_models)
        ]

    for idx_model in range(num_models):
        # print(f'idx_model: {idx_model}')
        if constraints_gp is not None:
            if "min_std_noise" in constraints_gp.keys():
                if (
                    type(constraints_gp["min_std_noise"]) != float
                    and type(constraints_gp["min_std_noise"]) != int
                ):
                    min_var_noise = np.power(
                        constraints_gp["min_std_noise"][idx_model], 2
                    )
                else:
                    min_var_noise = np.power(constraints_gp["min_std_noise"], 2)
                if (
                    type(constraints_gp["max_std_noise"]) != float
                    and type(constraints_gp["max_std_noise"]) != int
                ):
                    max_var_noise = np.power(
                        constraints_gp["max_std_noise"][idx_model], 2
                    )
                else:
                    max_var_noise = np.power(constraints_gp["max_std_noise"], 2)
                models[idx_model].likelihood.noise_covar.register_constraint(
                    "raw_noise",
                    gpytorch.constraints.Interval(
                        lower_bound=min_var_noise, upper_bound=max_var_noise
                    ),
                )

            if "min_outputscale" in constraints_gp.keys():
                if (
                    type(constraints_gp["min_outputscale"]) != float
                    and type(constraints_gp["min_outputscale"]) != int
                ):
                    min_outputscale = constraints_gp["min_outputscale"][idx_model]
                else:
                    min_outputscale = constraints_gp["min_outputscale"]

                if (
                    type(constraints_gp["max_outputscale"]) != float
                    and type(constraints_gp["max_outputscale"]) != int
                ):
                    max_outputscale = constraints_gp["max_outputscale"][idx_model]
                else:
                    max_outputscale = constraints_gp["max_outputscale"]
                models[idx_model].covar_module.register_constraint(
                    "raw_outputscale",
                    gpytorch.constraints.Interval(
                        lower_bound=min_outputscale, upper_bound=max_outputscale
                    ),
                )

            if "min_lengthscale" in constraints_gp.keys():
                if (
                    type(constraints_gp["min_lengthscale"]) == float
                    or type(constraints_gp["min_lengthscale"]) == int
                ):
                    min_lengthscale = constraints_gp["min_lengthscale"]
                else:
                    min_lengthscale = constraints_gp["min_lengthscale"][idx_model]
                if (
                    type(constraints_gp["max_lengthscale"]) == float
                    or type(constraints_gp["max_lengthscale"]) == int
                ):
                    max_lengthscale = constraints_gp["max_lengthscale"]
                else:
                    max_lengthscale = constraints_gp["max_lengthscale"][idx_model]

                models[idx_model].covar_module.base_kernel.register_constraint(
                    "raw_lengthscale",
                    gpytorch.constraints.Interval(
                        lower_bound=min_lengthscale, upper_bound=max_lengthscale
                    ),
                )

        # load parameters
        # dict type is used when initializing the models from the json config file
        # list type is used when initializing the models in the parallel training process
        # using the exported parameters

        if type(params) == dict:
            hypers = {
                "base_kernel.lengthscale": params["base_kernel.lengthscale"][idx_model],
                "outputscale": params["outputscale"][idx_model],
            }
            hypers_likelihood = {
                "noise_covar.noise": params["noise_covar.noise"][idx_model]
            }
            models[idx_model].likelihood.initialize(**hypers_likelihood)
            models[idx_model].covar_module.initialize(**hypers)
        elif type(params) == list:
            # print(f'params: {params[idx_model]}, idx_model: {idx_model}')
            models[idx_model].load_state_dict(params[idx_model])
    return models

def init_control(
    ctrl_obj, env, live_plot_obj, random_actions_init, num_repeat_actions=1
):
    """
    Initializes the control environment with random actions and updates visualization and memory.

    Args:
        ctrl_obj (GpMpcController): Control object for computing cost and managing memory.
        env (gym.Env): Gym environment for obtaining observations and applying actions.
        live_plot_obj: Object for real-time 2D graph visualization, with an `update` method.
        rec: Real-time environment visualization object, with a `capture_frame` method.
        params_general (dict): General parameters (render_env, save_render_env, render_live_plots_2d).
        random_actions_init (int): Number of initial random actions.
        costs_tests (np.array): Array to store costs for analysis (shape: (num_runs, num_timesteps)).
        idx_test (int): Current test index.
        num_repeat_actions (int): Number of consecutive constant actions; affects memory storage.
    """
    obs_lst, actions_lst, rewards_lst = [], [], []
    obs, _ = env.reset()
    action, cost, obs_prev_ctrl = None, None, None
    done = False
    idx_action = 0
    while idx_action < random_actions_init:
        if idx_action % num_repeat_actions == 0 or action is None:
            # action = env.action_space.sample()
            action = np.random.uniform(env.action_space.low, env.action_space.high)
            idx_action += 1
            if obs_prev_ctrl is not None and cost is not None:
                ctrl_obj.add_memory(
                    obs=obs_prev_ctrl,
                    action=action,
                    obs_new=obs,
                    reward=cost,
                    # check_storage=False,
                )
        if done:
            obs, _ = env.reset()
            idx_action += 1
        obs_new, reward, done, _, _ = env.step(action)
        obs_prev_ctrl = obs
        obs = obs_new
        cost, _ = ctrl_obj.compute_cost_unnormalized(obs_new, action)
        # cost=reward!
        # cost = cost

        # Update lists for visualization
        obs_lst.append(obs)
        actions_lst.append(action)
        rewards_lst.append(cost)

        # if params_general.get('render_live_plots_2d'):
        live_plot_obj.update(obs=obs, action=action, cost=cost, info_dict=None)

        # Store the last action for potential future use
        ctrl_obj.action_previous_iter = (
            action  # Adjust as needed, e.g., convert to tensor
        )

    return (
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
    )


class BaseControllerObject:
    def __init__(
        self, observation_space, action_space, n_points_init_memory=1000
    ):
        self.action_space = action_space
        self.obs_space = observation_space
        self.num_states = self.obs_space.shape[0]
        self.num_actions = action_space.shape[0]
        self.num_inputs = self.num_states + self.num_actions
        self.points_add_mem_when_full = n_points_init_memory
        # if use_time:
        #     self.num_inputs += 1
        self.x = torch.empty(n_points_init_memory, self.num_inputs)
        self.y = torch.empty(n_points_init_memory, self.num_states)
        self.rewards = torch.empty(n_points_init_memory)

        self.len_mem = 0

    def add_memory(self, observation, action, new_observation, reward, **kwargs):
        raise NotImplementedError()

    def compute_action(self, observation, s_observation):
        raise NotImplementedError()

    def train(self):
        raise NotImplementedError()


# precision double tensor necessary for the gaussian processes predictions
torch.set_default_dtype(torch.double)

class GpMpcController(BaseControllerObject):
    def __init__(self, observation_space, action_space, params_dict):

        params_dict = self.preprocess_params(params_dict)
        self.weight_matrix_cost = \
            torch.block_diag(
                torch.diag(params_dict['controller']['weight_state']),
                torch.diag(params_dict['controller']['weight_action']))

        self.weight_matrix_cost_terminal = torch.diag(params_dict['controller']['weight_state_terminal'])
        self.target_state_norm = params_dict['controller']['target_state_norm']
        self.target_state_action_norm = torch.cat(
            (self.target_state_norm, params_dict['controller']['target_action_norm']))
        # self.limit_action_change = params_dict['controller']['limit_action_change']
        # self.max_change_action_norm = np.array(params_dict['controller']['max_change_action_norm'])
        # self.clip_lower_bound_cost_to_0 = params_dict['controller']['clip_lower_bound_cost_to_0']
        # self.num_repeat_actions = params_dict['controller']['num_repeat_actions']
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

        # self.include_time_gp = params_dict['controller']['include_time_gp']

        BaseControllerObject.__init__(self, observation_space, action_space)

        # self.error_pred_memory = params_dict['memory']['min_error_prediction_state_for_memory']
        # self.std_pred_memory = params_dict['memory']['min_prediction_state_std_for_memory']

        self.gp_constraints = params_dict['gp_constraints']
        self.state_min = None
        self.state_max = None
        self.params_actions_optimizer = params_dict['actions_optimizer']

        # Trick to impose bounds on action changes if it is set to True:
        # - Use change in actions instead of absolute actions in the sklearn optimizer
        # 		to get the optimal actions for the mpc
        # - Impose bounds on the optimized values with the sklearn optimizer
        # - Since action changes are optimized by sklearn, action changes are bounded
        # The definition of bounds are then different in that case and
        # changes in action will be postprocessed to get the absolute action.
        # Initial values of the optimized values are also different
        # if self.limit_action_change:
        #     self.bounds = \
        #         [(-self.max_change_action_norm[idx_action], self.max_change_action_norm[idx_action]) \
        #          for idx_action in range(self.action_space.shape[0])] * self.len_horizon
        #     self.actions_pred_previous_iter = \
        #         np.dot(np.expand_dims(np.random.uniform(low=-1, high=1, size=(self.len_horizon)), 1),
        #                np.expand_dims(self.max_change_action_norm, 0))
        # else:
        self.bounds = [(0, 1)] * self.num_actions * self.len_horizon
        self.actions_pred_previous_iter = np.random.uniform(low=0, high=1,
                                                            size=(self.len_horizon, self.num_actions))

        self.models = create_models(train_inputs=None, train_targets=None, params=params_dict['gp_init'],
                                    constraints_gp=self.gp_constraints, dof=self.DoF)
        for idx_model in range(len(self.models)):
            self.models[idx_model].eval()

        self.num_cores_main = multiprocessing.cpu_count()
        self.ctx = multiprocessing.get_context('spawn')
        # self.ctx = multiprocessing.get_context('fork')
        self.queue_train = self.ctx.Queue()

        self.n_iter_ctrl = 0
        self.n_iter_obs = 0

        self.info_iters = {}
        self.idxs_mem_gp = []

    @staticmethod
    def preprocess_params(params_dict):
        """
		Return a dictionary with the same keys as the input dictionary
		where lists containing values have been transformed into torch.Tensor

		Args:
			params_dict (dict): contains all the parameters for the GP MPC controller,
				obtained from reading the json file. For more information, see parameters.md.

		Returns:
			params_dict (dict): dict parameters after the processing
		"""

        params_dict['controller']['target_state_norm'] = torch.Tensor(params_dict['controller']['target_state_norm'])
        params_dict['controller']['weight_state'] = torch.Tensor(params_dict['controller']['weight_state'])
        params_dict['controller']['weight_state_terminal'] = torch.Tensor(
            params_dict['controller']['weight_state_terminal'])
        params_dict['controller']['target_action_norm'] = torch.Tensor(params_dict['controller']['target_action_norm'])
        params_dict['controller']['weight_action'] = torch.Tensor(params_dict['controller']['weight_action'])

        for key in params_dict['gp_init']:
            if type(params_dict['gp_init'][key]) != float and type(params_dict['gp_init'][key]) != int:
                params_dict['gp_init'][key] = torch.Tensor(params_dict['gp_init'][key])

        for key in params_dict['gp_constraints']:
            if type(params_dict['gp_constraints'][key]) != float and type(params_dict['gp_constraints'][key]) != int:
                params_dict['gp_constraints'][key] = torch.Tensor(params_dict['gp_constraints'][key])
        params_dict['controller']['obs_var_norm'] = torch.Tensor(params_dict['controller']['obs_var_norm'])
        return params_dict

    def to_normed_obs_tensor(self, obs):
        """
		Compute the norm of observation using the min and max of the observation_space of the gym env.

		Args:
			obs  (numpy.array): observation from the gym environment. dim=(Ns,)

		Returns:
			state_mu_norm (torch.Tensor): normed states
		"""
        state_mu_norm = torch.Tensor((obs - self.obs_space.low) / (self.obs_space.high - self.obs_space.low))
        return state_mu_norm

    # def to_normed_var_tensor(self, obs_var):
    #     """
	# 	Compute the norm of the observation variance matrix using
	# 	the min and max of the observation_space of the gym env.
    #
	# 	Args:
	# 		obs_var  (numpy.array): unnormalized variance of the state. dim=(Ns,)
    #
	# 	Returns:
	# 		obs_var_norm (torch.Tensor): normed variance of the state
	# 	"""
    #     obs_var_norm = obs_var / (self.obs_space.high - self.obs_space.low)
    #     obs_var_norm = torch.Tensor(obs_var_norm / (self.obs_space.high - self.obs_space.low).T)
    #     return obs_var_norm

    def to_normed_var_tensor(self, obs_var):
        range_val = self.obs_space.high - self.obs_space.low
        obs_var_norm = obs_var / (range_val ** 2)
        return torch.Tensor(obs_var_norm)

    def to_normed_action_tensor(self, action):
        """
		Compute the norm of the action using the min and max of the action_space of the gym env.

		Args:
			action  (numpy.array): un-normalized action. dim=(Na,)
									Na: dimension of action_space
		Returns:
			action_norm (torch.Tensor): normed action

		"""
        action_norm = torch.Tensor((action - self.action_space.low) / (self.action_space.high - self.action_space.low))
        return action_norm

    def denorm_action(self, action_norm):
        """
		Denormalize the action using the min and max of the action_space of the gym env, so that
		it can be apllied on the gym env

		Args:
			action_norm  (numpy.array or torch.Tensor): normed action. dim=(Na,)
													Na: dimension of action_space
		Returns:
			action (numpy_array or torch.Tensor): un-normalised action. dim=(Na,)
														Na: dimension of action_space
		"""
        action = action_norm * (self.action_space.high - self.action_space.low) + self.action_space.low
        return action

    def barrier_cost(self, state, alpha):
        # Ensure numerical stability by adding a small epsilon
        epsilon = 1e-6
        state = torch.clamp(state, epsilon, 1 - epsilon)
        return -alpha * (torch.log(state) + torch.log(1 - state)).sum(-1)

    def compute_cost(self, state_mu: torch.Tensor, state_var:
    torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:

        """
        Compute the mean and variance of the quadratic cost function for given state and action distributions.

        The cost function is defined as:

            cost = (x - x_target)^T * W * (x - x_target)

        where:
        - x is the concatenation of state and action variables.
        - x_target is the target state-action vector.
        - W is the weight matrix (self.weight_matrix_cost).

        Given that the state and action are random variables characterized by their means and variances,
        this function computes the mean and variance of the cost.

        Parameters:
        - state_mu (torch.Tensor): Mean of the state(s). Shape (Ns,) or (Np, Ns)
        - state_var (torch.Tensor): Covariance matrix of the state(s). Shape (Ns, Ns) or (Np, Ns, Ns)
        - action (torch.Tensor): Action(s). Shape (Na,) or (Np, Na)

        Returns:
        - cost_mu (torch.Tensor): Mean of the cost distribution. Shape (1,) or (Np,)
        - cost_var (torch.Tensor): Variance of the cost distribution. Shape (1,) or (Np,)
        """

        # Determine if we're working with a single time-step or a trajectory
        single_timestep = state_var.ndim == 2  # state_var shape (Ns, Ns)

        if single_timestep:
            # Reshape inputs to add batch dimension for uniform processing
            state_mu = state_mu.unsqueeze(0)  # Shape: (1, Ns)
            state_var = state_var.unsqueeze(0)  # Shape: (1, Ns, Ns)
            action = action.unsqueeze(0)  # Shape: (1, Na)

        Np = state_mu.shape[0]  # Number of prediction steps (batch size)
        Ns = state_mu.shape[1]  # State dimension
        Na = action.shape[1]  # Action dimension

        # Concatenate state and action means to form x_mu
        x_mu = torch.cat((state_mu, action), dim=1)  # Shape: (Np, Ns + Na)

        # Compute error between current state-action and target
        x_target = self.target_state_action_norm  # Expected shape: (Ns + Na,)
        error = x_mu - x_target  # Shape: (Np, Ns + Na)

        # Construct the covariance matrix for the state-action
        # Since action variance is zero (actions are deterministic), we can construct a block diagonal matrix
        # with state_var and zeros for the action variance
        state_action_var = torch.zeros((Np, Ns + Na, Ns + Na), device=state_var.device, dtype=state_var.dtype)
        state_action_var[:, :Ns, :Ns] = state_var  # Shape: (Np, Ns + Na, Ns + Na)

        # Weight matrix W for the cost function
        W = self.weight_matrix_cost  # Shape: (Ns + Na, Ns + Na)

        # Compute the expected cost (mean of the cost)
        # E[cost] = trace(W * state_action_var) + error^T * W * error

        # First term: trace(W * state_action_var)
        # Using torch.einsum for efficient computation
        trace_term = torch.einsum('bii->b', W @ state_action_var)  # Shape: (Np,)

        # Second term: error^T * W * error
        error = error.unsqueeze(-1)  # Shape: (Np, Ns + Na, 1)
        quadratic_term = (error.transpose(-2, -1) @ W @ error).squeeze(-1).squeeze(-1)  # Shape: (Np,)

        # Total expected cost
        cost_mu = trace_term + quadratic_term  # Shape: (Np,)

        # Compute barrier cost
        alpha = self.barrier_weight  # You can define this parameter
        barrier = self.barrier_cost(state_mu, alpha)
        # Total cost mean
        cost_mu = cost_mu + barrier

        # Compute the variance of the cost
        # Var[cost] = 2 * sum over i,j of (TS_ij)^2 + 4 * error^T * W * state_action_var * W * error

        # Compute TS = W @ state_action_var
        TS = W @ state_action_var  # Shape: (Np, Ns + Na, Ns + Na)

        # First variance term: 2 * sum of squares of TS elements
        trace_var_term = 2 * (TS ** 2).sum(dim=(-2, -1))  # Shape: (Np,)

        # Second variance term: 4 * (error^T @ W @ state_action_var @ W @ error)
        var_quadratic_term = 4 * (error.transpose(-2, -1) @ W @ state_action_var @ W @ error).squeeze(-1).squeeze(
            -1)  # Shape: (Np,)

        # Total variance of the cost
        cost_var = trace_var_term + var_quadratic_term  # Shape: (Np,)

        if single_timestep:
            # Remove batch dimension for single time-step inputs
            cost_mu = cost_mu.squeeze(0)  # Shape: ()
            cost_var = cost_var.squeeze(0)  # Shape: ()

        return cost_mu, cost_var

    def compute_cost_terminal(self, state_mu: torch.Tensor,
                              state_var: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the mean and variance of the terminal cost for given state distributions.

        The terminal cost is defined as:

            cost = (x - x_target)^T * W * (x - x_target)

        where:
        - x is the state variable.
        - x_target is the target state vector.
        - W is the weight matrix (self.weight_matrix_cost_terminal).

        Given that the state is a random variable characterized by its mean and covariance,
        this function computes the mean and variance of the cost.

        Parameters:
        - state_mu (torch.Tensor): Mean of the terminal state(s). Shape (Ns,) or (Np, Ns)
        - state_var (torch.Tensor): Covariance matrix of the terminal state(s). Shape (Ns, Ns) or (Np, Ns, Ns)

        Returns:
        - cost_mu (torch.Tensor): Mean of the cost distribution. Shape (1,) or (Np,)
        - cost_var (torch.Tensor): Variance of the cost distribution. Shape (1,) or (Np,)
        """

        # Ensure inputs are tensors
        state_mu = torch.as_tensor(state_mu)
        state_var = torch.as_tensor(state_var)

        # Determine if we're working with a single state or batch of states
        single_timestep = state_mu.ndim == 1  # state_mu shape (Ns,)

        if single_timestep:
            # Reshape inputs to add batch dimension for uniform processing
            state_mu = state_mu.unsqueeze(0)  # Shape: (1, Ns)
            state_var = state_var.unsqueeze(0)  # Shape: (1, Ns, Ns)

        Np = state_mu.shape[0]  # Number of samples (batch size)
        Ns = state_mu.shape[1]  # State dimension

        # Target state vector
        x_target = self.target_state_norm  # Shape: (Ns,)
        x_target = x_target.unsqueeze(0) if x_target.ndim == 1 else x_target  # Ensure shape is (1, Ns)

        # Compute error vector
        error = state_mu - x_target  # Shape: (Np, Ns)

        # Weight matrix W for the cost function
        W = self.weight_matrix_cost_terminal  # Shape: (Ns, Ns)

        # Compute expected cost (mean of the cost)
        # E[cost] = trace(W * state_var) + error^T * W * error

        # First term: trace(W @ state_var)
        # W @ state_var: shape (Np, Ns, Ns)
        TS = W @ state_var  # Shape: (Np, Ns, Ns)
        trace_term = torch.einsum('bii->b', TS)  # Shape: (Np,)

        # Second term: error^T * W * error
        error = error.unsqueeze(-1)  # Shape: (Np, Ns, 1)
        quadratic_term = (error.transpose(-2, -1) @ W @ error).squeeze(-1).squeeze(-1)  # Shape: (Np,)

        # Total expected cost
        cost_mu = trace_term + quadratic_term  # Shape: (Np,)
        
        alpha = self.barrier_weight  # You can define this parameter
        barrier = self.barrier_cost(state_mu, alpha)
        # Total cost mean
        cost_mu = cost_mu + barrier

        # Compute the variance of the cost
        # Var[cost] = 2 * tr((W * state_var)^2) + 4 * error^T * W * state_var * W * error

        # First variance term: 2 * trace((TS)^2)
        var_term1 = 2 * torch.einsum('bij,bji->b', TS, TS)  # Shape: (Np,)

        # Second variance term: 4 * error^T * W * state_var * W * error
        W_error = W @ error  # Shape: (Np, Ns, 1)
        temp = W @ state_var @ W_error  # Shape: (Np, Ns, 1)
        var_term2 = 4 * (error.transpose(-2, -1) @ temp).squeeze(-1).squeeze(-1)  # Shape: (Np,)

        # Total variance of the cost
        cost_var = var_term1 + var_term2  # Shape: (Np,)

        if single_timestep:
            # Remove batch dimension for single time-step inputs
            cost_mu = cost_mu.squeeze(0)  # Shape: ()
            cost_var = cost_var.squeeze(0)  # Shape: ()

        return cost_mu, cost_var

    def compute_cost_unnormalized(self, obs, action, obs_var=None):
        """
		Compute the cost on un-normalized state and actions.
		Takes in numpy array and returns numpy array.
		Meant to be used to compute the cost outside the object.
		Args:
			obs (numpy.array): state (or observation). shape=(Ns,)
			action (numpy.array): action. Shape=(Na,)
			obs_var (numpy.array): state (or observation) variance. Default=None. shape=(Ns, Ns)
									If set to None, the observation constant stored inside the object will be used

		Returns:
			cost_mu (float): Mean of the cost
			cost_var (float): variance of the cost
		"""
        obs_norm = self.to_normed_obs_tensor(obs)
        action_norm = self.to_normed_action_tensor(action)
        if obs_var is None:
            obs_var_norm = self.obs_var_norm
        else:
            obs_var_norm = self.to_normed_var_tensor(obs_var)
        cost_mu, cost_var = self.compute_cost(obs_norm, obs_var_norm, action_norm)
        return cost_mu.item(), cost_var.item()

    @staticmethod
    def calculate_factorizations(x, y, models, dof):
        """
		Compute iK and beta using the points in memory, which are needed to make predictions with the gaussian processes.
		These two variables only depends on data in memory, and not on input distribution,
		so they separated from other computation such that they can be computed outside the optimisation function,
		which is computed multiple times at each iteration

		Function inspired from
		https://github.com/nrontsis/PILCO/blob/6a962c8e4172f9e7f29ed6e373c4be2dd4b69cb7/pilco/models/mgpr.py#L81,
		reimplemented from tensorflow to pytorch
		Args:
			x (torch.Tensor): matrix containing the states and actions. Dim=(Nm, Ns + Na)
			y (torch.Tensor): matrix containing the states change. Dim=(Nm, Ns)
			models (list of gpytorch.models.ExactGP): list containing the gp models used to predict each state change.
														Len=Ns
			Ns: number of states
			Na: number of actions
			Nm: number of points in memory

		Returns:
			iK (torch.Tensor): needed by the gaussian processes models to compute the predictions
			beta (torch.Tensor): needed by the gaussian processes models to compute the predictions

		"""
        train_inputs_indices = [list(range(i)) + list(range(dof, i + dof)) for i in range(1, dof + 1)]
        # K = torch.stack([model.covar_module(x).evaluate() for model in models])

        K = torch.stack([
            model.covar_module(x[:, idxs]).evaluate()
            for model, idxs in zip(models, train_inputs_indices)
        ])

        batched_eye = torch.eye(K.shape[1]).repeat(K.shape[0], 1, 1)
        L = torch.linalg.cholesky(K + torch.stack([model.likelihood.noise for model in models])[:, None] * batched_eye)
        iK = torch.cholesky_solve(batched_eye, L)
        Y_ = (y.t())[:, :, None]
        beta = torch.cholesky_solve(Y_, L)[:, :, 0]
        return iK, beta

    def predict_next_state_change(self, state_mu, state_var, iK, beta):
        """
        Predicts the next state change for each output (each GP model) given an input
        distribution (mean and covariance). For each model i, the following steps are performed:

        1. For the selected indices 𝕀ᵢ = {0, 1, …, i-1} ∪ {DoF, DoF+1, …, DoF+i-1}:
           - Compute the centered training inputs:
             δxₙ = xₙ^(i) - μ_new^(i), where μ_new^(i) = state_mu[𝕀ᵢ].
           - Whiten these differences using the lengthscales:
             x̃ₙ = L⁻¹ δxₙ, where L = diag(ℓ₁, …, ℓ_d₍ᵢ₎).

        2. Incorporate the input uncertainty by forming:
             B = L⁻¹ S_x L⁻¹ + I,
           where S_x = state_var[𝕀ᵢ, 𝕀ᵢ].

        3. Solve for tₙ:
             tₙ = B⁻¹ x̃ₙ.

        4. Compute the similarity weight for each training point:
             lbₙ = exp(–½ · x̃ₙᵀ B⁻¹ x̃ₙ) · βᵢ.

        5. Compute the scaling factor:
             c = σ² / √(det(B)),
           where σ² is the kernel output scale.

        6. The predicted mean change is:
             Mᵢ = c · Σₙ lbₙ.

        7. An auxiliary derivative-like term is computed as:
             Vᵢ = c · Σₙ (L⁻¹ tₙ) lbₙ.

        8. The predictive variance is computed using:
             k_* = c · lb, and then
             Sᵢ = σ² – k_*ᵀ (iK[i]) k_*.

        Returns:
            M: A tensor with predicted mean changes for each output (shape: (Ns,)).
            S: A diagonal covariance matrix for the predictions (shape: (Ns, Ns)).
            V: A tensor of derivative-like terms (shape: (Ns, num_inputs)), padded with zeros.
        """
        M_list = []
        V_list = []
        S_diag_list = []

        # Build index lists that capture the causal structure:
        # For i = 1, …, DoF, use indices [0, …, i-1] for states and [DoF, …, DoF+i-1] for actions.
        train_inputs_indices = [list(range(i)) + list(range(self.DoF, i + self.DoF))
                                for i in range(1, self.DoF + 1)]

        for i, model in enumerate(self.models):
            idxs = train_inputs_indices[i]
            # x_i: training inputs for model i, shape: (N, d_i)
            x_i = self.x[self.idxs_mem_gp][:, idxs]
            # inp: centered training inputs, where state_mu[𝕀ᵢ] is the new input mean.
            inp = x_i - state_mu[idxs]

            # Retrieve kernel hyperparameters:
            lengthscale = model.covar_module.base_kernel.lengthscale[0]  # vector of shape (d_i,)
            variance = model.covar_module.outputscale  # scalar σ²

            # Compute the whitening matrix L⁻¹:
            iL = torch.diag(1 / lengthscale)
            # Whiten the differences:
            iN = inp @ iL  # Each row is x̃ₙ = L⁻¹ (xₙ - μ_new).

            # S_x: covariance of the selected input components.
            Sx = state_var[np.ix_(idxs, idxs)]
            # Form B = L⁻¹ S_x L⁻¹ + I.
            epsilon = 1e-6  # or choose an appropriate small value
            B = iL @ Sx @ iL + torch.eye(len(idxs), dtype=Sx.dtype, device=Sx.device) + epsilon * torch.eye(len(idxs),
                                                                                                            dtype=Sx.dtype,
                                                                                                            device=Sx.device)
            # B = iL @ Sx @ iL + torch.eye(len(idxs), dtype=Sx.dtype, device=Sx.device)

            # Solve for tₙ: for each training point, compute t = B⁻¹ x̃.
            t = torch.linalg.solve(B, iN.T).T

            # Compute similarity weights:
            # lbₙ = exp(–½ x̃ₙᵀ B⁻¹ x̃ₙ) * beta[i]
            lb = torch.exp(-0.5 * torch.sum(iN * t, dim=1)) * beta[i]

            # Compute scaling factor: c = variance / sqrt(det(B))
            det_B = torch.clamp(torch.det(B), min=1e-6)
            c = variance / torch.sqrt(det_B)

            # Predicted mean change: Mᵢ = c * sumₙ lbₙ.
            M_i = c * torch.sum(lb)

            # Compute the derivative-like term:
            # First compute: t @ iL = L⁻¹ t, then Vᵢ = c * sumₙ (L⁻¹ tₙ) lbₙ.
            tiL = t @ iL
            V_i = (tiL.T @ lb[:, None])[:, 0] * c

            # Compute predictive variance:
            # Define k_* = c * lb (vector of kernel evaluations).
            k_star = c * lb
            # The quadratic term: k_*ᵀ iK[i] k_*.
            quad_term = k_star @ (iK[i] @ k_star)
            S_i_diag = variance - quad_term
            S_i_diag = torch.clamp(S_i_diag, min=0.0)

            M_list.append(M_i)
            # Pad the derivative term to full input dimension.
            V_pad = torch.zeros(self.num_inputs, dtype=V_i.dtype, device=V_i.device)
            V_pad[idxs] = V_i
            V_list.append(V_pad)
            S_diag_list.append(S_i_diag)

        # Stack the predicted means into a vector M (length = number of outputs).
        M = torch.stack(M_list)
        # Stack the derivative vectors into a matrix V.
        V = torch.stack(V_list)
        # Form a diagonal covariance matrix S from the list of variances.
        S = torch.diag(torch.tensor(S_diag_list, dtype=M.dtype, device=M.device))

        return M, S, V

    def predict_trajectory(self, actions, obs_mu, obs_var, iK, beta):
        """
        Compute the future predicted states distribution for the simulated trajectory without in-place modifications.
        """
        state_dim = obs_mu.shape[0]
        # Use lists to accumulate predictions for the state mean and covariance.
        mu_list = [obs_mu]
        var_list = [obs_var]

        for idx_time in range(1, self.len_horizon + 1):
            # Build the full input: concatenate current state and planned action.
            input_var = torch.zeros((self.num_inputs, self.num_inputs),
                                    dtype=obs_var.dtype, device=obs_var.device)
            # Only the state part of the covariance is relevant.
            input_var[:state_dim, :state_dim] = var_list[-1]

            input_mean = torch.empty((self.num_inputs,),
                                     dtype=obs_mu.dtype, device=obs_mu.device)
            input_mean[:self.num_states] = mu_list[-1]
            input_mean[self.num_states:(self.num_states + self.num_actions)] = actions[idx_time - 1]

            # Get prediction for the state change, its variance, and the full Jacobian-like term.
            state_change, state_change_var, v = self.predict_next_state_change(input_mean, input_var, iK, beta)

            # Update the state mean.
            new_mu = mu_list[-1] + state_change

            # Extract the Jacobian with respect to the state only (first num_states columns).
            v_state = v[:, :self.num_states]
            # Propagate covariance using the linearized model.
            new_var = state_change_var + v_state @ var_list[-1] @ v_state.T

            mu_list.append(new_mu)
            var_list.append(new_var)

        # Stack the lists to obtain tensors of shape (horizon+1, ...)
        states_mu_pred = torch.stack(mu_list)  # shape: (Nh+1, num_states)
        states_var_pred = torch.stack(var_list)  # shape: (Nh+1, num_states, num_states)

        # Compute the costs along the trajectory.
        costs_traj, costs_traj_var = self.compute_cost(states_mu_pred[:-1], states_var_pred[:-1], actions)
        cost_traj_final, costs_traj_var_final = self.compute_cost_terminal(states_mu_pred[-1], states_var_pred[-1])
        costs_traj = torch.cat((costs_traj, cost_traj_final[None]), 0)
        costs_traj_var = torch.cat((costs_traj_var, costs_traj_var_final[None]), 0)
        costs_traj_lcb = costs_traj - self.exploration_factor * torch.sqrt(costs_traj_var)

        return states_mu_pred, states_var_pred, costs_traj, costs_traj_var, costs_traj_lcb

    def compute_mean_lcb_trajectory(self, actions, obs_mu, obs_var, iK, beta):
        """
		Compute the mean lower bound cost of a trajectory given the actions of the trajectory
		and initial state distribution. The gaussian process models are used to predict the evolution of
		states (mean and variance). Then the cost is computed for each predicted state and the mean is returned.
		The partial derivatives of the mean lower bound cost with respect to the actions are also returned.
		They are computed automatically with autograd from pytorch.
		This function is called multiple times by an optimizer to find the optimal actions.

		Args:
			actions (numpy.array): actions to apply for the simulated trajectory.
									It is a flat 1d array, whatever the dimension of actions
									so that this function can be used by the minimize function of the scipy library.
									It is reshaped and transformed into a tensor inside.
									If self.limit_action_change is true, each element of the array contains the relative
									change with respect to the previous iteration, so that the change can be bounded by
									the optimizer. dim=(Nh x Na,)
									where Nh is the len of the horizon and Na the dimension of actions

			obs_mu (torch.Tensor):	mean value of the inital state distribution.
									dim=(Ns,) where Ns is the dimension of state

			obs_var (torch.Tensor): covariance matrix of the inital state distribution.
									dim=(Ns, Ns) where Ns is the dimension of state

			iK (torch.Tensor): intermediary result for the gp predictions that only depends on the points in memory
								and not on the points to predict.
								It is computed outside the optimization function in self.calculate_factorizations
								for more efficient predictions. Dim=(Ns, Np, Np)
								where Ns is the dimension of state and Np the number of points in gp memory

			beta (torch.Tensor): intermediary result for the gp predictions that only depends on the points in memory
								and not on the points to predict.
								It is computed outside the optimization function in self.calculate_factorizations
								for more efficient predictions. Dim=(Ns, Np)

		Returns:
			mean_cost_traj_lcb.item() (float): lower bound of the mean cost distribution
														of the predicted trajectory.


			gradients_dcost_dactions[:, 0].detach().numpy() (numpy.array):
																Derivative of the lower bound of the mean cost
																distribution with respect to each of the actions in the
																prediction horizon. Dim=(Nh,)
																where Nh is the len of the horizon
		"""
        # reshape actions from flat 1d numpy array into 2d tensor
        actions = np.atleast_2d(actions.reshape(self.len_horizon, -1))
        actions = torch.Tensor(actions)
        actions.requires_grad = True
        # If limit_action_change is true, actions are transformed back into absolute values from relative change
        # if self.limit_action_change:
        #     actions_input = actions.clone()
        #     actions_input[0] = self.action_previous_iter + actions_input[0]
        #     actions_input = torch.clamp(torch.cumsum(actions_input, dim=0), 0, 1)
        # else:
        actions_input = actions
        mu_states_pred, s_states_pred, costs_traj, costs_traj_var, costs_traj_lcb = \
            self.predict_trajectory(actions_input, obs_mu, obs_var, iK, beta)
        # if self.clip_lower_bound_cost_to_0:
        #     costs_traj_lcb = torch.clamp(costs_traj_lcb, 0, np.inf)
        mean_cost_traj_lcb = costs_traj_lcb.mean()
        gradients_dcost_dactions = torch.autograd.grad(mean_cost_traj_lcb, actions, retain_graph=False)[0]

        self.cost_traj_mean_lcb = mean_cost_traj_lcb.detach()
        self.mu_states_pred = mu_states_pred.detach()
        self.costs_trajectory = costs_traj.detach()
        self.states_var_pred = s_states_pred.detach()
        self.costs_traj_var = costs_traj_var.detach()

        return mean_cost_traj_lcb.item(), gradients_dcost_dactions.flatten().detach().numpy()

    def compute_action(self, obs_mu, obs_var=None):
        """
		Get the optimal action given the observation by optimizing
		the actions of the simulated trajectory with the gaussian process models such that the lower confidence bound of
		the mean cost of the trajectory is minimized.
		Only the first action of the prediction window is returned.

		Args:
			obs_mu (numpy.array): unnormalized observation from the gym environment. dim=(Ns)
			obs_var (numpy.array): unnormalized variance of the observation from the gym environment. dim=(Ns, Ns).
									default=None. If it is set to None,
									the observation noise from the json parameters will be used for every iteration.
									Ns is the dimension of states in the gym environment.

		Returns:
			action_denorm (numpy.array): action to use in the gym environment.
										It is denormalized, so it can be used directly.
										dim=(Na), where Ns is the dimension of the action_space
			info_dict (dict): contains all additional information about the iteration.
							Keys:
							- iteration (int): index number of the iteration
							- state (torch.Tensor): current normed state (before applying the action)
							- predicted states (torch.Tensor): mean value of the predicted distribution of the
																normed states in the mpc
							- predicted states std (torch.Tensor): predicted normed standard deviation of the
																	distribution of the states in the mpc
							- predicted actions (torch.Tensor): predicted optimal normed actions that minimize
																the long term cost in the mpc
							cost (float): mean value of the current cost distribution
							cost std (float): standard deviation of the current cost distribution
							predicted costs (torch.Tensor): mean value of the predicted cost distribution in the mpc
							predicted costs std (torch.Tensor): standard deviation of the
																predicted cost distribution in the mpc
							mean predicted cost (float): mean value of the predicted cost distribution in the mpc,
																averaged over future predicted time steps
							mean predicted cost std (float): standard deviation of the predicted cost distribution in the mpc,
																averaged over future predicted time steps
							lower bound mean predicted cost (float): lower bound of the predicted cost distribution
												(cost_mean_future_mean - self.exploration_factor * cost_std_future_mean).
												It is the value minimized by the mpc.

		"""
        # Check for parallel process that are open but not alive at each iteration to retrieve the results and close them
        self.check_and_close_processes()
        torch.set_num_threads(self.num_cores_main)

        with torch.no_grad():
            obs_mu_normed = self.to_normed_obs_tensor(obs=obs_mu)
            if obs_var is None:
                obs_var_norm = self.obs_var_norm
            else:
                obs_var_norm = self.to_normed_var_tensor(obs_var=obs_var)
            # iK and beta are computed outside of the optimization function since it depends only on the points in memory,
            # and not on the input. Otherwise, the optimization time at each iteration would be too high

            self.iK, self.beta = self.calculate_factorizations(self.x[self.idxs_mem_gp],
                                                               self.y[self.idxs_mem_gp], self.models, self.DoF)
            # The initial actions_norm values are fixed using the actions_norm predictions of the mpc of the previous iteration,
            # offset by 1, so that the initial values have a correct guess, which allows to get good results
            # by using only 1 to 3 iteration of the action optimizer at each iteration.
            # The value of the last init value of action in the prediction window is set as the same as
            # the last of the prevous iteration.
            init_actions_optim = (np.concatenate((self.actions_pred_previous_iter[1:],
                                                  np.expand_dims(self.actions_pred_previous_iter[-1], 0)), axis=0))
            # See comment in __init__ above the definition of bounds for more information about limit_action change trick
            # Actions in the minimize function fo scipy must be a 1d vector.
            # If the action is multidimensional, it is resized to a 1d array and passed into the minimize function as
            # a 1d array. The init values and bounds must match the dimension of the passed array.
            # It is reshaped inside the minimize function to get back the true dimensions
            # if self.limit_action_change:
            #     init_actions_optim_absolute = np.empty_like(init_actions_optim)
            #     init_actions_optim_absolute[0] = self.action_previous_iter
            #     init_actions_optim_absolute += init_actions_optim
            #     init_actions_optim_absolute = np.cumsum(init_actions_optim_absolute, axis=0)
            #     if np.logical_or(np.any(init_actions_optim_absolute > 1),
            #                      np.any(init_actions_optim_absolute < 0)):
            #         for idx_time in range(1, len(init_actions_optim)):
            #             init_actions_optim_absolute[idx_time] = init_actions_optim_absolute[idx_time - 1] + \
            #                                                     init_actions_optim[idx_time]
            #             indexes_above_1 = np.nonzero(init_actions_optim_absolute[idx_time] > 1)[0]
            #             indexes_under_0 = np.nonzero(init_actions_optim_absolute[idx_time] < 0)[0]
            #             init_actions_optim[idx_time][indexes_above_1] = \
            #                 1 - init_actions_optim_absolute[idx_time - 1][indexes_above_1]
            #             init_actions_optim[idx_time][indexes_under_0] = \
            #                 - init_actions_optim_absolute[idx_time - 1][indexes_under_0]
            #             init_actions_optim_absolute[idx_time][indexes_above_1] = 1
            #             init_actions_optim_absolute[idx_time][indexes_under_0] = 0

            init_actions_optim = init_actions_optim.flatten()
        # The optimize function from the scipy library.
        # It is used to get the optimal actions_norm in the prediction window
        # that minimizes the lower bound of the predicted cost. The jacobian is used,
        # otherwise the computation times would be 5 to 10x slower (for the tests I used)
        time_start_optim = time.time()
        res = minimize(fun=self.compute_mean_lcb_trajectory,
                       x0=init_actions_optim,
                       jac=True,
                       args=(obs_mu_normed, obs_var_norm, self.iK, self.beta),
                       method='L-BFGS-B',
                       bounds=self.bounds,
                       options=self.params_actions_optimizer)

        time_end_optim = time.time()
        print("Optimisation time for iteration: %.3f s" % (time_end_optim - time_start_optim))

        actions_norm = res.x.reshape(self.len_horizon, -1)
        # prepare init values for the next iteration
        self.actions_pred_previous_iter = actions_norm.copy()

        with torch.no_grad():
            # if self.limit_action_change:
            #     actions_norm[0] += np.array(self.action_previous_iter)
            #     actions_norm = np.clip(np.cumsum(actions_norm, axis=0), 0, 1)
            #     action_next = actions_norm[0]
            #     self.action_previous_iter = torch.Tensor(action_next)
            # else:
            action_next = actions_norm[0]

            actions_norm = torch.Tensor(actions_norm)
            action = self.denorm_action(action_next)

            cost, cost_var = self.compute_cost(obs_mu_normed, obs_var_norm, actions_norm[0])
            # states_denorm = self.states_mu_pred[1:] * \
            # (self.observation_space.high - self.observation_space.low) + self.observation_space.low
            # states_std_denorm = states_std_pred * (self.observation_space.high - self.observation_space.low)
            states_std_pred = torch.diagonal(self.states_var_pred, dim1=-2, dim2=-1).sqrt()
            info_dict = {'iteration': self.n_iter_ctrl,
                         'state': self.mu_states_pred[0],
                         'predicted states': self.mu_states_pred[1:],
                         'predicted states std': states_std_pred[1:],
                         'predicted actions': actions_norm,
                         'cost': cost.item(), 'cost std': cost_var.sqrt().item(),
                         'predicted costs': self.costs_trajectory[1:],
                         'predicted costs std': self.costs_traj_var[1:].sqrt(),
                         # 'mean predicted cost': np.min([self.costs_trajectory[1:].mean().item(), 3]),
                         'mean predicted cost': self.costs_trajectory[1:].mean().item(),
                         'mean predicted cost std': self.costs_traj_var[1:].sqrt().mean().item(),
                         'lower bound mean predicted cost': self.cost_traj_mean_lcb.item()}
            for key in info_dict.keys():
                if not key in self.info_iters:
                    self.info_iters[key] = [info_dict[key]]
                else:
                    self.info_iters[key].append(info_dict[key])
            self.n_iter_ctrl += 1#self.num_repeat_actions
            return action, info_dict

    def add_memory(self, obs, action, obs_new, reward,
                   # check_storage=True,
                   predicted_state=None, predicted_state_std=None):
        """
		Add an observation, action and observation after applying the action to the memory that is used
		by the gaussian process models.
		At regular number of points interval (self.training_frequency),
		the training process of the gaussian process models will be launched to optimize the hyper-parameters.

		Args:
			obs (numpy.array): non-normalized observation. Dim=(Ns,)
			action (numpy.array): non-normalized action. Dim=(Ns,)
			obs_new (numpy.array): non-normalized observation obtained after applying the action on the observation.
									Dim=(Ns,)
			reward (float): reward obtained from the gym env. Unused at the moment.
							The cost given state and action is computed instead.
			check_storage (bool): If check_storage is true,
									predicted_state and predicted_state_std will be checked (if not None) to
									know weither to store the point in memory or not.

			predicted_state (numpy.array or torch.Tensor or None):
								if check_storage is True and predicted_state is not None,
								the prediction error for that point will be computed.
								and the point will only be stored in memory if the
								prediction error is larger than self.error_pred_memory. Dim=(Ns,)

			predicted_state_std (numpy.array or torch.Tensor or None):
								If check_storage is true, and predicted_state_std is not None, the point will only be
								stored in memory if it is larger than self.error_pred_memory. Dim=(Ns,)

			where Ns: dimension of states, Na: dimension of actions
		"""
        if obs is None:
            return
        obs_norm = self.to_normed_obs_tensor(obs)
        action_norm = self.to_normed_action_tensor(action=action)
        obs_new_norm = self.to_normed_obs_tensor(obs_new)

        if len(self.x) < (self.len_mem + 1):
            # self.x = torch.cat(self.x, torch.empty(self.points_add_mem_when_full, self.x.shape[1]))
            self.x = torch.cat([self.x, torch.empty(self.points_add_mem_when_full, self.x.shape[1])])
            self.y = torch.cat(self.y, torch.empty(self.points_add_mem_when_full, self.y.shape[1]))
            self.rewards = torch.cat(self.rewards, torch.empty(self.points_add_mem_when_full))

        self.x[self.len_mem, :(obs_norm.shape[0] + action_norm.shape[0])] = \
            torch.cat((obs_norm, action_norm))[None]

        self.y[self.len_mem] = obs_new_norm - obs_norm
        self.rewards[self.len_mem] = reward

        # if self.include_time_gp:
        #     self.x[self.len_mem, -1] = self.n_iter_obs

        store_gp_mem = True
        # if check_storage:
        #     if predicted_state is not None:
        #         error_prediction = torch.abs(predicted_state - obs_new_norm)
        #         store_gp_mem = torch.any(error_prediction > self.error_pred_memory)
        #
        #     if predicted_state_std is not None and store_gp_mem:
        #         store_gp_mem = torch.any(predicted_state_std > self.std_pred_memory)

        if store_gp_mem:
            self.idxs_mem_gp.append(self.len_mem)

        self.len_mem += 1
        self.n_iter_obs += 1

        if self.len_mem % self.training_frequency == 0 and \
                not ('p_train' in self.__dict__ and not self.p_train._closed):
            # self.train(self.queue_train, self.x[self.idxs_mem_gp],
            #            self.y[self.idxs_mem_gp],
            #            [model.state_dict() for model in self.models],
            #            self.gp_constraints, self.lr_train,
            #            self.iter_train, self.clip_grad_value,
            #            self.print_train, self.step_print_train)

            self.p_train = self.ctx.Process(target=self.train, args=(self.queue_train,
                                                                     self.x[self.idxs_mem_gp],
                                                                     self.y[self.idxs_mem_gp],
                                                                     [model.state_dict() for model in self.models],
                                                                     self.gp_constraints, self.lr_train,
                                                                     self.iter_train, self.clip_grad_value,
                                                                     self.print_train, self.step_print_train,
                                                                     self.DoF))
            self.p_train.start()
            self.num_cores_main -= 1

    # @staticmethod
    # def train(queue, train_inputs, train_targets, parameters, constraints_hyperparams, lr_train, num_iter_train,
    #           clip_grad_value, print_train=False, step_print_train=25, dof=False):
    #     """
	# 	Train the gaussian process models hyper-parameters such that the marginal-log likelihood
	# 	for the predictions of the points in memory is minimized.
	# 	This function is launched in parallel of the main process, which is why a queue is used to transfer
	# 	information back to the main process and why the gaussian process models are reconstructed
	# 	using the points in memory and hyper-parameters (they cant be sent directly as argument).
	# 	If an error occurs, returns the parameters sent as init values
	# 	(hyper-parameters obtained by the previous training process)
	# 	Args:
	# 		queue (multiprocessing.queues.Queue): queue object used to transfer information to the main process
	# 		train_inputs (torch.Tensor): input data-points of the gaussian process models (concat(obs, actions)). Dim=(Np, Ns + Na)
	# 		train_targets (torch.Tensor): targets data-points of the gaussian process models (obs_new - obs). Dim=(Np, Ns)
	# 		parameters (list of OrderedDict): contains the hyper-parameters of the models used as init values.
	# 											They are obtained by using [model.state_dict() for model in models]
	# 											where models is a list containing gaussian process models of the gpytorch library:
	# 											gpytorch.models.ExactGP
	# 		constraints_hyperparams (dict): Constraints on the hyper-parameters. See parameters.md for more information
	# 		lr_train (float): learning rate of the training
	# 		num_iter_train (int): number of iteration for the training optimizer
	# 		clip_grad_value (float): value at which the gradient are clipped, so that the training is more stable
	# 		print_train (bool): weither to print the information during training. default=False
	# 		step_print_train (int): If print_train is True, only print the information every step_print_train iteration
	# 	"""
    #
    #     torch.set_num_threads(1)
    #     start_time = time.time()
    #     # create models, which is necessary since this function is used in a parallel process
    #     # that do not share memory with the principal process
    #     models = create_models(train_inputs, train_targets, parameters, constraints_hyperparams, dof=dof)
    #     best_outputscales = [model.covar_module.outputscale.detach() for model in models]
    #     best_noises = [model.likelihood.noise.detach() for model in models]
    #     best_lengthscales = [model.covar_module.base_kernel.lengthscale.detach() for model in models]
    #     previous_losses = torch.empty(len(models))
    #
    #     for model_idx in range(len(models)):
    #         output = models[model_idx](models[model_idx].train_inputs[0])
    #         mll = gpytorch.mlls.ExactMarginalLogLikelihood(models[model_idx].likelihood, models[model_idx])
    #         previous_losses[model_idx] = -mll(output, models[model_idx].train_targets)
    #
    #     best_losses = previous_losses.detach().clone()
    #     # Random initialization of the parameters showed better performance than
    #     # just taking the value from the previous iteration as init values.
    #     # If parameters found at the end do not better performance than previous iter,
    #     # return previous parameters
    #     for model_idx in range(len(models)):
    #         models[model_idx].covar_module.outputscale = \
    #             models[model_idx].covar_module.raw_outputscale_constraint.lower_bound + \
    #             torch.rand(models[model_idx].covar_module.outputscale.shape) * \
    #             (models[model_idx].covar_module.raw_outputscale_constraint.upper_bound - \
    #              models[model_idx].covar_module.raw_outputscale_constraint.lower_bound)
    #
    #         models[model_idx].covar_module.base_kernel.lengthscale = \
    #             models[model_idx].covar_module.base_kernel.raw_lengthscale_constraint.lower_bound + \
    #             torch.rand(models[model_idx].covar_module.base_kernel.lengthscale.shape) * \
    #             (models[model_idx].covar_module.base_kernel.raw_lengthscale_constraint.upper_bound - \
    #              models[model_idx].covar_module.base_kernel.raw_lengthscale_constraint.lower_bound)
    #
    #         models[model_idx].likelihood.noise = \
    #             models[model_idx].likelihood.noise_covar.raw_noise_constraint.lower_bound + \
    #             torch.rand(models[model_idx].likelihood.noise.shape) * \
    #             (models[model_idx].likelihood.noise_covar.raw_noise_constraint.upper_bound -
    #              models[model_idx].likelihood.noise_covar.raw_noise_constraint.lower_bound)
    #         mll = gpytorch.mlls.ExactMarginalLogLikelihood(models[model_idx].likelihood, models[model_idx])
    #         models[model_idx].train()
    #         models[model_idx].likelihood.train()
    #         optimizer = torch.optim.LBFGS([
    #             {'params': models[model_idx].parameters()},  # Includes GaussianLikelihood parameters
    #         ], lr=lr_train, line_search_fn='strong_wolfe')
    #         try:
    #             for i in range(num_iter_train):
    #                 def closure():
    #                     optimizer.zero_grad()
    #                     # Output from the model
    #                     output = models[model_idx](models[model_idx].train_inputs[0])
    #                     # Calculate loss and backpropagation gradients
    #                     loss = -mll(output, models[model_idx].train_targets)
    #                     loss.backward()
    #                     torch.nn.utils.clip_grad_value_(models[model_idx].parameters(), clip_grad_value)
    #                     return loss
    #                 loss = optimizer.step(closure)
    #                 if print_train and i % step_print_train == 0:
    #                     lengthscale = models[model_idx].covar_module.base_kernel.lengthscale.detach().numpy()
    #                     print(f'Iter {i + 1}/{num_iter_train} - \nLoss: {loss.item():.5f}   '
    #                           f'output_scale: {models[model_idx].covar_module.outputscale.item():.5f}   '
    #                           f'lengthscale: {str(lengthscale)}   '
    #                           f'noise: {models[model_idx].likelihood.noise.item() ** 0.5:.5f}')
    #
    #                 if loss < best_losses[model_idx]:
    #                     best_losses[model_idx] = loss.item()
    #                     best_lengthscales[model_idx] = models[model_idx].covar_module.base_kernel.lengthscale
    #                     best_noises[model_idx] = models[model_idx].likelihood.noise
    #                     best_outputscales[model_idx] = models[model_idx].covar_module.outputscale
    #
    #         except Exception as e:
    #             print(e)
    #
    #         print(
    #             'training process - model %d - time train %f - output_scale: %s - lengthscales: %s - noise: %s' % (
    #                 model_idx, time.time() - start_time, str(best_outputscales[model_idx].detach().numpy()),
    #                 str(best_lengthscales[model_idx].detach().numpy()),
    #                 str(best_noises[model_idx].detach().numpy())))
    #
    #     print('training process - previous marginal log likelihood: %s - new marginal log likelihood: %s' %
    #           (str(previous_losses.detach().numpy()), str(best_losses.detach().numpy())))
    #
    #     params_dict_list = []
    #     for model_idx in range(len(models)):
    #         params_dict_list.append({
    #             'covar_module.base_kernel.lengthscale': best_lengthscales[model_idx].detach().numpy(),
    #             'covar_module.outputscale': best_outputscales[model_idx].detach().numpy(),
    #             'likelihood.noise': best_noises[model_idx].detach().numpy()})
    #
    #
    #     queue.put(params_dict_list)
    @staticmethod
    def train(queue, train_inputs, train_targets, parameters, constraints_hyperparams, lr_train, num_iter_train,
              clip_grad_value, print_train=False, step_print_train=25, dof=False):
        # print("[DEBUG] train_inputs ", train_inputs)
        # print("[DEBUG] train_targets ", train_targets)
        # print("[DEBUG] train_inputs std per dim:", train_inputs.std(dim=0))
        # print("[DEBUG] train_targets std per dim:", train_targets.std(dim=0))
        torch.set_num_threads(1)
        start_time = time.time()

        models = create_models(train_inputs, train_targets, parameters, constraints_hyperparams, dof=dof)
        best_outputscales = [model.covar_module.outputscale.detach() for model in models]
        best_noises = [model.likelihood.noise.detach() for model in models]
        best_lengthscales = [model.covar_module.base_kernel.lengthscale.detach() for model in models]
        previous_losses = torch.empty(len(models))

        for model_idx in range(len(models)):
            output = models[model_idx](models[model_idx].train_inputs[0])
            mll = gpytorch.mlls.ExactMarginalLogLikelihood(models[model_idx].likelihood, models[model_idx])
            previous_losses[model_idx] = -mll(output, models[model_idx].train_targets)

        best_losses = previous_losses.detach().clone()

        for model_idx in range(len(models)):
            lengthscale_bounds = models[model_idx].covar_module.base_kernel.raw_lengthscale_constraint
            models[model_idx].covar_module.base_kernel.lengthscale = (
                    0.5 * (lengthscale_bounds.lower_bound + lengthscale_bounds.upper_bound) +
                    0.1 * (lengthscale_bounds.upper_bound - lengthscale_bounds.lower_bound) *
                    torch.rand_like(models[model_idx].covar_module.base_kernel.lengthscale)
            )

            models[model_idx].covar_module.outputscale = (
                    0.5 * (models[model_idx].covar_module.raw_outputscale_constraint.lower_bound +
                           models[model_idx].covar_module.raw_outputscale_constraint.upper_bound) +
                    0.1 * (models[model_idx].covar_module.raw_outputscale_constraint.upper_bound -
                           models[model_idx].covar_module.raw_outputscale_constraint.lower_bound) *
                    torch.rand_like(models[model_idx].covar_module.outputscale)
            )

            models[model_idx].likelihood.noise = (
                    0.5 * (models[model_idx].likelihood.noise_covar.raw_noise_constraint.lower_bound +
                           models[model_idx].likelihood.noise_covar.raw_noise_constraint.upper_bound) +
                    0.1 * (models[model_idx].likelihood.noise_covar.raw_noise_constraint.upper_bound -
                           models[model_idx].likelihood.noise_covar.raw_noise_constraint.lower_bound) *
                    torch.rand_like(models[model_idx].likelihood.noise)
            )

            mll = gpytorch.mlls.ExactMarginalLogLikelihood(models[model_idx].likelihood, models[model_idx])
            models[model_idx].train()
            models[model_idx].likelihood.train()

            optimizer = torch.optim.LBFGS([
                {'params': models[model_idx].parameters()},
            ], lr=lr_train, line_search_fn='strong_wolfe')

            try:
                for i in range(num_iter_train):
                    def closure():
                        optimizer.zero_grad()
                        output = models[model_idx](models[model_idx].train_inputs[0])
                        loss = -mll(output, models[model_idx].train_targets)
                        loss.backward()
                        torch.nn.utils.clip_grad_value_(models[model_idx].parameters(), clip_grad_value)
                        # for name, param in models[model_idx].named_parameters():
                            # if param.grad is not None:
                                # print(f"[DEBUG] Iter {i} - Param: {name} grad norm: {param.grad.norm():.4e}")
                        return loss

                    loss = optimizer.step(closure)

                    if print_train and i % step_print_train == 0:
                        lengthscale = models[model_idx].covar_module.base_kernel.lengthscale.detach().numpy()
                        print(f'Iter {i + 1}/{num_iter_train} - \nLoss: {loss.item():.5f}   '
                              f'output_scale: {models[model_idx].covar_module.outputscale.item():.5f}   '
                              f'lengthscale: {str(lengthscale)}   '
                              f'noise: {models[model_idx].likelihood.noise.item() ** 0.5:.5f}')

                    if loss < best_losses[model_idx]:
                        best_losses[model_idx] = loss.item()
                        best_lengthscales[model_idx] = models[model_idx].covar_module.base_kernel.lengthscale.clone()
                        best_noises[model_idx] = models[model_idx].likelihood.noise.clone()
                        best_outputscales[model_idx] = models[model_idx].covar_module.outputscale.clone()

            except Exception as e:
                print(f"[ERROR] Exception in training model {model_idx}: {e}")

            print('[RESULT] Model %d - time train %.6fs - output_scale: %s - lengthscales: %s - noise: %s' % (
                model_idx, time.time() - start_time, str(best_outputscales[model_idx].detach().numpy()),
                str(best_lengthscales[model_idx].detach().numpy()),
                str(best_noises[model_idx].detach().numpy())))

        print('Previous marginal log likelihood: %s - New marginal log likelihood: %s' % (
            str(previous_losses.detach().cpu().numpy()), str(best_losses.detach().cpu().numpy())))

        params_dict_list = []
        for model_idx in range(len(models)):
            params_dict_list.append({
                'covar_module.base_kernel.lengthscale': best_lengthscales[model_idx].detach().cpu().numpy(),
                'covar_module.outputscale': best_outputscales[model_idx].detach().cpu().numpy(),
                'likelihood.noise': best_noises[model_idx].detach().cpu().numpy()})

        queue.put(params_dict_list)

    def check_and_close_processes(self):
        """
		Check active parallel processes, wait for their resolution, get the parameters and close them
		"""
        if 'p_train' in self.__dict__ and not self.p_train._closed and not (self.p_train.is_alive()):
            params_dict_list = self.queue_train.get()
            self.p_train.join()
            for model_idx in range(len(self.models)):
                self.models[model_idx].initialize(**params_dict_list[model_idx])
            self.p_train.close()
            self.iK, self.beta = self.calculate_factorizations(self.x[self.idxs_mem_gp],
                                                               self.y[self.idxs_mem_gp], self.models, self.DoF)
            self.num_cores_main += 1
