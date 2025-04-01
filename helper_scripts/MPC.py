"""
Simon Hirlaender
This script optimizes control inputs for a model predictive control (MPC) system using the AwakeSteering environment.
The primary tasks include:
1. Defining a root mean square (RMS) function to measure deviations.
2. Predicting control actions by minimizing a cost function subject to system dynamics and constraints.
3. Visualizing results of state and control values over time, and analyzing performance across different tolerance settings in a parallelized execution environment.

Dependencies:
- autograd.numpy for automatic differentiation.
- scipy.optimize for numerical optimization.
- matplotlib for plotting results.
- concurrent.futures for parallel execution.
"""
import time

import matplotlib.pyplot as plt
# import autograd.numpy as np
import numpy as np
import time
from scipy.optimize import minimize

from awake_steering_simulated import AwakeSteering



def rms(x):
    return np.sqrt(np.mean((x ** 2)))




def predict_actions(initial_state, horizon_length, response_matrix, threshold, **kwargs):
    """
    Predicts the optimal actions using Model Predictive Control (MPC) with detailed optimization tracking.

    Args:
        initial_state (np.array): Initial system state.
        horizon_length (int): Prediction horizon.
        response_matrix (np.array): Response matrix for system dynamics.
        threshold (float): Control threshold.
        **kwargs: Additional optimization parameters.

    Returns:
        state_final (np.array): Optimized state trajectory.
        action_final (np.array): Optimized action sequence.
        costs (list): List of cost values per iteration.
        rms_final (np.array): RMS values of the state trajectory.
        time_run (float): Total optimization runtime.
    """

    # print("\n" + "*" * 30)
    # print(f"🔍 Optimization Setup:")
    # print(f"Initial State: {initial_state}")
    # print(f"Response Matrix: {response_matrix}")
    # print(f"Horizon Length: {horizon_length}, Threshold: {threshold}")
    # print("*" * 30 + "\n")

    # Optimization parameters
    disp = kwargs.get('disp', False)
    tol = kwargs.get('tol', 1e-16)
    discount_factor = kwargs.get('discount_factor', 0.9)

    dimension = response_matrix.shape[0]
    # print(f"State Dimension: {dimension}")

    def rms(x):
        """Compute Root Mean Square (RMS) of the input vector."""
        return np.sqrt(np.mean((x ** 2)))

    # System dynamics matrix (identity for simplicity)
    A = np.eye(dimension)

    # Initial guess for optimization
    z_initial = np.zeros(dimension * (horizon_length + 1) + dimension * horizon_length)
    # Change to meaningful initialisation
    z_initial[:dimension ] = initial_state# Set initial state
    # z_initial[: ] = initial_state  # Set initial state

    def step_function(x, threshold):
        """Step function for constraint enforcement."""
        return 1  # Always return 1 (modify if needed)

    def special_cost_function(x):
        """Cost function based on RMS."""
        return rms(x)

    # Define the objective function
    def objective(z):
        x = z[:dimension * (horizon_length + 1)].reshape((horizon_length + 1, dimension))
        # print(f'x {x}')
        # print(f'cost: {sum(special_cost_function(x_vector) for x_vector in x)}')
        # return sum(special_cost_function(x_vector) for x_vector in x)
        return sum(discount_factor ** t * special_cost_function(x_vector) for t, x_vector in enumerate(x))

    # Define system constraints
    def create_constraints():
        constraints = [{'type': 'eq', 'fun': lambda z: (z[:dimension] - initial_state).flatten()}]  # Initial condition
        # print('x' * 10)
        # print((z_initial[:dimension] - initial_state))
        # print('x' * 10)
        for i in range(horizon_length):
            constraints.append({
                'type': 'eq',
                'fun': (lambda z, k=i: (z[dimension * (k + 1):dimension * (k + 2)] -
                                        (step_function(rms(z[dimension * k:dimension * (k + 1)]), threshold) *
                                         (A @ z[dimension * k:dimension * (k + 1)] + response_matrix @
                                          z[dimension * (horizon_length + 1) + dimension * k:
                                            dimension * (horizon_length + 1) + dimension * (k + 1)]))
                                        ).flatten())
            })
            # print('x'*10)
            # print(z_initial[dimension * (i + 1):dimension * (i + 2)])
            # print((A @ z_initial[dimension * i:dimension * (i + 1)] + response_matrix @
            #                               z_initial[dimension * (horizon_length + 1) + dimension * i:
            #                                 dimension * (horizon_length + 1) + dimension * (i + 1)]))
            # print('x' * 10)
        return constraints

    # Define bounds (no limits on states, actions bounded in [-1, 1])
    control_bounds = 1
    bounds = [(-control_bounds, control_bounds)] * dimension * (horizon_length + 1) + [
        (-control_bounds, control_bounds)] * dimension * horizon_length

    # Storage for tracking optimization progress
    costs = []
    rms_run = []
    constraint_violations = []

    def callback(z):
        """Callback function to track optimization progress at each iteration."""
        current_x = z[:dimension * (horizon_length + 1)].reshape((horizon_length + 1, dimension))
        current_rms = [rms(x_vector) for x_vector in current_x]
        rms_run.append(current_rms)
        costs.append(objective(z))

        # Check constraint violations
        constraint_values = [con['fun'](z) for con in create_constraints()]
        violation_sum = sum(np.linalg.norm(v) for v in constraint_values)
        constraint_violations.append(violation_sum)

        # print(
        #     f"🔹 Iteration {len(costs)} | Cost: {costs[-1]:.6f} | RMS: {current_rms[0]:.6f} | Constraint Violation: {violation_sum:.6f}")

    def display_constraints(create_constraints, z_initial):
        """
        Displays constraints details including their type and initial violation.

        Args:
            create_constraints (function): Function that generates constraints.
            z_initial (np.array): Initial values for optimization variables.
        """
        print(f'z_initial {z_initial}')
        constraints = create_constraints()
        print("\n" + "=" * 50)
        print("🔍 **Constraint Details**")
        print("=" * 50)

        for i, con in enumerate(constraints):
            con_type = con['type']  # 'eq' for equality, 'ineq' for inequality
            con_value = con['fun'](z_initial)  # Evaluate the constraint at initial values

            print(f"🔹 Constraint {i + 1}")
            print(f"   - Type: {con_type}")
            print(f"   - Initial Value: {con_value}")
            print("-" * 50)

    # Generate and display constraints before optimization
    # display_constraints(create_constraints, z_initial)
    # Start timing
    start_time = time.time()

    # Run the optimization
    result = minimize(
        objective,
        z_initial,
        constraints=create_constraints(),
        bounds=bounds,
        method='SLSQP',
        options={'disp': disp, 'maxiter': 10000},
        tol=tol,
        callback=callback
    )

    # Compute total time
    time_run = time.time() - start_time

    # Extract final results
    rms_final = np.array(
        [special_cost_function(result.x[j * dimension:(j + 1) * dimension]) for j in range(horizon_length + 1)])
    state_final = result.x[:dimension * (horizon_length + 1)].reshape((horizon_length + 1, dimension))
    action_final = result.x[dimension * (horizon_length + 1):].reshape(horizon_length, dimension)

    # Print final summary
    print("\n" + "=" * 40)
    print("✅ Optimization Completed!")
    print(f"Final Objective Value: {result.fun:.6f}")
    print(f"Optimal Parameters: {result.x}")
    print(f"Total Iterations: {result.nit}")
    print(f"Total Function Evaluations: {result.nfev}")
    print(f"Total Jacobian Evaluations: {result.get('njev', 'N/A')}")
    print(f"Final RMS: {rms_final[-1]:.6f}")
    print(f"Total Constraint Violations: {constraint_violations[-1]:.6f}")
    print(f"Total Execution Time: {time_run:.4f} seconds")
    print("=" * 40 + "\n")

    return state_final, action_final, costs, rms_final, time_run


def model_predictive_control(x0, mpc_horizon, b, threshold, plot=False, **kwargs):
    x_final, u_final, costs, rms_final, time_run = predict_actions(x0, mpc_horizon, b, threshold, **kwargs)
    if plot:
        plot_results(x_final, u_final, costs, time_run, threshold)
    return u_final[0]


def plot_results(x_final, u_final, costs, time_run, threshold):
    dimension = x_final.shape[1]
    # print('threshold_inner', threshold)
    # Assuming x_final is (time_steps, dimensions)
    time_steps = np.arange(x_final.shape[0])
    control_steps = np.arange(1, u_final.shape[0] + 1)

    plt.figure(figsize=(15, 15))

    # RMS Over Time
    plt.subplot(3, 1, 2)
    rms_final = [rms(x) for x in x_final]
    plt.plot(rms_final, '-o', label='RMS Over Time')
    plt.axhline(y=threshold, color='r', linestyle='--', label='Threshold')
    plt.title('RMS Over Time (Final Solution)')
    plt.xlabel('Time Step')
    plt.ylabel('RMS of State Vector')
    plt.legend()
    plt.grid(True)

    # States Over Time
    plt.subplot(3, 1, 1)
    for dim in range(dimension):
        plt.plot(time_steps, x_final[:, dim], label=f'State {dim + 1}')

    plt.title('States Over Time')
    plt.xlabel('Time Step')
    plt.ylabel('State Value')
    plt.legend()
    plt.grid(True)

    # Control Inputs Over Time
    plt.subplot(3, 1, 3)
    for dim in range(dimension):
        plt.plot(control_steps, u_final[:, dim], label=f'Control {dim + 1}')
    plt.title('Control Inputs Over Time')
    plt.xlabel('Time Step')
    plt.ylabel('Control Value')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()
    plt.pause(2)

    # Cost Function Over Iterations
    plt.figure(figsize=(8, 5))
    plt.plot(costs, '-x', label='Cost Over Iterations')
    plt.title('Cost Function Over Iterations')
    plt.xlabel('Iteration')
    plt.ylabel('Cost Function Value')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.show()

    # Print timing information
    print(f"Total execution time: {time_run:.2f}s")



