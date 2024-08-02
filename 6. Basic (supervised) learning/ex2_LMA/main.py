import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# Get the current working directory
current_directory = os.path.dirname(os.path.abspath(__file__))

# Define the file paths for input and output files
file_path = os.path.join(current_directory, "breit_wigner.csv")
plot_path = os.path.join(current_directory, "plot.pdf")

def breit_wigner(x, a, b, c):
    return a / ((b - x)**2 + c)

def eval_jacobian(x, func, params, h=0.0001):

    num_variables = len(params)
    num_functions = len(func(x,*params))

    jacobian_matrix = np.zeros((num_functions, num_variables))

    for i in range(num_variables):
        perturbation = np.zeros_like(params)
        perturbation[i] = h

        # Forward finite difference
        func_plus_epsilon = func(x,*params + perturbation)
        func_minus_epsilon = func(x,*params - perturbation)

        partial_derivative = (func_plus_epsilon - func_minus_epsilon) / (h*2)

        jacobian_matrix[:, i] = partial_derivative

    return jacobian_matrix

def eval_errors(x_all, y_all, func, params):
    return y_all - func ( x_all , * params )

def _lma_quality_measure(y_all, y_fit, lma_lambda):
    residuals = y_all - y_fit
    rho = (np.linalg.norm(residuals)**2) / (2 * lma_lambda)
    return rho

def lma(x_data, y_data, func, param_guess, max_iterations=1000, tolerance=.00001, lma_lambda=None):
    params = np.array(param_guess, dtype=float)

    for iteration in range(max_iterations):
        # Update parameters using Levenberg-Marquardt formula
        y_fit = func(x_data, *params)
        residuals = y_data - y_fit

        # Calculate Jacobian
        jac = eval_jacobian(x_data, func, params)

        if lma_lambda is None:
            lma_lambda = np.linalg.norm(jac.T @ jac)

        # Update parameters
        delta_params = np.linalg.inv(jac.T @ jac + lma_lambda * np.eye(len(params))) @ jac.T @ residuals
        new_params = params + delta_params

        # Calculate quality measure
        lma_rho = _lma_quality_measure(y_data, y_fit, lma_lambda)

        # Adjust damping parameter based on quality measure
        if lma_rho > 0.75:
            lma_lambda /= 3
        elif lma_rho < 0.25:
            lma_lambda *= 2

        # Update parameters only if the quality measure is greater than 0
        if lma_rho > 0:
            params = new_params

        # Check convergence
        param_change = np.linalg.norm(delta_params) / np.linalg.norm(params)
        #print(f"Iteration {iteration + 1}: Param Change = {param_change}, Quality Measure = {lma_rho}, Lambda = {lma_lambda}")     #debug

        if param_change < tolerance:
            break

    return params

def plot_fit(x_data, y_data, func, params):
    a ,b ,c = params
    x_curve = np.linspace(min(x_data), max(x_data), 1000)
    plt.plot(x_data, y_data, 'o')
    plt.plot(x_curve, func(x_curve, *params))
    #print(func(x_curve, *params))      #debug
    plt.title(f'Breit-Wigner function for params a={a:.3f}, b={b:.3f}, c={c:.3f}')
    plt.savefig(plot_path)
    plt.show()

# Load data from breit_wigner.csv
data = pd.read_csv(file_path)
X_all = data["x"].to_numpy()
y_all = data["g"].to_numpy()

# Define initial parameter guess
initial_params = [0.1, 0.2, 0.1]

# Perform Levenberg-Marquardt fitting with quality measure and damping adjustment
fitted_params = lma(X_all, y_all, breit_wigner, initial_params)

print("Fitted Parameters:", fitted_params)

# Plot the data and the fitted curve
plot_fit(X_all, y_all, breit_wigner, fitted_params)

'''
import numpy as np
import pandas as pd
from scipy.optimize import leastsq
import matplotlib.pyplot as plt
import os

# Get the current working directory
current_directory = os.path.dirname(os.path.abspath(__file__))

# Define the file paths for input and output files
file_path = os.path.join(current_directory, "breit_wigner.csv")
plot_path = os.path.join(current_directory, "plot.pdf")

# Define the Breit-Wigner function
def breit_wigner(params, x):
    a, b, c = params
    return a / ((x - b)**2 + c)

# Define the objective function (residuals)
def eval_errors(params, x, y):
    return y - breit_wigner(params, x)

# Generate some example data
data = pd.read_csv(file_path)
x_data = data[["x"]].to_numpy().flatten()
y_data = data["g"].to_numpy().flatten()

# Initial guess for the parameters
initial_params = [1.0, 2.0, 0.2]

# Fit the data using the Levenberg-Marquardt algorithm
result, _ = leastsq(eval_errors, initial_params, args=(x_data, y_data))
a ,b ,c = result

# Plot the results
x_curve = np.linspace(min(x_data), max(x_data), 1000)
plt.plot(x_data, y_data, 'o')
plt.plot(x_curve, breit_wigner(result, x_curve))
plt.title(f'Breit-Wigner function for params a={a:.3f}, b={b:.3f}, c={c:.3f}')
plt.savefig(plot_path)
plt.show()
'''
