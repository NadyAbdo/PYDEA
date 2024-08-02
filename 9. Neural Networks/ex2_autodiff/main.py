import torch
import numpy as np
import matplotlib.pyplot as plt
import os

def polynomial(X):
    return (-0.0533 * (X**3)) + (0.2 * (X**2)) + (10 * X)

def breit_wigner(X, params):
    a, b, c = params
    return a / ((b - X)**2 + c)

def eval_jacobian_torch(X, func, params):
    # Define a function that returns the evaluated function for given parameters
    fixed_func = lambda p: func(X, p)
    
    # Calculate Jacobian using torch.autograd.functional.jacobian
    jacobian = torch.autograd.functional.jacobian(fixed_func, params)
    
    return jacobian

# Create the input tensor x
#x = torch.tensor(np.linspace(-15, 15, num=100).reshape(-1, 1), dtype=torch.float, requires_grad=True)


# Create the input tensor x
x = torch.tensor(np.linspace(-15, 15, num=100).reshape(-1, 1), dtype=torch.float, requires_grad=True)

# Example usage:
params_example = torch.tensor([80e3, 80, 600], dtype=torch.float, requires_grad=True)

# Evaluate the functions
result_polynomial = polynomial(x)
result_breit_wigner = breit_wigner(x, params_example)

# Derivative calculation for polynomial function
y = polynomial(x).sum()
dy_dx = torch.autograd.grad(y, x, create_graph=True)[0]

# Second derivative calculation for polynomial function
d2y_dx2 = torch.autograd.grad(dy_dx.sum(), x, create_graph=True)[0]

# Convert tensors to NumPy arrays for plotting
x_np = x.detach().numpy().flatten()
y_np = result_polynomial.detach().numpy().flatten()
dy_dx_np = dy_dx.detach().numpy().flatten()
d2y_dx2_np = d2y_dx2.detach().numpy().flatten()

# Plot the results
plt.figure(figsize=(10, 6))

plt.plot(x_np, y_np, label='Polynomial')
plt.plot(x_np, dy_dx_np, label='First Derivative')
plt.plot(x_np, d2y_dx2_np, label='Second Derivative')

# Set axis limits
plt.xlim(-20, 20)
plt.ylim(-100, 100)

plt.title('Polynomial and its Derivatives')
plt.legend()

# Save the plot to the specified path
current_directory = os.path.dirname(os.path.abspath(__file__))
plot_path = os.path.join(current_directory, "plot.pdf")
plt.savefig(plot_path)
plt.show()

# Evaluate the Jacobian for the breit_wigner function
bw_params = torch.tensor([80e3, 80, 600], dtype=torch.float, requires_grad=True)
jacobian_bw = eval_jacobian_torch(x, breit_wigner, bw_params)
#print(jacobian_bw)