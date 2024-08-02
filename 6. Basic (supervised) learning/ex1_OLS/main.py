import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import math as m
from typing import List
import os

# Get the current working directory
current_directory = os.path.dirname(os.path.abspath(__file__))

# Define the file paths for input and output files
file_path = os.path.join(current_directory, "data.csv")
output_path = os.path.join(current_directory, "coeffs.txt")
plot_path = os.path.join(current_directory, "plot.pdf")

def ols_fit(x, y, transformations: List[callable]) -> List[float]:
    transformed_x = np.column_stack([transformation(x) for transformation in transformations])
    model = LinearRegression()
    model.fit(transformed_x, y)
    return model.coef_

def target_function(x, a, b, c):
    return a * x**2 + b * x**5 + c * np.sin(x)

# Load data from data.csv
data = np.genfromtxt(file_path, delimiter=',', skip_header=1)

# Separate x and y values from the data
x_data = data[:, 0]
y_data = data[:, 1]

# Use ols_fit function to get coefficients
coeffs = ols_fit(x_data, y_data, [lambda x: x**2, lambda x: x**5, np.sin ])
a, b, c = coeffs

# Print the coefficients
print("Coefficients:", coeffs)

# Save coefficients to coeffs.txt
with open(output_path, 'w') as file:
    file.write(' '.join(map(str, coeffs)))

# Plot the regression model and the datapoints
plt.scatter(x_data, y_data)
x_range = np.linspace(min(x_data)-.75, max(x_data)+1, 100)
#plt.plot(x_range, target_function(x_range, *coeffs), color='blue', label='true function')
plt.plot(x_range, target_function(x_range, *coeffs),'--', color='orange', label='OLS fit')
plt.title(f'OLS fit, a={a:.3f}, b={b:.3f}, c={c:.3f}')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.xlim(-11.5,11.5)

# Save the plot as plot.pdf
plt.savefig(plot_path)

# Show the plot (debug)
plt.show()
