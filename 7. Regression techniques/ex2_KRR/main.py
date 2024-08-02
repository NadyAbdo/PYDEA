import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
from joblib import dump
import json
import os

# Get the current working directory
current_directory = os.path.dirname(os.path.abspath(__file__))

# Define the file paths for input and output files
file_path = os.path.join(current_directory, "wave.csv")

# Load data
data = pd.read_csv(file_path)
sorted_data = data.sort_values(by='x')

# split into train / validation + test set (80 ,20)
x_tv, x_test, y_tv, y_test = train_test_split(sorted_data['x'], sorted_data['y'], shuffle=True, test_size=.2)

# Save the splitted data to CSV files
train_data = pd.DataFrame({'x': x_tv, 'y': y_tv})
test_data = pd.DataFrame({'x': x_test, 'y': y_test})

train_data.to_csv(os.path.join(current_directory, 'train.csv'), index=False)
test_data.to_csv(os.path.join(current_directory, 'test.csv'), index=False)

# Now we want to fit a linear model
lm = LinearRegression()

# Now we want to fit a Kernel Ridge Regression model with an "rbf" kernel
krr = KernelRidge(kernel='rbf')

# Define the hyperparameters for the KernelRidge model
krr_params = {
    'alpha': [0.1, 1.0, 10.0],   # Adjust alpha values as needed
    'gamma': [0.1, 1.0, 10.0]    # Adjust gamma values as needed
}

# Use a 5-fold cross-validation and score by mean-squared-error (negative to choose the best result with max(score))
hypermodel = GridSearchCV(krr, krr_params, cv=5, scoring="neg_mean_squared_error")

# Now fit the model with the training data
hypermodel.fit(x_tv.values.reshape(-1, 1), y_tv)

# Save the trained KRR model to a file
model_filename = os.path.join(current_directory, 'model.joblib')
dump(hypermodel.best_estimator_, model_filename)

# Get the best hyperparameters for the LinearRegression model
best_params = hypermodel.best_params_

# Get all results
cv_results = hypermodel.cv_results_

# Now we can use the best model to predict test-data and calculate test-metrics
y_pred = hypermodel.predict(x_test.values.reshape(-1, 1))

#Now we can use the best model to predict test-data and calculate test-metrics
y_pred_test = hypermodel.predict(x_test.values.reshape(-1, 1))
y_pred_train = hypermodel.predict(x_tv.values.reshape(-1, 1))

# Evaluate the model on test data
test_mse = mean_squared_error(y_test, y_pred_test)
test_mae = mean_absolute_error(y_test, y_pred_test)
test_r2 = r2_score(y_test, y_pred_test)

# Evaluate the model on train data
train_mse = mean_squared_error(y_tv, y_pred_train)
train_mae = mean_absolute_error(y_tv, y_pred_train)
train_r2 = r2_score(y_tv, y_pred_train)

# Store scores in a dictionary
scores = {
    "test_mae" : test_mae,
    "test_mse" : test_mse,
    "test_r2" : test_r2,
    "train_mae" : train_mae,
    "train_mse" : train_mse,
    "train_r2" : train_r2
}

# Save the scores to a JSON file
scores_filename = os.path.join(current_directory, 'scores.json')
with open(scores_filename, 'w') as json_file:
    json.dump(scores, json_file, indent=4)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Plotting
X_points = np.linspace(-10, 10, 1000)
plt.scatter(x_tv, y_tv, label='training data', marker='o', color='blue')
plt.scatter(x_test, y_test, label='test data', marker='o', color='orange') 
plt.plot(X_points, hypermodel.predict(X_points.reshape(-1, 1)), label='predicted f', color='red') 
plt.plot(X_points, np.exp(-(X_points/4)**2) * np.cos(4*X_points), label='f', color='blue')
plt.title(f'MSE: {mse:.3f}, MAE: {mae:.3f}, R²: {r2:.3f}')
plt.legend(['f', 'predicted f', 'training data', 'test data'], loc='upper left')
plt.show()


# Save the argument text to a TXT file
args_text = """
To potentially enhance the performance of the Kernel Ridge Regression (KRR) model, the following strategies could be explored:

1. **Hyperparameter Tuning:** Conduct an extensive search for optimal hyperparameters to ensure the model is fine-tuned to the dataset.

2. **Feature Engineering:** Consider creating new relevant features or transforming existing ones to improve the model's ability to capture complex patterns.

3. **Data Scaling:** Standardize or normalize input features to ensure a consistent scale, which might positively impact the model's performance.

4. **Advanced Kernels:** Experiment with different kernel functions or combinations to capture more intricate relationships within the data.

5. **Ensemble Methods:** Explore ensemble methods by combining predictions from multiple models or integrating with other algorithms to improve overall generalization.
"""

args_filename = os.path.join(current_directory, 'args.txt')
with open(args_filename, 'w') as args_file:
    args_file.write(args_text)