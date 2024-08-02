import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, learning_curve
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import joblib
import os

# Get the current working directory
current_directory = os.path.dirname(os.path.abspath(__file__))

# Define the file paths for input and output files
file_path = os.path.join(current_directory, "nitride_compounds.csv")

# Load the data
data = pd.read_csv(file_path)

# Extract features (R1-I8 columns) and target variable (HSE band gap)
X = data[['R1', 'X1', 'V1', 'R2', 'X2', 'V3', 'R3', 'X3', 'R4', 'X4', 'R5', 'X5', 'R6', 'X6', 'R7', 'X7', 'R8', 'X8', 'I1', 'I2', 'I3', 'I4', 'I5', 'I6', 'I7', 'I8']]
y = data['HSE Eg (eV)']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define KernelRidge model
model = KernelRidge(kernel='rbf')

# Hyperparameter tuning using GridSearchCV
param_grid = {'alpha': [0.1, 1.0, 10.0],
              'gamma': [0.01, 0.1, 1.0]}
grid_search = GridSearchCV(model, param_grid, cv=5, scoring='r2')
grid_search.fit(X_train_scaled, y_train)

# Get the best model from the grid search
best_model = grid_search.best_estimator_

# Save the best model
model_filename = os.path.join(current_directory, 'model.joblib')
joblib.dump(best_model, model_filename)

# Predict on the test set
y_pred_best = best_model.predict(X_test_scaled)

# Learning Curve
train_sizes, train_scores, test_scores = learning_curve(best_model, X, y, cv=5, scoring='neg_mean_squared_error')

# Calculate the percentage of training data used
percentage_train_sizes = (train_sizes / len(X))
train_r2_scores = [r2_score(y_train[:int(size)], best_model.predict(X_train_scaled[:int(size)])) for size in train_sizes]
train_mae_scores = [mean_absolute_error(y_train[:int(size)], best_model.predict(X_train_scaled[:int(size)])) for size in train_sizes]

# Calculate R2 score and MSE on the test set for the best model
highest_r2_index = max(range(len(train_r2_scores)), key=train_r2_scores.__getitem__)
r2_best = train_r2_scores[highest_r2_index]
highest_mae_index = max(range(len(train_mae_scores)), key=train_mae_scores.__getitem__)
mae_best = train_mae_scores[highest_mae_index]
print(f'R2 Score for the Best Model: {r2_best}')

# Plot the learning curve with MSE and training data percentage
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(percentage_train_sizes, -train_scores.mean(axis=1))
plt.xlabel('fraction of training data used')
plt.ylabel('MSE', color='blue')
plt.tick_params(axis='y', labelcolor='blue')

# Add another y-axis for R2
plt.twinx()  # Create a new y-axis sharing the same x-axis
plt.plot(percentage_train_sizes, train_r2_scores, label='Training R2', color='red')
plt.ylabel('R²', color='red')
plt.tick_params(axis='y', labelcolor='red')

# Model Performance Plot
plt.subplot(1, 2, 2)
plt.scatter(y_train, best_model.predict(X_train_scaled), label='training Data', color='blue')
plt.scatter(y_test, y_pred_best, label='test Data', color='orange')
plt.plot([y.min()-1, y.max()+1], [y.min()-1, y.max()+1], color='blue')
plt.xlabel('Calculated gap')
plt.ylabel('Model gap')
plt.xlim(0, 6)
plt.ylim(0, 6)
plt.legend()
plt.title(f'Model R²: {r2_best:.3f}, MAE: {mae_best:.3f}')

# Show the plots
plt.tight_layout()
plt.show()
