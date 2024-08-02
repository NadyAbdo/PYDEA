import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from joblib import dump
import os

# Get the current working directory
current_directory = os.path.dirname(os.path.abspath(__file__))

# Define the file paths for input and output files
csv_file_path = os.path.join(current_directory, "credit.csv")

# Load your dataset
data = pd.read_csv(csv_file_path)

# Assuming your dataset has columns: 'Rating', 'Limit', 'Cards', 'Income', and 'Balance'
X = data[['Rating', 'Limit', 'Cards', 'Income']]
y = data['Balance']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2) #, random_state=11

# Initialize variables to store results
alphas = np.arange(0, 10001, 1)  # Alpha values ranging from 0 to 10000
coefficients = []
r2_scores = []

# Build LASSO models for different alpha values
for alpha in alphas:
    # Create and fit the LASSO model
    lasso_model = Lasso(alpha=alpha, max_iter=10000)
    lasso_model.fit(X_train, y_train)

    #save models
    if alpha == 100 or alpha == 1000 or alpha == 10000 :
        for x in range(3) :
            model_file_path = os.path.join(current_directory, f'model_{x+1}.joblib')
            dump(lasso_model,model_file_path)

    # Store coefficients
    coefficients.append(lasso_model.coef_)
    
    # Make predictions on the test set and calculate R2 score
    y_pred = lasso_model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    r2_scores.append(r2)

# Plotting the coefficients and R2 scores vertically
plt.figure(figsize=(10, 7))  # Adjust the figure size according to your preference

# Coefficients plot
plt.subplot(2, 1, 1)  # 2 rows, 1 column, first subplot

# Plot each variable with a specific color
plt.plot(alphas, [coef[0] for coef in coefficients], color='blue')   # Rating in blue
plt.plot(alphas, [coef[1] for coef in coefficients], color='orange')  # Limit in orange
plt.plot(alphas, [coef[2] for coef in coefficients], color='red')     # Cards in red
plt.plot(alphas, [coef[3] for coef in coefficients], color='green')   # Income in green

plt.xscale('log')
plt.ylabel('Value of the coefficient')
plt.yticks([-5, 0, 5])

# Add a legend for better interpretation
plt.legend(['Rating', 'Limit', 'Income', 'Cards'], loc='lower left')


# R2 scores plot
plt.subplot(2, 1, 2)  # 2 rows, 1 column, second subplot
plt.plot(alphas, r2_scores)
plt.xscale('log')
plt.xlabel('Alpha')
plt.ylabel('R²')
plt.yticks([0.7, 0.8, 0.9])

# Show the plots
plt.tight_layout()
plt.show()
