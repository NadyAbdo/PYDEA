import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score
import joblib
import json
import matplotlib.pyplot as plt
import os

# Get the current working directory
current_directory = os.path.dirname(os.path.abspath(__file__))

# Define the file paths for input and output files
csv_file_path = os.path.join(current_directory, "sba_small.csv")

# Load the data
data = pd.read_csv(csv_file_path)

# (a) Split the data into train/test sets by the Selected column
train_data, test_data = train_test_split(data, test_size=0.2, stratify=data['Selected'])

# Save train and test data to CSV without the index
train_data.to_csv(os.path.join(current_directory, 'train.csv'), index=False)
test_data.to_csv(os.path.join(current_directory, 'test.csv'), index=False)

# (b) Build a logistic model using specified columns
features = ['Recession', 'RealEstate', 'Portion']
X_train = train_data[features]
y_train = train_data['Default']

model = LogisticRegression()
model.fit(X_train, y_train)

# Save the model
model_filename = os.path.join(current_directory, 'model.joblib')
joblib.dump(model, model_filename)

# (c) Save a confusion matrix of the test set to a JSON file
X_test = test_data[features]
y_test = test_data['Default']

y_pred = model.predict(X_test)
conf_matrix = confusion_matrix(y_test, y_pred)

# Convert int64 to int for JSON serialization
conf_matrix_dict = {
    'tn': int(conf_matrix[0, 0]),
    'fp': int(conf_matrix[0, 1]),
    'fn': int(conf_matrix[1, 0]),
    'tp': int(conf_matrix[1, 1])
}

with open(os.path.join(current_directory, 'confusion_matrix.json'), 'w') as json_file:
    json.dump(conf_matrix_dict, json_file)

# (d) Plot the ROC curve
y_score = model.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_score)

plt.figure(figsize=(8, 8))
plt.plot(fpr, tpr, label=f'ROC curve = {roc_auc_score(y_test, y_score):.2f}', color='orange')
plt.plot([0, 1], [0, 1], '--', color='blue')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
#plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.xlim([0, 1])
plt.ylim([0, 1.02])
plt.legend(loc='lower right')
plt.show()
