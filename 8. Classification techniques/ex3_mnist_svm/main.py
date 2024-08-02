import numpy as np
import gzip
from sklearn import datasets, model_selection, svm
import matplotlib.pyplot as plt
import os
import joblib

current_directory = os.path.dirname(os.path.abspath(__file__))
plot_path = os.path.join(current_directory, "plot.pdf")

def load_mnist_images(filename):
    file_path = os.path.join(current_directory, filename)
    with gzip.open(file_path, 'rb') as f:
        data = np.frombuffer(f.read(), dtype=np.uint8, offset=16)
    return data.reshape(-1, 28, 28)

def load_mnist_labels(filename):
    file_path = os.path.join(current_directory, filename)
    with gzip.open(file_path, 'rb') as f:
        labels = np.frombuffer(f.read(), dtype=np.uint8, offset=8)
    return labels

# Load MNIST dataset
X_train = load_mnist_images('train-images-idx3-ubyte.gz')#[:60000]
y_train = load_mnist_labels('train-labels-idx1-ubyte.gz')#[:60000]
X_test = load_mnist_images('t10k-images-idx3-ubyte.gz')#[:10000]
y_test = load_mnist_labels('t10k-labels-idx1-ubyte.gz')#[:10000]

# Flatten the images for SVM
X_train_flat = X_train.reshape(X_train.shape[0], -1)
X_test_flat = X_test.reshape(X_test.shape[0], -1)

# Reshape labels for SVM
y_train = y_train.reshape(-1)
y_test = y_test.reshape(-1)

# Build SVM model
model = svm.SVC(kernel='rbf', C=10)

# Fit the model
model.fit(X_train_flat, y_train)

# Evaluate the model
accuracy = model.score(X_test_flat, y_test)
print(f'Test Accuracy: {accuracy * 100}%')

# Save the model
model_path = os.path.join(current_directory, 'model.joblib')
joblib.dump(model, model_path)

# Plot a grid of 10 test results
predictions = model.predict(X_test_flat[:10])

plt.figure(figsize=(10, 5))
for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(X_test[i], cmap='gray')
    plt.title(f'True: {int(y_test[i])}, Predicted: {int(predictions[i])}')
    plt.axis('off')

plt.tight_layout()
plt.savefig(plot_path)
