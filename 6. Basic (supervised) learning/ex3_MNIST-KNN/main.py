import os
import gzip
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import joblib
import matplotlib.pyplot as plt

# Get the current working directory
current_directory = os.path.dirname(os.path.abspath(__file__))
files_path = os.path.join(current_directory, "__files")
plot_path = os.path.join(current_directory, "numbers.pdf")
accuracy_plot_path = os.path.join(current_directory, "knn.pdf")

def unzip_files_in_directory(files_path):
    # Iterate over all files in the directory
    for filename in os.listdir(files_path):
        if filename.endswith('.gz'):
            gz_file_path = os.path.join(files_path, filename)
            decompressed_file_path = gz_file_path.rstrip('.gz')

            # Check if the decompressed file already exists, if not, unzip the file
            if not os.path.exists(decompressed_file_path):
                with gzip.open(gz_file_path, 'rb') as f_in, open(decompressed_file_path, 'wb') as f_out:
                    f_out.write(f_in.read())

def load_mnist(folder='__files', train=True):
    if not os.path.exists(folder):
        print(f"Folder '{folder}' does not exist.")
        return None
    
    unzip_files_in_directory(files_path)

    prefix = 'train' if train else 't10k'
    images_path = os.path.join(folder, f'{prefix}-images-idx3-ubyte')
    labels_path = os.path.join(folder, f'{prefix}-labels-idx1-ubyte')

    with open(labels_path, 'rb') as file:
        labels = np.frombuffer(file.read(), dtype=np.uint8, offset=8)

    with open(images_path, 'rb') as file:
        images = np.frombuffer(file.read(), dtype=np.uint8, offset=16).reshape(len(labels), -1)

    return images, labels

def plot_images(images, labels, n=10):
    fig, axes = plt.subplots(1, n, figsize=(20, 2))

    for i in range(n):
        axes[i].imshow(images[i].reshape(28, 28), cmap='gray')
        axes[i].set_title(f'Label: {labels[i]}')
        axes[i].axis('off')
    plt.savefig(plot_path)
    plt.show()

def plot_knn(neighbors_list, accuracies):
    plt.figure(figsize=(12, 6))  # Set the figure size to be wider (12 units) and longer (6 units)
    plt.plot(neighbors_list, accuracies, marker='o')
    #plt.title('KNN Classifier Accuracy for Different Numbers of Neighbors')
    plt.xlabel('Number of NN')
    plt.ylabel('Accuracy')
    plt.xticks(neighbors_list)
    plt.yticks(np.arange(0, 1.2, 0.2))
    plt.savefig(accuracy_plot_path)
    plt.show()

def train_and_evaluate_knn(images_train, labels_train, images_test, labels_test, neighbors):
    knn = KNeighborsClassifier(n_neighbors=neighbors)
    knn.fit(images_train, labels_train)
    predictions = knn.predict(images_test)
    accuracy = accuracy_score(labels_test, predictions)
    return accuracy, knn

def save_model(model, filename='model.sk'):
    current_directory = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_directory, filename)
    joblib.dump(model, model_path)

# Load MNIST dataset
images, labels = load_mnist(folder=files_path, train=True)

# Split the dataset into train and test sets
images_train, images_test, labels_train, labels_test = train_test_split(images, labels, test_size=0.2, random_state=42)

# Plot some example images
plot_images(images_train, labels_train)

# Train and evaluate KNN classifier for different settings
neighbors_list = [1,2, 3, 4, 5, 7, 10, 15, 20]
accuracies = []
best_accuracy = 0
best_knn_model = None

for neighbors in neighbors_list:
    accuracy, knn_model = train_and_evaluate_knn(images_train, labels_train, images_test, labels_test, neighbors)
    accuracies.append(accuracy)
    print(f'Neighbors: {neighbors}, Accuracy: {accuracy}')
    
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_knn_model = knn_model

# Plot KNN graph
plot_knn(neighbors_list, accuracies)

# Save the best performing model
save_model(best_knn_model, filename='model.sk')
