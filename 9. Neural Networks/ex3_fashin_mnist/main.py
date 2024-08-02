import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
import os

current_directory = os.path.dirname(os.path.abspath(__file__))
download_path = os.path.join(current_directory, "data")
model_path = os.path.join(current_directory, "model.torch")

# Set random seed for reproducibility
torch.manual_seed(12)
np.random.seed(12)

# Load Fashion-MNIST dataset
transform = transforms.Compose([transforms.ToTensor()])
train_data = torchvision.datasets.FashionMNIST(root=download_path, train=True, download=True, transform=transform)
test_data = torchvision.datasets.FashionMNIST(root=download_path, train=False, download=True, transform=transform)

# Define neural network model
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fc1 = nn.Linear(in_features=64*6*6, out_features=600)
        self.drop = nn.Dropout(0.25)
        self.fc2 = nn.Linear(in_features=600, out_features=120)
        self.fc3 = nn.Linear(in_features=120, out_features=10)
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.drop(out)
        out = self.fc2(out)
        out = self.fc3(out)
        return out

# Instantiate the model, loss function, and optimizer
model = SimpleNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Set up data loaders for training and validation
train_size = int(0.95 * len(train_data))
train_data, val_data = torch.utils.data.random_split( train_data, (train_size, len(train_data) - train_size))

train_loader = data.DataLoader(train_data, batch_size=256, shuffle=True)
val_loader = data.DataLoader(val_data, batch_size=256, shuffle=False)

# Training loop
num_epochs = 40
train_losses = []
val_losses = []


for epoch in range(num_epochs):
    model.train()
    running_train_loss = 0.0

    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_train_loss += loss.item()

    average_train_loss = running_train_loss / len(train_loader.dataset)
    train_losses.append(average_train_loss)

    # Validation loop
    model.eval()
    running_val_loss = 0.0

    with torch.no_grad():
        for inputs, labels in val_loader:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_val_loss += loss.item()

    average_val_loss = running_val_loss / len(val_loader.dataset)
    val_losses.append(average_val_loss)

    print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {average_train_loss}, Val Loss: {average_val_loss}")

    # Evaluate on the test set
    model.eval()
    correct_predictions = 0
    total_samples = len(test_data)

    with torch.no_grad():
        for sample, label in test_data:
            output = model(sample.unsqueeze(0))
            prediction = torch.argmax(output).item()
            correct_predictions += (prediction == label)
    print(f"Model Accuracy: {correct_predictions / total_samples:.2%}")
    
# Evaluate on the test set
model.eval()
correct_predictions = 0
total_samples = len(test_data)

with torch.no_grad():
    for sample, label in test_data:
        output = model(sample.unsqueeze(0))
        prediction = torch.argmax(output).item()
        correct_predictions += (prediction == label)

test_accuracy = correct_predictions / total_samples
print(f"... Final Model Accuracy: {test_accuracy:.2%}")


# Plot learning curve
plt.plot(range(1, num_epochs + 1), train_losses, label='Train Loss')
plt.plot(range(1, num_epochs + 1), val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Crossentropy Loss')
plt.suptitle(f'Test Accuracy: {test_accuracy:.2}')
plt.legend()
plt.show()

# Display 30 samples with annotations in one plot
model.eval()
with torch.no_grad():
    num_samples = 30
    num_rows = 5
    num_cols = 6

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(12, 8))
    axes = axes.flatten()

    sample_indices = np.random.choice(len(test_data), num_samples, replace=False)
    for i, index in enumerate(sample_indices):
        sample, label = test_data[index]
        output = model(sample.unsqueeze(0))
        prediction = torch.argmax(output).item()

        axes[i].imshow(sample.squeeze().numpy(), cmap='gray')
        axes[i].set_title(f"True: {label}, Predicted: {prediction}")
        axes[i].axis('off')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout to include title
    plt.show()

# Save the model
if model_accuracy < 0.9 :
        print("Condition not met. Rerun the code.")
else :
    print("Condition met. Model accuracy is greater than 90%.")
    print("Model Saved.")
    torch.jit.save(torch.jit.script(model), model_path)
