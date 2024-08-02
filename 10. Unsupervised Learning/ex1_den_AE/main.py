import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
import os

current_directory = os.path.dirname(os.path.abspath(__file__))
download_path = os.path.join(current_directory, "data")
model_path = os.path.join(current_directory, "model.torch")
plot_path = os.path.join(current_directory, "plot.pdf")

# Set random seed for reproducibility
torch.manual_seed(12)
np.random.seed(12)

# Define the function to add Gaussian noise to the image
def add_noise(image, gw=10):
    noise = np.random.normal(loc=0, scale=gw, size=image.shape)
    noisy_image = image + noise
    return np.clip(noisy_image, 0, 255).astype(np.uint8)

# Function to add noise during the training loop
def add_noise_torch(image, gw=10):
    noise = torch.randn_like(image) * gw
    noisy_image = image + noise
    return torch.clamp(noisy_image, 0, 1)

# Define the denoising autoencoder model
class DenoisingAutoencoder(nn.Module):
    def __init__(self):
        super(DenoisingAutoencoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=2, stride=2),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# Function to plot images in a grid
def plot_images(images, title):
    fig, axes = plt.subplots(len(images), 3, figsize=(10, 2 * len(images)))
    fig.suptitle(title)

    for i, (base, noisy, denoised) in enumerate(images):
        axes[i, 0].imshow(base.squeeze(), cmap='gray')
        axes[i, 0].set_title('Base Image')

        axes[i, 1].imshow(noisy.squeeze(), cmap='gray')
        axes[i, 1].set_title('Noisy Image')

        axes[i, 2].imshow(denoised.squeeze(), cmap='gray')
        axes[i, 2].set_title('Denoised Image')

        for ax in axes[i]:
            ax.axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(plot_path)
    plt.show()

# Load FashionMNIST dataset
transform = transforms.Compose([transforms.ToTensor()])
full_dataset = datasets.FashionMNIST(root=download_path, train=True, download=True, transform=transform)

# Split the dataset into training and validation sets
train_size = int(0.80 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

# Create data loaders for training and validation
train_loader = DataLoader(train_dataset, batch_size=100, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=100, shuffle=False)

# Create denoising autoencoder model
model = DenoisingAutoencoder()

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=.001)

# Train the denoising autoencoder
num_epochs = 10
for epoch in range(num_epochs):
    model.train()  # Set model to training mode
    for data in train_loader:
        images, _ = data
        noisy_images = add_noise_torch(images)

        optimizer.zero_grad()
        outputs = model(noisy_images)
        loss = criterion(outputs, images)
        loss.backward()
        optimizer.step()

    # Calculate validation loss
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for val_data in val_loader:
            val_images, _ = val_data
            val_noisy_images = add_noise_torch(val_images)
            val_outputs = model(val_noisy_images)
            val_loss += criterion(val_outputs, val_images).item()

    average_val_loss = val_loss / len(val_loader.dataset)
    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}, Validation Loss: {average_val_loss:.4f}')

    if loss.item() < 0.03 :
        break

# Save the model
torch.save(model.state_dict(), model_path)

# Test the denoising autoencoder on some images
num_samples = 5
test_images = []

for i in range(num_samples):
    base_image, _ = train_dataset[i]
    noisy_image = add_noise_torch(base_image.unsqueeze(0)).squeeze()
    denoised_image = model(noisy_image.unsqueeze(0).unsqueeze(0)).squeeze().detach().numpy()

    test_images.append((base_image.numpy(), noisy_image.numpy(), denoised_image))

# Plot the results in a grid
plot_images(test_images, 'Denoising Autoencoder Results')

'''
import torch
from torchvision import datasets, transforms
import pandas as pd
import os
import numpy as np

# Standard Data Science
import pandas as pd
import numpy as np

# Data Processing
from sklearn.model_selection import train_test_split

# Visualization
import matplotlib.pyplot as plt

# Pytorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

current_directory = os.path.dirname(os.path.abspath(__file__))
download_path = os.path.join(current_directory, "data")
fashion_mnist_train_path = os.path.join(download_path, "fashion-mnist_train.csv")
fashion_mnist_test_path = os.path.join(download_path, "fashion-mnist_test.csv")
model_path = os.path.join(current_directory, "model.torch")
plot_path = os.path.join(current_directory, "plot.pdf")

def convert_to_csv(dataset, csv_filename, name):
    # Check if the file already exists
    if os.path.exists(csv_filename):
        print(f"{name} already exists. Skipping conversion.")
        return

    data = []
    for i in range(len(dataset)):
        image, label = dataset[i]
        image = image.numpy().ravel()  # Flatten the 28x28 image to a 1D array
        data.append([label] + list(image))

    columns = ['label'] + [f'pixel_{i}' for i in range(28 * 28)]
    df = pd.DataFrame(data, columns=columns)

    # Save the DataFrame to a CSV file
    df.to_csv(csv_filename, index=False)
    print(f"{csv_filename} created successfully.")

# Define a transform to normalize the data
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

# Download and convert the Fashion MNIST training dataset to CSV
train_dataset = datasets.FashionMNIST(root=download_path, train=True, transform=transform, download=True)
convert_to_csv(train_dataset, fashion_mnist_train_path, "fashion-mnist_train.csv")

# Download and convert the Fashion MNIST test dataset to CSV
test_dataset = datasets.FashionMNIST(root=download_path, train=False, transform=transform, download=True)
convert_to_csv(test_dataset, fashion_mnist_test_path, "fashion-mnist_test.csv")

train = pd.read_csv(fashion_mnist_train_path)
test = pd.read_csv(fashion_mnist_test_path)

print(f"Training Data size: {train.shape[0]}")
print(f"Testing Data size: {test.shape[0]}")

def normalize(x):
    """ Convert pixel values to [0,1]. """
    return x / 255  # Fix: changed X to x

X = train
y = X.pop('label')

X_test = test
y_test = X_test.pop('label')

X = normalize(X)
X_test = normalize(X_test)

# Function to add Gaussian noise to a numpy array
def add_noise(image: np.array, gw=10) -> np.array:
    noise = np.random.normal(0, np.sqrt(gw), image.shape)
    noisy_image = image + noise
    return np.clip(noisy_image, 0, 255).astype(np.uint8)

# Apply noise to all training and test images
X_noisy = np.array([add_noise(image) for image in X.values])
X_test_noisy = np.array([add_noise(image) for image in X_test.values])

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_noisy, y, test_size=0.2, shuffle=True, random_state=0)

# Function to create PyTorch DataLoader
def create_dataloader(my_x, my_y, batch_size=64, shuffle=True, num_workers=0):  # Fix: set num_workers to 0
    my_torch_x = torch.from_numpy(my_x).type(torch.FloatTensor)
    my_torch_y = torch.from_numpy(my_y.to_numpy())
    my_dataset = TensorDataset(my_torch_x, my_torch_y)
    my_dataloader = DataLoader(my_dataset,
                               batch_size=batch_size,
                               shuffle=shuffle,
                               num_workers=num_workers)
    return my_dataloader

if __name__ == '__main__':
    trn_dataloader = create_dataloader(X_train, y_train, shuffle=True)
    val_dataloader = create_dataloader(X_val, y_val, shuffle=False)

    # Function to add Gaussian noise to a PyTorch tensor
    def add_noise_torch(tensor, gw=10):
        noise = torch.randn_like(tensor) * np.sqrt(gw)
        noisy_tensor = tensor + noise
        return torch.clamp(noisy_tensor, 0, 1)  # Assuming input tensor is normalized [0, 1]

    # Define the AutoEncoder class
    class AutoEncoder(nn.Module):
        def __init__(self, img_size, lr_size):
            super(AutoEncoder, self).__init__()
            # Encoder
            self.e1 = nn.Linear(img_size * img_size, 500)
            self.e2 = nn.Linear(500, 250)
            # Latent Representation
            self.lr = nn.Linear(250, lr_size)
            # Decoder
            self.d1 = nn.Linear(lr_size, 250)
            self.d2 = nn.Linear(250, 500)
            # Output
            self.o1 = nn.Linear(500, img_size * img_size)
            
        def forward(self, x):
            # Encoder
            x = F.relu(self.e1(x))
            x = F.relu(self.e2(x))
            # Latent Representation
            x = torch.sigmoid(self.lr(x))
            x = F.relu(self.d1(x))
            x = F.relu(self.d2(x))
            x = self.o1(x)
            return x

    # Function to train the autoencoder
    def train(net, num_epochs, dataloader):
        loss_func = nn.MSELoss()
        optimizer = optim.AdamW(net.parameters(), lr=0.001)

        losses = []
        
        for epoch in range(num_epochs):
            for batch_idx, (data, target) in enumerate(dataloader):
                
                noisy_data = add_noise_torch(data)  # Adding noise during training

                optimizer.zero_grad()
                pred = net(noisy_data)
                loss = loss_func(pred, noisy_data)
                losses.append(loss.cpu().data.item())

                loss.backward()
                optimizer.step()

                # Display
                if batch_idx % 300 == 0:
                    print(f"EPOCH {epoch+1} [{batch_idx * len(data)}/{len(dataloader.dataset)}]"
                          f"({(batch_idx * len(data))/len(dataloader.dataset)*100.0:.1f}%)"
                          f" Loss={loss.cpu().data.item():.6f}")

        return losses

    # Instantiate the AutoEncoder model
    img_size = 28  # Assuming images are 28x28
    lr_size = 20   # Latent

    autoencoder = AutoEncoder(img_size, lr_size)

    # Train the autoencoder
    num_epochs = 100
    losses = train(autoencoder, num_epochs, trn_dataloader)

    # Save the trained model
    torch.save(autoencoder.state_dict(), model_path)
    print(f"Model saved to {model_path}")

    def display(img, img_size=28):
        """ Show image. """
        if type(img) == pd.Series:
            plt.imshow(img.to_numpy().reshape((img_size, img_size)), cmap='gray')
        else:
            plt.imshow(img.numpy().reshape((img_size, img_size)), cmap='gray')
        plt.show()

    def test(net, dataloader):
        net.eval()
        predictions = []
        for batch_idx, (data, target) in enumerate(dataloader):
            data = torch.autograd.Variable(data)
            pred = net(data)
            predictions.extend(pred)
        return predictions

    ae = AutoEncoder(28, 10)
    ae2 = AutoEncoder(28, 5)
    ae3 = AutoEncoder(28, 20)

    ae1_predictions = test(ae, val_dataloader)
    ae2_predictions = test(ae2, val_dataloader)
    ae3_predictions = test(ae3, val_dataloader)

    num = 0
    display(X.iloc[num])
    display(ae1_predictions[num].detach())
    display(ae2_predictions[num].detach())
    display(ae3_predictions[num].detach())

    '''