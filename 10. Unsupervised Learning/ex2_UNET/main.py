import torch
import torch.nn as nn
from monai.networks.nets import UNet
from torch.optim import Adam
from monai.transforms import Compose, AddChannel, RandFlip, RandRotate90, ToTensor
from monai.data import Dataset, DataLoader
import os
import tifffile  # Make sure to install this library
import torch.nn.functional as F 
from monai.data.utils import pad_list_data_collate


# Define the UNet model
class MyUNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MyUNet, self).__init__()
        self.unet = UNet(
            dimensions=2,
            in_channels=in_channels,
            out_channels=out_channels,
            channels=(16, 32, 64, 128, 256),
            strides=(2, 2, 2, 2),
            num_res_units=2,
            dropout=0.1,
            kernel_size=3,
        )

    def forward(self, x):
        # Get the spatial dimensions of the input and the expected size
        spatial_dims = x.shape[-2:]
        expected_size = (65, 65)  # Adjust this size based on your model's architecture

        # Crop or pad the input tensor to match the expected size
        if spatial_dims != expected_size:
            diff_y = expected_size[0] - spatial_dims[0]
            diff_x = expected_size[1] - spatial_dims[1]

            x = F.pad(x, [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2])

        return self.unet(x)

# Define the dataset class
class YourLIVECellDataset(Dataset):
    def __init__(self, image_files, transform=None):
        self.image_files = image_files
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        # Load image files here
        image = tifffile.imread(self.image_files[index])

        if self.transform:
            image = self.transform(image)

        return image.to(torch.float)  # Convert to float using torch.to

# Define transformations
transforms = Compose([
    AddChannel(),
    RandFlip(spatial_axis=0),
    RandRotate90(prob=0.5),
    ToTensor(),
])

# Define dataset and dataloader for training
current_directory = os.path.dirname(os.path.abspath(__file__))
train_folder = os.path.join(current_directory, "images/livecell_train_val_images")
train_image_list = [os.path.join(train_folder, img) for img in os.listdir(train_folder)]
batch_size = 2
train_dataset = YourLIVECellDataset(train_image_list, transform=transforms)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, collate_fn=pad_list_data_collate)

# Initialize model, loss, and optimizer
in_channels = 1  # Modify based on your input data
out_channels = 1  # Modify based on your output data
model = MyUNet(in_channels, out_channels)
criterion = nn.MSELoss()  # Use appropriate loss function for regression
optimizer = Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 10

for epoch in range(num_epochs):
    model.train()
    for inputs in train_dataloader:
        optimizer.zero_grad()
        outputs = model(inputs)
        
        # Here, you would need a ground truth to compute the loss.
        # In weak supervision, you might use some heuristic or unsupervised method to generate pseudo-labels.

        loss = criterion(outputs, pseudo_labels)
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch + 1}/{num_epochs}, Training Loss: {loss.item()}")

# Save the model
torch.save(model.state_dict(), 'model.pt')
