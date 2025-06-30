import torch
from torchvision.datasets import MNIST
from torchvision import transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Define a transform to convert images to tensors and normalize them
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Load the MNIST training dataset
train_dataset = MNIST(root='./data', train=True, download=True, transform=transform)

# Load the MNIST test dataset
test_dataset = MNIST(root='./data', train=False, download=True, transform=transform)

# Print the size of the datasets to verify
print(f"Size of training dataset: {len(train_dataset)}")
print(f"Size of test dataset: {len(test_dataset)}")

# Access a single sample from the train_dataset using indexing (which calls __getitem__)
# Let's get the sample at index 0
image_sample, label_sample = train_dataset[0]

# Print information about the single sample
print("Shape of the single image sample:", image_sample.shape)
print("Data type of the single image sample:", image_sample.dtype)
print("Value of the single label sample:", label_sample)

# You can also access other samples by changing the index, e.g., train_dataset[1], train_dataset[100], etc.


from torch.utils.data import DataLoader

# Create data loaders
batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Print the number of batches
print(f"Number of batches in train_loader: {len(train_loader)}")
print(f"Number of batches in test_loader: {len(test_loader)}")

# Get one batch of training data
images, labels = next(iter(train_loader))

# Print the shape of images and labels in the batch
print("Shape of images in one training batch:", images.shape)
print("Shape of labels in one training batch:", labels.shape)

import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # First convolutional layer
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        # Second convolutional layer
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        # Max pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # Fully connected layers
        # Calculate the size of the flattened output after conv and pool layers
        # Input image size is 28x28
        # After conv1 (3x3 kernel, no padding): (28 - 3 + 1) = 26x26
        # After pool1 (2x2 kernel, stride 2): 26 / 2 = 13x13
        # After conv2 (3x3 kernel, no padding): (13 - 3 + 1) = 11x11
        # After pool2 (2x2 kernel, stride 2): 11 / 2 = 5x5 (integer division)
        # The number of channels after conv2 is 64
        self.fc1 = nn.Linear(64 * 5 * 5, 128)
        self.fc2 = nn.Linear(128, 10) # 10 output classes for digits 0-9

    def forward(self, x):
        # Apply first conv and pool, then ReLU
        x = self.pool(F.relu(self.conv1(x)))
        # Apply second conv and pool, then ReLU
        x = self.pool(F.relu(self.conv2(x)))
        # Flatten the output for the fully connected layers
        x = x.view(-1, 64 * 5 * 5)
        # Apply first fully connected layer and ReLU
        x = F.relu(self.fc1(x))
        # Apply second fully connected layer (output layer)
        x = self.fc2(x)
        return x

# Instantiate and move model to device
model = Net().to(device)
print(model)

# Print the model structure
print(model)

import torch.optim as optim
import torch.nn as nn

# Specify the loss function
criterion = nn.CrossEntropyLoss()

# Specify the optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)

print("Loss function and optimizer specified.")

# Set the model to training mode
model.train()

epochs = 5
for epoch in range(epochs):
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        # Get the inputs and labels
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device) # Move to device (GPU or CPU)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)

        # Calculate the loss
        loss = criterion(outputs, labels)

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        # Print statistics
        running_loss += loss.item()
        if i % 100 == 99:    # print every 100 batches
            print(f'Epoch [{epoch + 1}/{epochs}], Batch [{i + 1}/{len(train_loader)}], Loss: {running_loss / 100:.4f}')
            running_loss = 0.0

    # Optional: Evaluate on the test set after each epoch
    model.eval() # Set the model to evaluation mode
    correct = 0
    total = 0
    with torch.no_grad(): # Disable gradient calculation for evaluation
        for data in test_loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device) # Move to device (GPU or CPU)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Epoch [{epoch + 1}/{epochs}], Test Accuracy: {100 * correct / total:.2f}%')
    model.train() # Set the model back to training mode


    # Set the model to evaluation mode
model.eval()

# Disable gradient calculation
with torch.no_grad():
    correct = 0
    total = 0
    # Iterate through the test loader
    for data in test_loader:
        # Get the images and labels
        images, labels = data
        images, labels = images.to(device), labels.to(device) # Move to device (GPU or CPU)
        # Pass the images through the model to get the outputs
        outputs = model(images)
        # Find the predicted class
        _, predicted = torch.max(outputs.data, 1)
        # Update the total count of images
        total += labels.size(0)
        # Update the count of correctly predicted images
        correct += (predicted == labels).sum().item()

# Calculate the final test accuracy
accuracy = 100 * correct / total

# Print the test accuracy
print(f"Test Accuracy: {accuracy:.2f}%")

# Select a few images from the test dataset
num_images_to_predict = 5
selected_images = []
true_labels = []

for i in range(num_images_to_predict):
    image, label = test_dataset[i]
    selected_images.append(image)
    true_labels.append(label)

selected_images = torch.stack(selected_images).to(device) # Stack the images and move to device
true_labels_tensor = torch.tensor(true_labels).to(device) # Convert to tensor and move to device

# Set the model to evaluation mode
model.eval()

# Disable gradient calculation
with torch.no_grad():
    # Pass the selected images through the model
    outputs = model(selected_images)

    # Determine the predicted class for each image
    _, predicted_labels = torch.max(outputs.data, 1)


# Print the predicted and true labels for comparison
print("Predictions vs True Labels:")
for i in range(num_images_to_predict):
    print(f"Image {i+1}: Predicted = {predicted_labels[i].item()}, True = {true_labels_tensor[i].item()}")