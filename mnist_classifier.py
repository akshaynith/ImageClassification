import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# 1. Define Transformations and Load Data
# Define a sequence of transformations to apply to the images.
# transforms.ToTensor() converts the image to a PyTorch tensor.
# transforms.Normalize() normalizes the tensor with a given mean and standard deviation.
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Load the MNIST training and test datasets.
# The data will be downloaded to the './data' directory if not already present.
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# Create DataLoaders for the training and test datasets.
# DataLoader provides an iterable over the given dataset, with options for batching, shuffling, etc.
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# 2. Define the Neural Network Model (CNN)
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # Convolutional Layer 1: 1 input channel (grayscale), 32 output channels, 3x3 kernel
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        # Convolutional Layer 2: 32 input channels, 64 output channels, 3x3 kernel
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        # Max Pooling Layer: 2x2 kernel, stride of 2
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # Fully Connected Layer 1: input features from flattened conv layers, 128 output features
        # Calculation: After conv1 (28-3+1=26), pool1 (26/2=13), conv2 (13-3+1=11), pool2 (11/2=5) -> 64x5x5
        self.fc1 = nn.Linear(64 * 5 * 5, 128)
        # Fully Connected Layer 2 (Output Layer): 128 input features, 10 output features (for digits 0-9)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        # Apply conv1, relu activation, and pooling
        x = self.pool(F.relu(self.conv1(x)))
        # Apply conv2, relu activation, and pooling
        x = self.pool(F.relu(self.conv2(x)))
        # Flatten the tensor for the fully connected layer
        x = x.view(-1, 64 * 5 * 5)
        # Apply fc1 and relu activation
        x = F.relu(self.fc1(x))
        # Apply fc2 to get the final output logits
        x = self.fc2(x)
        return x

model = Net()

# 3. Define Loss Function and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 4. Training Loop
epochs = 5
for epoch in range(epochs):
    model.train() # Set the model to training mode
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)
        # Calculate loss
        loss = criterion(outputs, labels)
        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 200 == 199:    # Print every 200 mini-batches
            print(f'Epoch [{epoch + 1}/{epochs}], Batch [{i + 1}/{len(train_loader)}], Loss: {running_loss / 200:.4f}')
            running_loss = 0.0

    # Optional: Evaluate on the test set after each epoch
    model.eval() # Set the model to evaluation mode
    correct = 0
    total = 0
    with torch.no_grad(): # Disable gradient calculation for evaluation
        for data in test_loader:
            images, labels = data
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Epoch [{epoch + 1}/{epochs}], Test Accuracy: {100 * correct / total:.2f}%')

print('Finished Training')

# 5. Save the trained model (optional)
torch.save(model.state_dict(), 'mnist_cnn_model.pth')
print('Model saved to mnist_cnn_model.pth')
