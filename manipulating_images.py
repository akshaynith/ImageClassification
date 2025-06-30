import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

# Create a dummy image (a simple colored square for demonstration)
# In a real scenario, you would load an image using PIL or similar
dummy_image = Image.fromarray(np.uint8(np.random.randint(0, 255, (100, 100, 3))))

# Display the original image
plt.imshow(dummy_image)
plt.title("Original Image")
plt.axis('off')
plt.show()


# Define the ToTensor transform
to_tensor_transform = transforms.ToTensor()

# Apply the transform
tensor_image = to_tensor_transform(dummy_image)

# Print the shape and data type of the tensor
print("Shape after ToTensor:", tensor_image.shape)
print("Data type after ToTensor:", tensor_image.dtype)
print("Pixel values range:", tensor_image.min(), "-", tensor_image.max())


# Define the Normalize transform (using arbitrary mean and std for demonstration)
# In a real scenario, you'd use the mean and std of your dataset
normalize_transform = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

# Apply the transform to the tensor image
normalized_tensor_image = normalize_transform(tensor_image)

# Print the pixel values range after normalization
print("Pixel values range after Normalize:", normalized_tensor_image.min(), "-", normalized_tensor_image.max())

# Define the Resize transform
resize_transform = transforms.Resize((224, 224)) # Resize to 224x224 pixels

# Apply the transform to the original PIL image
resized_image = resize_transform(dummy_image)

# Display the resized image
plt.imshow(resized_image)
plt.title("Resized Image")
plt.axis('off')
plt.show()

# Define the RandomCrop transform
random_crop_transform = transforms.RandomCrop((50, 50)) # Crop to 50x50 pixels

# Apply the transform to the original PIL image
random_cropped_image = random_crop_transform(dummy_image)

# Display the random cropped image
plt.imshow(random_cropped_image)
plt.title("Random Cropped Image")
plt.axis('off')
plt.show()


# Define the RandomHorizontalFlip transform with a probability of 0.5
random_flip_transform = transforms.RandomHorizontalFlip(p=0.5)

# Apply the transform to the original PIL image
random_flipped_image = random_flip_transform(dummy_image)

# Display the random flipped image
plt.imshow(random_flipped_image)
plt.title("Random Horizontal Flip Image")
plt.axis('off')
plt.show()

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# Create a simple dummy image using PIL
# This creates a 100x150 red image with a blue square in the middle
img = Image.new('RGB', (150, 100), color = 'red')

# You can manipulate the image, for example, draw on it
# Let's put a blue square in the center
pixels = img.load()
for i in range(50, 100):
    for j in range(25, 75):
        pixels[i,j] = (0, 0, 255) # Blue color (R, G, B)

# Display some information about the PIL Image
print(f"Image mode: {img.mode}")
print(f"Image size: {img.size} (width, height)")

# Visualize the image
plt.imshow(img)
plt.title("Demonstration PIL Image")
plt.axis('off')
plt.show()