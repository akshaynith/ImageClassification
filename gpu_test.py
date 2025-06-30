import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Default device:", device)

x = torch.tensor([1.0], device=device)
print("x device:", x.device)

print("GPU Name:", torch.cuda.get_device_name(0))