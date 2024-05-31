import torch

# Check if CUDA is available
if torch.cuda.is_available():
    print(f"Number of CUDA devices: {torch.cuda.device_count()}")
    print(f"Current CUDA device: {torch.cuda.current_device()}")
    device_name = torch.cuda.get_device_name(0)
    print(f"Device {0} name: {device_name}")
else:
    print("CUDA is not available.")