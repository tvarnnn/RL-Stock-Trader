import torch

# Check PyTorch version
print("Torch version:", torch.__version__)

# Check CUDA version PyTorch was built with
print("CUDA version:", torch.version.cuda)

# Check if CUDA (GPU) is available
print("CUDA available:", torch.cuda.is_available())

# Check number of GPUs detected
print("Number of GPUs:", torch.cuda.device_count())

# Check the name of the current GPU
if torch.cuda.is_available():
    print("Current GPU:", torch.cuda.get_device_name(torch.cuda.current_device()))
