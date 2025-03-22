import torch

# Set up device for MPS (Apple GPU) or fallback to CPU
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Running on MPS (Apple Silicon GPU)")
else:
    device = torch.device("cpu")
    print("Running on CPU")
