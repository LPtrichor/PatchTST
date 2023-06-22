import torch

if torch.cuda.is_available():
    print("CUDA is available! You have at least one GPU!")
else:
    print("CUDA is not available. CPU will be used for computations.")

