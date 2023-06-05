import torch

print("CUDA available:", torch.cuda.is_available())
print("GPU(s) detected:", torch.cuda.device_count())
print(torch.__version__)

