import torch
import sys

print(f"Python Version: {sys.version}")
print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA Version (usata da Torch): {torch.version.cuda}")
print(f"CUDA Disponibile: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"Nome GPU: {torch.cuda.get_device_name(0)}")
    print(f"Capability GPU: {torch.cuda.get_device_capability(0)}")
else:
    print("Nessuna GPU rilevata da PyTorch.")