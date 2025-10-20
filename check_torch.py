# check_torch.py
import sys, torch
print("Python exe:", sys.executable)
print("Torch version:", torch.__version__)
print("Torch built with CUDA:", torch.version.cuda)   # None means CPU-only build
print("CUDA available:", torch.cuda.is_available())
print("CUDA compiled in this build:", torch.backends.cuda.is_built())
