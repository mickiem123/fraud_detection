import os


from src.utils import setup_logger
from src.components.data_ingestion import DataIngestorFactory

print(os.getcwd())
import torch
# print("CUDA available:", torch.cuda.is_available())
# print("CUDA device count:", torch.cuda.device_count())
# if torch.cuda.is_available():
#     print("CUDA current device:", torch.cuda.current_device())
#     print("CUDA device name:", torch.cuda.get_device_name(torch.cuda.current_device()))
#     print("CUDA device capability:", torch.cuda.get_device_capability(torch.cuda.current_device()))
#     print("CUDA memory allocated (MB):", torch.cuda.memory_allocated() / 1024 ** 2)
#     print("CUDA memory cached (MB):", torch.cuda.memory_reserved() / 1024 ** 2)
#     print("CUDA version:", torch.version.cuda)
#     print("cuDNN version:", torch.backends.cudnn.version())
# else:
#     print("CUDA is not available on this system.")

x = torch.tensor([1.0, 2.0, 3.0])  # 1D tensor
x_gpu = x.to("cuda")  # Move to GPU
print(x_gpu)  # Runs on RTX 3050 if CUDA is available