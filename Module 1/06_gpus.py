import numpy as np
import torch

# print(torch.cuda.is_available())
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)
print(torch.cuda.device_count())

# print(tensor, tensor.device)
# tensor = tensor.to(device) : move tensor to another device
# move to cpu for numpy: tensor = tensor.cpu().numpy()