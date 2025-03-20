"""Tensor Datatypes"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from sympy.codegen.ast import float16

# Float 32
float_tensor = torch.tensor([3.0,6.0,9.0], dtype=None)
# print(float_tensor)
# print(float_tensor.dtype) # default float 32

# Note: multiplying tensors need the same shape, dtype & device
float_tensor2 = torch.tensor(
    [3.0,6.0,9.0],
    dtype=None, # specify datatype of tensor
    device=None, # what device the tensor is one
    requires_grad=False) # track gradient?

# Change tensor dtype
float_16 = float_tensor.type(torch.float16) # half = float16
# print(float_16)

# Note not all different dtypes will error, but some will
int_32_tensor = torch.tensor([3,6,8], dtype=torch.int32)
int_64_tensor = int_32_tensor.type(torch.int64)
long_tensor = int_32_tensor.type(torch.long)
float_32_tensor  = torch.tensor([3,6,9], dtype=torch.float32)
# print(int_32_tensor * float_32_tensor)
# print(int_64_tensor * float_32_tensor)
# print(float_32_tensor * long_tensor)

# Get info from Tensors
TENSOR = torch.tensor([4,5,6])
# dtype?
print(TENSOR.dtype)
# shape?
print(TENSOR.shape)
# device?
print(TENSOR.device)

print(f"Dtype: {TENSOR.dtype}, shape: {TENSOR.shape}, device: {TENSOR.device}")
