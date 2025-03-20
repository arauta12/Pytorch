import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

## Tensors

# scalar
print("Scalar Data:")
scalar = torch.tensor(7)
print(scalar)
print(scalar.ndim)  # dimensions
print(scalar.item())  # tensor as int

# Vector
print("\nVector Data:")
vector = torch.tensor([7,7]) # 2D input, 1D output
print(vector.ndim)  # dimensions
print(vector.shape)

#  Matrix
print("\nMatrix Data:")
MATRIX = torch.tensor([[7,2], # 1D input, 3D output
                      [9,2]])
print(MATRIX.ndim)
print(MATRIX.shape)
print(MATRIX[0])

# Tensor (manually)
print("\nTensor Data:")
TENSOR = torch.tensor([[[1,2,3],
                        [1,1,1],
                        [4,5,6]]])
print(TENSOR.ndim)
print(TENSOR.shape)
print(TENSOR[0])

# Random Tensors : start with tensor of random numbers then adjust
print("\nRandom Tensor Data:")
random_tensor = torch.rand(1,2,3,4) # 1st: num of lists, 2nd: num entries in each list
print(random_tensor.ndim)
print(random_tensor)

# Image tensor
print("\nImage Tensor Data:")
random_image_tensor = torch.rand(size=(3,4)) # height, width, color_channel
print(random_image_tensor.shape)
print(random_image_tensor)

# Binary (0/1) tensor
print("\nBinary Tensor Data:")
zero = torch.zeros(size=(3,4))
ones = torch.ones(size=(3,4))
print(zero.ndim)
print(zero.shape)
print(ones * random_image_tensor) # tensor multiplication by entry

# Range of tensors and tensors-like
print("\nTensor Range:")
rnge = torch.arange(0, 11) # torch.range but is: [start, end)
stp = torch.arange(0, 1000, 25) # [start,end) w/ step
print(stp)

print("\nTensor Like:")
ten_zeros = torch.zeros_like(input=rnge) # same shape as input
print(ten_zeros)
