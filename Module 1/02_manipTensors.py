import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

"""
Operations: addition, subtraction, multiplication, division, matrix multiplication
"""

T = torch.tensor([1,2,3])
# print(T + 10) # entry-wise
# print(T - 10) # entry-wise
# print(T * 10) # entry-wise

# print(torch.mul(T,10))
# print(torch.add(T,10))

# Multiplication of Tensors
T2 = torch.tensor([4,5,6])
print(T * T2)

print(torch.matmul(T, T2))
# print(T @ T2)

# Rules for tensor multiplication (not by entry)
# 1) inner dim. must match
print(torch.matmul(torch.rand(3,2), torch.rand(2,3)))
# 2) resulting tensor has shape of outer dim.

# Handling shape errors
tA = torch.tensor([[1,2],
                   [3,4],
                   [5,6]])

tB = torch.tensor([[7,10],
                   [8,11],
                   [9,12]])
# print(torch.mm(tA, tB))
print(tB.T) # transpose
print(torch.mm(tA, tB.T))
