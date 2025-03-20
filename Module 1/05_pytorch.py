# torch.from_numpy(array) : numpy to pytorch
# torch.tensor().numpy() : pytorch to numpy

import torch
import numpy as np

# accessing: tensor[:, 1, :], ':' means all of the dimension

array = np.arange(1.,8.)
tensor = torch.from_numpy(array).type(torch.float32)
# print(tensor)
# print(array)

tensor1 = torch.ones(7)
nummy = tensor1.numpy()
# print(nummy + 1)

# Reproducibility (random out of random)
# predictable random... -> random seed
rand_tens = torch.rand(3, 4)
rand_tens2 = torch.rand(3, 4)
# print(rand_tens)
# print(rand_tens2)
# print(rand_tens == rand_tens2) # makes a tensor of boolean

# 'random'
RANDOM_SD = 42
torch.manual_seed(RANDOM_SD)
rand3 = torch.rand(3, 4)

torch.manual_seed(RANDOM_SD)
rand4 = torch.rand(3, 4)
print(rand3)
print(rand4)
print(rand3 == rand4)
