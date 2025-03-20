import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

# Find min and max (returns dim 0 tensor)
x = torch.arange(0,100,10)
print(torch.min(x),torch.max(x))
print(x.min(), x.max())

# Find mean (needs float or complex)
print(torch.mean(x.type(torch.float32)))
print(x.type(torch.float32).mean())

# Find sum
print(torch.sum(x), x.sum())

# Positional min and max
print(x.argmin(), x.argmax())

# Video: 2:58:02
