# Reshaping, stacking, squeezing, and unsqueezing

# Reshaping: reshapes tensor to defined shape
# View: return view of input tensor of a shape (same memory)
# Stacking: combine tensors on top of each other / side-by-side
# Squeeze: removes all '1' dimension from tensor
# Unsqueeze: add '1' dimension
# Permute: return view of input with dimensions permuted

import torch
x = torch.arange(1., 10.)
# print(x.shape)

x_reshape = x.reshape(3, 1, 1, 3) # note dimensions product must be the same
# print(x_reshape)
# print(x_reshape.shape)

# View
z = x.view(1,9) # NOTE changing z changes x!
# print(z)
# print(z.shape)

z[:, 0] = -1.
# print(z)
# print(x)

# Stacking tensors
x_stk = torch.stack([x, x, x, x], dim=0)
# print(x_stk)

# Squeezing: removes all single dimensions
q = torch.squeeze(x_reshape)
# print(q.shape)
y = q.unsqueeze(1)
# print(y.shape)

# permute
w = x_reshape.permute(3,0,1,2)
print(w.shape)

x = torch.rand(size=(224, 260, 3))
q = x.permute(2,0,1)
print(q.shape)
