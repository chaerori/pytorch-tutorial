import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# create a tensor from data(data can be scalar, list, tuple, ndarray and other types)
x = torch.tensor([[1,2,3],[4,5,6]], dtype=torch.float, device=torch.device('cpu'), requires_grad=True)

# create a tensor filled with 0 value
zero_tensor = torch.zeros(2,3)

# create a tensor filled with 1 value
one_tensor = torch.ones(2,3)

# create a tensor filled with fill_value
fill_tensor = torch.full((2,3), 3.14)

# create a tensor filled with RV from normal distribution with mean 0 and variance 1
normal_tensor = torch.randn(2,3)

# create a tensor filled with RV from uniform distribution on interval [0, 1)
uniform_tensor = torch.rand(2,3)

vector = x[1]
slicing = x[0][1:2]

# tensor operation
print(x + one_tensor)
print(x * zero_tensor)
print(10 * normal_tensor) # scalar values are computed element wise
print(torch.mv(x, vector))