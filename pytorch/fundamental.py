import torch

# Create a tensor
#some_tensor = torch.rand(3, 4)

# Find out details about it
#print(some_tensor)
#print(f"Shape of tensor: {some_tensor.shape}")
#print(f"Datatype of tensor: {some_tensor.dtype}")
#print(f"Device tensor is stored on: {some_tensor.device}") # will default to CPU

# Create a tensor of values and add a number to it
#tensor = torch.tensor([1, 2, 3])
#tensor + 10
# Multiply it by 10
#tensor * 10

# Element-wise multiplication (each element multiplies its equivalent, index 0->0, 1->1, 2->2)
#print(tensor, "*", tensor)
#print("Equals:", tensor * tensor)

# Create a tensor
x = torch.arange(10, 100, 10)
x
print(f"Minimum: {x.min()}")
print(f"Maximum: {x.max()}")
# print(f"Mean: {x.mean()}") # this will error
print(f"Mean: {x.type(torch.float32).mean()}") # won't work without float datatype
print(f"Sum: {x.sum()}")

# Returns index of max and min values
print(f"Index where max value occurs: {x.argmax()}")
print(f"Index where min value occurs: {x.argmin()}")

import torch 

# Create random tensor
X = torch.rand(size=(7, 7))
X, X.shape
print(X)