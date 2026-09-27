# =============================================================================
# PyTorch Tensors
# =============================================================================
# A Tensor is like a NumPy array but it can run on GPU.
# - 1D tensor = vector
# - 2D tensor = matrix
# - 3D+ tensor = multi-dimensional array
# PyTorch tensors also support automatic differentiation (autograd).
# =============================================================================

import torch
import numpy

print(torch.__version__)


# -----------------------------------------------------------------------------
# Creating Tensors
# torch.empty   -> uninitialized memory (random garbage values)
# torch.zeros   -> all zeros
# torch.ones    -> all ones
# torch.rand    -> random values between 0 and 1
# -----------------------------------------------------------------------------

x = torch.empty(2, 3)
print(x, type(x))

x = torch.zeros(2, 3)
print(x, type(x))

x = torch.ones(2, 3)
print(x, type(x))

x = torch.rand(2, 3)
print(x, type(x))

# manual_seed makes rand give same values every run (important for reproducibility)
torch.manual_seed(100)
x = torch.rand(2, 3)
print(x, type(x))

# torch.tensor -> create from a Python list directly
x = torch.tensor([[2, 3], [4, 5]])
print(x, type(x))

# torch.arange -> like Python range(), returns a 1D tensor
x = torch.arange(2, 10, 2)
print(x, type(x))

# torch.linspace -> n evenly spaced values between start and end
x = torch.linspace(0, 10, 10)
print(x, type(x))

# torch.eye -> identity matrix (1s on diagonal, 0s elsewhere)
x = torch.eye(4)
print(x, type(x))

# torch.full -> fill a matrix of given shape with one specific value
x = torch.full((2, 3), 5)
print(x, type(x))


# -----------------------------------------------------------------------------
# Creating Tensors with the Same Shape as Another (like_ functions)
# Useful when you want same size but different values
# -----------------------------------------------------------------------------

x = torch.tensor([[2, 3], [4, 5]])
print(x, type(x))

x = torch.empty_like(x)      # same shape, uninitialized
print(x, type(x))

x = torch.zeros_like(x)      # same shape, all zeros
print(x, type(x))

x = torch.ones_like(x)       # same shape, all ones
print(x, type(x))

x = torch.rand_like(x, dtype=torch.float)   # same shape, random floats
print(x, type(x))


# -----------------------------------------------------------------------------
# Tensor Datatypes
# Choosing the right dtype is important:
# - int32 for integer labels
# - float32 for model inputs/outputs (standard in deep learning)
# - float64 for high precision math
# -----------------------------------------------------------------------------

x = torch.tensor([1, 2, 3, 4], dtype=torch.int32)
print(x)

# .to() converts one dtype to another
x = torch.tensor([1, 2, 3, 4], dtype=torch.int32)
print(x.to(torch.float32))


# -----------------------------------------------------------------------------
# Mathematical Operations (Scalar Broadcasting)
# PyTorch applies the scalar to every element in the tensor
# -----------------------------------------------------------------------------

x = torch.rand(2, 2)
print(x)
print(x + 2)    # add 2 to every element
print(x - 2)    # subtract 2 from every element
print(x / 2)    # divide every element by 2
print(x * 2)    # multiply every element by 2
print(x ** 2)   # square every element


# -----------------------------------------------------------------------------
# Element-wise Operations
# Operations happen between matching positions of two tensors
# Both tensors must have the same shape
# -----------------------------------------------------------------------------

a = torch.tensor([6, 7])
b = torch.tensor([6, 9])

print(a + b)     # [12, 16]
print(a - b)     # [0, -2]
print(a / b)     # [1.0, 0.77]
print(a * b)     # [36, 63]
print(a ** b)    # [6^6, 7^9]

# Math functions applied element-wise
c = torch.tensor([-1, 3.1, -7, 8, -9])
print(torch.abs(c))           # absolute value
print(torch.neg(c))           # negate every value
print(torch.round(c))         # round to nearest integer
print(torch.ceil(c))          # round up
print(torch.floor(c))         # round down
print(torch.clamp(c, 2, 5))  # force all values into range [2, 5]


# -----------------------------------------------------------------------------
# Reduction Operations
# These reduce a tensor to a smaller size (e.g., sum all values)
# dim=0 means reduce along rows (collapse rows)
# dim=1 means reduce along columns (collapse columns)
# -----------------------------------------------------------------------------

v = torch.randint(size=(2, 3), low=0, high=10, dtype=torch.float32)
print(v)
print(torch.sum(v))           # sum of all elements
print(torch.sum(v, dim=1))    # sum of each row
print(torch.mean(v, dim=0))   # mean of each column
print(torch.median(v, dim=0)) # median of each column
print(torch.min(v, dim=0))    # min value in each column
print(torch.max(v, dim=0))    # max value in each column
print(torch.prod(v, dim=0))   # product of each column
print(torch.std(v, dim=0))    # standard deviation of each column
print(torch.var(v, dim=0))    # variance of each column
print(torch.argmax(v, dim=0)) # index of max value in each column
print(torch.argmin(v, dim=0)) # index of min value in each column


# -----------------------------------------------------------------------------
# Matrix Operations
# These are the core operations used in neural network forward passes
# matmul  -> dot product / matrix multiplication
# transpose -> flip rows and columns
# det     -> determinant (used in linear algebra)
# inverse -> matrix inverse (A^-1 such that A * A^-1 = I)
# -----------------------------------------------------------------------------

v = torch.randint(size=(3, 3), low=0, high=10, dtype=torch.float32)
b = torch.randint(size=(3, 3), low=0, high=10, dtype=torch.float32)

print(torch.matmul(v, b))       # matrix multiplication
print(torch.transpose(v, 0, 1)) # swap dim 0 and dim 1
print(torch.det(b))             # determinant
print(torch.inverse(b))         # inverse matrix


# -----------------------------------------------------------------------------
# Comparison Operations
# Return a boolean tensor (True/False) for each element
# Used for thresholding predictions, masking, filtering
# -----------------------------------------------------------------------------

print(v > b)   # True where v is greater
print(v < b)   # True where v is smaller
print(v == b)  # True where they are equal
print(v != b)  # True where they are not equal
print(v >= b)
print(v <= b)


# -----------------------------------------------------------------------------
# Special (Activation) Functions
# These are used as activation functions in neural networks
# sigmoid  -> squashes output between 0 and 1 (binary classification)
# softmax  -> squashes to probabilities that sum to 1 (multi-class)
# relu     -> max(0, x) — sets negatives to zero
# -----------------------------------------------------------------------------

q = torch.randint(size=(3, 3), low=0, high=10, dtype=torch.float32)
print(torch.log(q))             # natural log
print(torch.sqrt(q))            # square root
print(torch.sigmoid(q))         # sigmoid activation
print(torch.exp(q))             # exponential (e^x)
print(torch.softmax(q, dim=0))  # softmax along columns
print(torch.relu(q))            # ReLU activation


# -----------------------------------------------------------------------------
# Inplace Operations (trailing underscore _ means inplace)
# Inplace operations modify the tensor directly in memory
# This saves memory — no new tensor is created
# Warning: inplace ops can break autograd, use carefully in training
# -----------------------------------------------------------------------------

z = torch.rand(2, 3)
b = torch.rand(2, 3)
print(z)
print(b)
print(z.add_(b))    # z is permanently changed (z = z + b)
print(z.relu_(b))   # inplace relu


# -----------------------------------------------------------------------------
# Reshaping Tensors
# reshape  -> change shape, total elements must stay same
# flatten  -> collapse to 1D
# unsqueeze -> add a new dimension of size 1
#   Used when model expects batch dimension: (H,W,C) -> (1,H,W,C)
# -----------------------------------------------------------------------------

y = torch.rand(4, 4)
print(y)
print(y.reshape(2, 2, 2, 2))   # same 16 elements, new shape
print(y.flatten())              # all 16 elements in a single row

c = torch.rand(226, 226, 3)
print(c.unsqueeze(0).shape)     # adds batch dim: (1, 226, 226, 3)


# -----------------------------------------------------------------------------
# NumPy <-> PyTorch Conversion
# .numpy()        -> PyTorch tensor to NumPy array
# torch.from_numpy() -> NumPy array to PyTorch tensor
# Note: they share the same memory — changing one changes the other
# -----------------------------------------------------------------------------

a = torch.tensor([1, 2, 3])
b = a.numpy()       # tensor -> numpy
print(b)

d = numpy.array([5, 7])
g = torch.from_numpy(d)   # numpy -> tensor
print(g)
