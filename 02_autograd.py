# =============================================================================
# PyTorch Autograd (Automatic Differentiation)
# =============================================================================
# Autograd is PyTorch's automatic system for computing derivatives.
# It tracks every operation done on a tensor and builds a computation graph.
# When you call .backward(), it walks that graph in reverse and computes
# the gradient (derivative) for every tensor that has requires_grad=True.
#
# This is the engine behind training neural networks:
#   - forward pass  -> compute predictions and loss
#   - backward pass -> compute gradients via autograd
#   - update weights -> use gradients to reduce loss
# =============================================================================

import torch
from torch.nn.functional import binary_cross_entropy


# -----------------------------------------------------------------------------
# Basic Gradient: dy/dx where y = x^2
# If y = x^2, then dy/dx = 2x
# At x = 3, gradient should be 2 * 3 = 6
# requires_grad=True tells PyTorch to track this tensor for autograd
# -----------------------------------------------------------------------------

x = torch.tensor(3.0, requires_grad=True)
y = x ** 2
print(y)         # tensor(9., grad_fn=<PowBackward0>)
y.backward()     # compute gradient
print(x.grad)    # should print 6.0  (dy/dx = 2x = 2*3 = 6)


# -----------------------------------------------------------------------------
# Chained Operations: gradient flows through multiple functions
# y = x^2, z = sin(y)
# By chain rule: dz/dx = dz/dy * dy/dx = cos(y) * 2x
# PyTorch handles chain rule automatically
# -----------------------------------------------------------------------------

x = torch.tensor(3.0, requires_grad=True)
y = x ** 2
z = torch.sin(y)   # chain another operation
z.backward()       # gradient flows backwards through sin and then x^2
print(x.grad)      # PyTorch computes this automatically using chain rule
print("")


# -----------------------------------------------------------------------------
# Loss Function Gradient Calculation
# In binary classification, we use sigmoid to get probability,
# then binary cross-entropy to measure how wrong our prediction is.
# Autograd computes gradients of loss with respect to weights and bias.
# These gradients tell us in which direction to adjust w and b.
# -----------------------------------------------------------------------------

x = torch.tensor(6.7)           # input feature
y = torch.tensor(0.0)           # true label (0 or 1)
w = torch.tensor(1.0, requires_grad=True)   # weight (we want grad for this)
b = torch.tensor(0.0, requires_grad=True)   # bias  (we want grad for this)

z = w * x + b                   # linear combination: z = wx + b
y_pred = torch.sigmoid(z)       # sigmoid converts z to probability [0, 1]
loss = binary_cross_entropy(y_pred, y)   # how wrong is our prediction?

loss.backward()                  # compute gradients of loss w.r.t w and b
print(w.grad)    # gradient of loss with respect to w -> how to change w
print(b.grad)    # gradient of loss with respect to b -> how to change b
print("")


# -----------------------------------------------------------------------------
# Gradient on a Vector (not just a scalar)
# When input is a vector, we get a gradient vector of the same shape.
# Each value tells us: if I change this input slightly, how does y change?
# y = mean(x^2), so dy/dx_i = 2 * x_i / n
# For x = [2, 3, 4] with n=3: grads = [4/3, 6/3, 8/3] = [1.33, 2.0, 2.67]
# -----------------------------------------------------------------------------

x = torch.tensor([2.0, 3.0, 4.0], requires_grad=True)
y = (x ** 2).mean()   # .mean() makes it a scalar so backward() works
y.backward()
print(x.grad)          # [1.33, 2.0, 2.67] -> gradient for each element
