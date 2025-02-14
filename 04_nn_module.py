# =============================================================================
# Training Pipeline with nn.Module
# =============================================================================
# This script rewrites the manual training pipeline from 03_training_pipeline.py
# using PyTorch's built-in tools. Four improvements:
#
#   1. nn.Module     -> cleaner model definition with automatic parameter tracking
#   2. nn.Sigmoid    -> built-in activation function (no manual sigmoid)
#   3. nn.BCELoss    -> built-in binary cross-entropy loss (no manual formula)
#   4. torch.optim   -> built-in optimizer (no manual weight update)
#
# This is the standard way to write PyTorch models in production.
# =============================================================================

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder


# =============================================================================
# PART 1 — Understanding nn.Module: Simple Model
# =============================================================================
# nn.Module is the base class for all neural networks in PyTorch.
# When you inherit from it, PyTorch automatically:
#   - tracks all parameters (weights and biases)
#   - lets you call model.parameters() for the optimizer
#   - handles saving and loading checkpoints
#   - manages train/eval modes


# class Model(nn.Module):
#     def __init__(self, num_features):
#         super().__init__()  # always call parent constructor first
#
#         # nn.Linear(in, out) creates a fully connected layer
#         # it automatically creates weights of shape (in, out) and bias of shape (out,)
#         self.linear = nn.Linear(num_features, 1)
#         self.sigmoid = nn.Sigmoid()
#
#     def forward(self, features):
#         out = self.linear(features)   # z = Xw + b (done internally)
#         out = self.sigmoid(out)       # probability = sigmoid(z)
#         print(out)
#         return out
#
#
# features = torch.rand(10, 5)  # 10 samples, 5 features each
# model = Model(features.shape[1])
# model(features)               # calling model() automatically calls forward()
# print(model.linear.weight)    # you can inspect parameters
# print(model.linear.bias)


# =============================================================================
# PART 2 — Adding Hidden Layers
# =============================================================================
# A single linear layer is just logistic regression.
# Adding hidden layers with activation functions creates a true neural network.
# ReLU(x) = max(0, x) — it adds non-linearity and helps learn complex patterns.


# class Model(nn.Module):
#     def __init__(self, num_features):
#         super().__init__()
#         self.linear1 = nn.Linear(num_features, 3)   # input -> hidden (3 neurons)
#         self.relu = nn.ReLU()
#         self.linear2 = nn.Linear(3, 1)              # hidden -> output (1 neuron)
#         self.sigmoid = nn.Sigmoid()
#
#     def forward(self, features):
#         out = self.linear1(features)   # first layer
#         out = self.relu(out)           # activation
#         out = self.linear2(out)        # second layer
#         out = self.sigmoid(out)        # final activation for binary output
#         print(out)
#         return out
#
# features = torch.rand(10, 5)
# model = Model(features.shape[1])
# model(features)
# print("weight of layer1", model.linear1.weight)
# print("bias of layer1", model.linear1.bias)
# print("weight of layer2", model.linear2.weight)
# print("bias of layer2", model.linear2.bias)


# =============================================================================
# PART 3 — Using nn.Sequential
# =============================================================================
# When forward() is just "pass through layers in order", nn.Sequential
# simplifies it. You define the stack once, and forward is automatic.
# No need to write each step manually in forward().


# class Model(nn.Module):
#     def __init__(self, num_features):
#         super().__init__()
#         self.network = nn.Sequential(
#             nn.Linear(num_features, 3),
#             nn.ReLU(),
#             nn.Linear(3, 1),
#             nn.Sigmoid()
#         )
#
#     def forward(self, features):
#         return self.network(features)   # one line replaces all forward logic
#
# features = torch.rand(10, 5)
# model = Model(features.shape[1])
# model(features)


# =============================================================================
# PART 4 — Full Training Pipeline with nn.Module
# (This is the active, runnable code)
# =============================================================================

# -----------------------------------------------------------------------------
# Load and Prepare Data (same as previous script)
# -----------------------------------------------------------------------------

df = pd.read_csv("data/breast-cancer.csv")
pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)
df.drop(columns=["id"], inplace=True)

x_train, x_test, y_train, y_test = train_test_split(df.iloc[:, 1:], df.iloc[:, 0], test_size=0.2)

scalar = StandardScaler()
x_train = scalar.fit_transform(x_train)
x_test = scalar.transform(x_test)

encoder = LabelEncoder()
y_train = encoder.fit_transform(y_train)
y_test = encoder.transform(y_test)

# float32 is needed here — nn.Linear uses float32 by default
x_train_tensor = torch.from_numpy(x_train.astype(np.float32))
x_test_tensor = torch.from_numpy(x_test.astype(np.float32))
y_train_tensor = torch.from_numpy(y_train.astype(np.float32))
y_test_tensor = torch.from_numpy(y_train.astype(np.float32))
print(x_train_tensor.shape)


# -----------------------------------------------------------------------------
# Model Definition using nn.Module
# Much cleaner than the manual version — no manual weight matrices,
# no manual sigmoid, no manual matmul. PyTorch handles all of it.
# -----------------------------------------------------------------------------

class MySimpleNN(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.linear = nn.Linear(num_features, 1)   # one neuron output
        self.sigmoid = nn.Sigmoid()                 # built-in sigmoid

    def forward(self, features):
        out = self.linear(features)    # z = Xw + b
        out = self.sigmoid(out)        # probability
        return out


# -----------------------------------------------------------------------------
# Loss Function and Optimizer
# nn.BCELoss: Binary Cross-Entropy Loss (standard for binary classification)
# torch.optim.SGD: Stochastic Gradient Descent
#   - reads model.parameters() so it knows which tensors to update
#   - replaces our manual weight -= lr * grad update from the previous script
# -----------------------------------------------------------------------------

learning_rate = 0.1
epochs = 35

loss_function = nn.BCELoss()
model = MySimpleNN(x_train_tensor.shape[1])
optimizer = torch.optim.SGD(model.parameters(), learning_rate)


# -----------------------------------------------------------------------------
# Training Loop (much cleaner with built-in tools)
# Key difference from manual loop:
#   - optimizer.zero_grad() replaces manual grad.zero_()
#   - optimizer.step()      replaces manual weight -= lr * grad
#   - loss_function()       replaces manual BCE formula
# -----------------------------------------------------------------------------

for epoch in range(epochs):
    # forward pass
    y_pred = model(x_train_tensor)

    # compute loss
    # y_train_tensor.view(-1, 1) reshapes labels to (N, 1) to match y_pred shape
    loss = loss_function(y_pred, y_train_tensor.view(-1, 1))

    # reset gradients from previous epoch (important — they accumulate by default)
    optimizer.zero_grad()

    # backward pass — compute gradients
    loss.backward()

    # update all parameters using computed gradients
    optimizer.step()

    print(f'epoch: {epoch + 1}, loss: {loss.item()}')


# print final learned weights and bias
print("weight of layer", model.linear.weight)
print("bias of layer", model.linear.bias)


# -----------------------------------------------------------------------------
# Evaluation
# -----------------------------------------------------------------------------

with torch.no_grad():
    y_pred = model.forward(x_test_tensor)
    y_pred = (y_pred > 0.5).float()
    accuracy = (y_pred == y_test_tensor).float().mean()
    print(f'Accuracy : {accuracy.item()}')
