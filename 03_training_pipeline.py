# =============================================================================
# Manual Training Pipeline from Scratch
# =============================================================================
# In this script we build a logistic regression model by hand using raw PyTorch.
# We do NOT use nn.Module, nn.Linear, or any optimizer — everything is manual.
# This helps you understand what happens inside PyTorch at the lowest level.
#
# Dataset: Breast Cancer Wisconsin (binary classification)
#   - Features: 30 numeric measurements from tumor cells
#   - Target: Malignant (1) or Benign (0)
#
# Training Steps each epoch:
#   1. Forward pass    -> compute predictions
#   2. Loss            -> measure how wrong we are
#   3. Backward        -> compute gradients via autograd
#   4. Weight update   -> adjust weights using gradient descent
#   5. Zero gradient   -> reset gradients before next epoch
# =============================================================================

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import torch


# -----------------------------------------------------------------------------
# Load and Prepare Data
# StandardScaler: makes all features have mean=0 and std=1
#   This is important — without scaling, features with big values
#   dominate gradient updates and training becomes unstable
# LabelEncoder: converts string labels like 'M'/'B' into 0 and 1
# -----------------------------------------------------------------------------

df = pd.read_csv("data/breast-cancer.csv")
pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)
df.drop(columns=["id"], inplace=True)   # id column is not a feature, drop it

# split: first column is the label, rest are features
x_train, x_test, y_train, y_test = train_test_split(df.iloc[:, 1:], df.iloc[:, 0], test_size=0.2)

# fit on train, transform both — never fit on test data (that would be data leakage)
scalar = StandardScaler()
x_train = scalar.fit_transform(x_train)
x_test = scalar.transform(x_test)

# encode string labels to integers (M -> 1, B -> 0 or vice versa)
encoder = LabelEncoder()
y_train = encoder.fit_transform(y_train)
y_test = encoder.transform(y_test)

# convert NumPy arrays to PyTorch tensors — model only works with tensors
x_train_tensor = torch.from_numpy(x_train)
x_test_tensor = torch.from_numpy(x_test)
y_train_tensor = torch.from_numpy(y_train)
y_test_tensor = torch.from_numpy(y_train)   # note: used for evaluation shape
print(x_train_tensor.shape)


# -----------------------------------------------------------------------------
# Manual Neural Network Class (no nn.Module)
# We manually define:
#   - weights (w): one per feature, randomly initialized
#   - bias (b): starts at zero
#   - forward(): matrix multiply inputs with weights, then sigmoid
#   - loss_function(): binary cross-entropy calculated from scratch
# requires_grad=True tells PyTorch to track gradients for w and b
# -----------------------------------------------------------------------------

class MySimpleNN():
    def __init__(self, x):
        # weight shape: (num_features, 1) for one output neuron
        self.weight = torch.rand(x.shape[1], 1, dtype=torch.float64, requires_grad=True)
        self.bias = torch.zeros(1, dtype=torch.float64, requires_grad=True)

    def forward(self, x):
        # z = Xw + b  (linear transformation)
        z = torch.matmul(x, self.weight) + self.bias
        # sigmoid squashes z to [0,1] which we treat as probability
        y_pred = torch.sigmoid(z)
        return y_pred

    def loss_function(self, y_pred, y):
        # epsilon prevents log(0) which is -infinity
        epsilon = 1e-7
        y_pred = torch.clamp(y_pred, epsilon, 1 - epsilon)
        # binary cross-entropy formula: -[y*log(p) + (1-y)*log(1-p)]
        loss = -(y_train_tensor * torch.log(y_pred) + (1 - y_train_tensor) * torch.log(1 - y_pred)).mean()
        return loss


# -----------------------------------------------------------------------------
# Training Loop (Gradient Descent from Scratch)
# Each epoch:
#   - We feed ALL training data at once (not mini-batches)
#   - Compute loss
#   - Autograd fills in .grad for weight and bias
#   - We subtract gradient * learning_rate from weights
#   - torch.no_grad() prevents this update step from being tracked by autograd
#   - .zero_() resets gradients to 0 so they don't accumulate next epoch
# -----------------------------------------------------------------------------

learning_rate = 0.1
epochs = 35

model = MySimpleNN(x_train_tensor)

for epoch in range(epochs):
    # step 1: forward pass
    y_pred = model.forward(x_train_tensor)

    # step 2: compute loss
    loss = model.loss_function(y_pred, y_train_tensor)

    # step 3: backward pass — autograd fills model.weight.grad and model.bias.grad
    loss.backward()

    # step 4: update weights (inside no_grad so update is not tracked)
    with torch.no_grad():
        model.weight -= learning_rate * model.weight.grad
        model.bias -= learning_rate * model.bias.grad

    # step 5: reset gradients to zero before next epoch
    model.weight.grad.zero_()
    model.bias.grad.zero_()

    print(f'epoch: {epoch + 1}, loss: {loss.item()}')


# -----------------------------------------------------------------------------
# Model Evaluation
# torch.no_grad() disables autograd during evaluation
# This makes inference faster and saves memory — we don't need gradients here
# Threshold 0.5: if predicted probability > 0.5 -> class 1, else class 0
# -----------------------------------------------------------------------------

with torch.no_grad():
    y_pred = model.forward(x_test_tensor)
    y_pred = (y_pred > 0.5).float()   # convert probabilities to class labels
    accuracy = (y_pred == y_test_tensor).float().mean()
    print(f'Accuracy : {accuracy.item()}')
