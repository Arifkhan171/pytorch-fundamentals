# =============================================================================
# ANN (Artificial Neural Network) on Fashion MNIST
# =============================================================================
# Now we work on a real image classification problem.
# Fashion MNIST has 10 classes: T-shirt, Trouser, Pullover, Dress, Coat,
# Sandal, Shirt, Sneaker, Bag, Ankle boot.
# Each image is 28x28 pixels = 784 input features (when flattened).
#
# Architecture:
#   Input (784) -> Dense (128) -> ReLU -> Dense (64) -> ReLU -> Output (10)
#
# This is a multi-class problem, so we use CrossEntropyLoss (not BCELoss).
# CrossEntropyLoss = Softmax + Negative Log Likelihood Loss combined.
# =============================================================================

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


# set seed so results are reproducible every run
torch.manual_seed(42)

# -----------------------------------------------------------------------------
# Load Fashion MNIST CSV
# Each row: first column = label (0-9), remaining 784 columns = pixel values
# Pixel values are 0-255 (grayscale intensity)
# Note: download from Kaggle and place in data/fmnist_small.csv
# -----------------------------------------------------------------------------

df = pd.read_csv("data/fmnist_small.csv")


# -----------------------------------------------------------------------------
# Visualization (optional — uncomment to see sample images)
# Creates a 4x4 grid showing the first 16 images with their labels
# Useful for sanity checking: confirm images and labels are loaded correctly
# -----------------------------------------------------------------------------

# fig, axes = plt.subplots(4, 4, figsize=(10, 10))
# fig.suptitle("First 16 images", fontsize=16)
# for i, ax in enumerate(axes.flat):
#     img = df.iloc[i, 1:].values.reshape(28, 28)  # flatten row back to 2D image
#     ax.imshow(img)
#     ax.axis("off")
#     ax.set_title(f"Label: {df.iloc[i, 0]}")
# plt.tight_layout(rect=[0, 0, 1, 0.96])
# plt.show()


# -----------------------------------------------------------------------------
# Prepare Features and Labels
# x: all columns after first = 784 pixel values per image
# y: first column = class label (0-9)
# Normalize pixels: divide by 225 to get values between 0 and 1
#   This helps gradient descent converge faster and more stably
# -----------------------------------------------------------------------------

x = df.iloc[:, 1:].values    # pixel values (NumPy array)
y = df.iloc[:, 0].values     # labels

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# normalize pixel values to [0, 1]
x_train = x_train / 225.0
x_test = x_test / 225.0


# -----------------------------------------------------------------------------
# Custom Dataset Class
# Converts NumPy arrays to tensors inside __init__ (done once at startup)
# Features: float32 (model weights are float32 by default)
# Labels:   long (int64) — required by CrossEntropyLoss
# -----------------------------------------------------------------------------

class CustomDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        return self.features[index], self.labels[index]


train_dataset = CustomDataset(x_train, y_train)
test_dataset = CustomDataset(x_test, y_test)

# shuffle=True on train, False on test (order doesn't affect test evaluation)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


# -----------------------------------------------------------------------------
# ANN Architecture
# Three fully connected layers with ReLU activations
# No Softmax at the end — CrossEntropyLoss applies it internally
# Layer sizes: 784 -> 128 -> 64 -> 10
#   128 and 64 are hidden neurons (hyperparameters, tuned by experimentation)
#   10 outputs = one score per class (logits, not probabilities)
# -----------------------------------------------------------------------------

class MyNN(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(num_features, 128),   # 784 inputs -> 128 hidden
            nn.ReLU(),                       # non-linearity

            nn.Linear(128, 64),             # 128 -> 64 hidden
            nn.ReLU(),                       # non-linearity

            nn.Linear(64, 10)               # 64 -> 10 class scores (logits)
        )

    def forward(self, x):
        return self.model(x)


# -----------------------------------------------------------------------------
# Training Setup
# CrossEntropyLoss: expects raw logits (not softmax output) and integer labels
# SGD with fixed learning rate — simple and effective for this problem
# 100 epochs to let the model converge
# -----------------------------------------------------------------------------

epochs = 100
learning_rate = 0.1

model = MyNN(x_train.shape[1])   # x_train.shape[1] = 784 features
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate)


# -----------------------------------------------------------------------------
# Training Loop
# total_epoch_loss accumulates batch losses within an epoch
# ave_epoch_loss divides by number of batches -> average loss per epoch
# We track per-epoch average to monitor how training is progressing
# -----------------------------------------------------------------------------

for epoch in range(epochs):
    total_epoch_loss = 0
    for batch_features, batch_labels in train_loader:
        # forward pass: get raw scores (logits) for each class
        outputs = model(batch_features)

        # compute loss: CrossEntropyLoss applies softmax and then NLL internally
        loss = criterion(outputs, batch_labels)

        # backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_epoch_loss = total_epoch_loss + loss.item()

    ave_epoch_loss = total_epoch_loss / len(train_loader)
    # print(f"Epochs: {epoch + 1}, loss: {ave_epoch_loss},")


# -----------------------------------------------------------------------------
# Evaluation on Test Data
# torch.max(output, 1) returns (max_value, index_of_max)
# The index of the highest logit = predicted class
# We compare predicted class to true label and count correct predictions
# -----------------------------------------------------------------------------

model.eval()

total = 0
correct = 0
with torch.no_grad():
    for batch_features, batch_labels in test_loader:
        output = model(batch_features)
        _, predicted = torch.max(output, 1)   # get index of highest score
        total = total + batch_labels.shape[0]
        correct = correct + (predicted == batch_labels).sum().item()

print("Accuracy score for test data is ", correct / total)   # Result: 0.8225


# -----------------------------------------------------------------------------
# Evaluation on Training Data
# Comparing train vs test accuracy tells us if the model is overfitting.
# If train accuracy >> test accuracy -> model memorized training data (overfitting)
# -----------------------------------------------------------------------------

model.eval()

total = 0
correct = 0
with torch.no_grad():
    for batch_features, batch_labels in train_loader:
        output = model(batch_features)
        _, predicted = torch.max(output, 1)
        total = total + batch_labels.shape[0]
        correct = correct + (predicted == batch_labels).sum().item()

print("Accuracy score for training data is ", correct / total)   # Result: 0.9975

# =============================================================================
# Conclusion:
#   Training accuracy: 99.75%
#   Test accuracy:     82.25%
#   Gap of ~17%  ->  model is clearly overfitting
#   Solution: add Dropout and Regularization (see 07_ann_with_regularization.py)
# =============================================================================
