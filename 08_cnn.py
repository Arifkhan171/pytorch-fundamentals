# =============================================================================
# CNN (Convolutional Neural Network) on Fashion MNIST
# =============================================================================
# ANN treats each pixel independently — it does not understand spatial patterns.
# CNN is designed for image data. It uses filters (kernels) to detect features
# like edges, curves, and textures, and it does this by looking at local regions.
#
# Key CNN operations used here:
#   Conv2d     -> applies filters to extract spatial features from image regions
#   BatchNorm2d -> normalizes feature maps after each conv layer (stabilizes training)
#   MaxPool2d  -> reduces spatial size by taking the max in each region (downsampling)
#   Dropout    -> randomly zeroes neurons to prevent overfitting
#   Flatten    -> converts 2D feature maps to 1D vector for the classifier head
#
# Architecture:
#   Input (1, 28, 28)
#   -> Conv2d(1, 32) + ReLU + BatchNorm + MaxPool -> (32, 14, 14)
#   -> Conv2d(32, 64) + ReLU + BatchNorm + MaxPool -> (64, 7, 7) = 3136 values
#   -> Flatten -> Dense(3136, 128) -> ReLU -> Dropout
#   -> Dense(128, 64) -> ReLU -> Dropout -> Dense(64, 10)
# =============================================================================

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


torch.manual_seed(42)
df = pd.read_csv("data/fmnist_small.csv")


# -----------------------------------------------------------------------------
# Visualization (optional)
# Uncomment to see a 4x4 grid of sample images
# -----------------------------------------------------------------------------

# fig, axes = plt.subplots(4, 4, figsize=(10, 10))
# fig.suptitle("First 16 images", fontsize=16)
# for i, ax in enumerate(axes.flat):
#     img = df.iloc[i, 1:].values.reshape(28, 28)
#     ax.imshow(img)
#     ax.axis("off")
#     ax.set_title(f"Label: {df.iloc[i, 0]}")
# plt.tight_layout(rect=[0, 0, 1, 0.96])
# plt.show()


# -----------------------------------------------------------------------------
# Load and Normalize Data
# -----------------------------------------------------------------------------

x = df.iloc[:, 1:].values
y = df.iloc[:, 0].values

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

x_train = x_train / 225.0
x_test = x_test / 225.0


# -----------------------------------------------------------------------------
# Custom Dataset with 2D Reshape for CNN
# Critical difference from ANN: CNN expects 4D input: (batch, channels, H, W)
#   batch   -> number of samples in a batch (handled by DataLoader)
#   channels -> 1 for grayscale, 3 for RGB
#   H, W    -> height and width of the image (28 x 28 here)
#
# We reshape from flat (784,) to (1, 28, 28) using .reshape(-1, 1, 28, 28)
#   -1 means "infer this dimension from total size" (becomes batch size)
# ANN used flat 784 input, CNN needs 2D spatial structure.
# -----------------------------------------------------------------------------

class CustomDataset(Dataset):
    def __init__(self, features, labels):
        # reshape each image from flat 784 -> (1, 28, 28) for CNN
        self.features = torch.tensor(features, dtype=torch.float32).reshape(-1, 1, 28, 28)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        return self.features[index], self.labels[index]


train_dataset = CustomDataset(x_train, y_train)
test_dataset = CustomDataset(x_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


# -----------------------------------------------------------------------------
# CNN Architecture
#
# Feature Extractor (self.features):
#   Conv2d(1, 32, kernel_size=3, padding="same")
#     -> 1 input channel (grayscale), 32 filters, 3x3 kernel
#     -> padding="same" keeps output size same as input (28x28)
#     -> each filter learns to detect a different feature (edges, curves, etc.)
#   BatchNorm2d(32)   -> normalizes 32 feature maps after conv
#   MaxPool2d(2, 2)   -> takes max in each 2x2 window, halves spatial size
#                     -> output becomes (32, 14, 14)
#
#   Conv2d(32, 64, kernel_size=3, padding="same")
#     -> 32 input channels, 64 output filters
#     -> learns more complex combinations of features from first layer
#   MaxPool2d(2, 2)   -> halves again: (64, 7, 7) = 3136 total values
#
# Classifier Head (self.classifier):
#   Flatten()         -> converts (64, 7, 7) to (3136,) vector
#   Linear(3136, 128) -> dense layer
#   Dropout(0.4)      -> drop 40% of neurons to prevent overfitting
#   Linear(128, 64)   -> dense layer
#   Dropout(0.4)      -> drop 40% again
#   Linear(64, 10)    -> 10 class scores
# -----------------------------------------------------------------------------

class MyNN(nn.Module):
    def __init__(self, input_features):
        super().__init__()

        # convolutional backbone: extracts spatial features from images
        self.features = nn.Sequential(
            nn.Conv2d(input_features, 32, kernel_size=3, padding="same"),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=2, stride=2),   # 28x28 -> 14x14

            nn.Conv2d(32, 64, kernel_size=3, padding="same"),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2, stride=2)    # 14x14 -> 7x7
        )

        # fully connected classifier: maps features to class predictions
        self.classifier = nn.Sequential(
            nn.Flatten(),                # (64, 7, 7) -> (3136,)
            nn.Linear(3136, 128),
            nn.ReLU(),
            nn.Dropout(p=0.4),

            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(p=0.4),

            nn.Linear(64, 10)           # 10 class output logits
        )

    def forward(self, x):
        x = self.features(x)       # extract spatial features
        x = self.classifier(x)     # classify based on features
        return x


# -----------------------------------------------------------------------------
# Training Setup
# Same as before: CrossEntropyLoss + SGD with weight_decay for L2 regularization
# model is instantiated with input_features=1 (1 channel for grayscale images)
# -----------------------------------------------------------------------------

epochs = 100
learning_rate = 0.1

model = MyNN(1)   # 1 = number of input channels (grayscale)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate, weight_decay=1e-4)


# -----------------------------------------------------------------------------
# Training Loop
# ave_epoch_loss is printed to monitor convergence
# If loss does not decrease over epochs, learning rate may be too high or too low
# -----------------------------------------------------------------------------

for epoch in range(epochs):
    total_epoch_loss = 0
    for batch_features, batch_labels in train_loader:
        outputs = model(batch_features)
        loss = criterion(outputs, batch_labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_epoch_loss = total_epoch_loss + loss.item()

    ave_epoch_loss = total_epoch_loss / len(train_loader)
    print(f"Epochs: {epoch + 1}, loss: {ave_epoch_loss},")


# -----------------------------------------------------------------------------
# Evaluation on Test Data
# -----------------------------------------------------------------------------

model.eval()   # disables dropout and batchnorm randomness

total = 0
correct = 0
with torch.no_grad():
    for batch_features, batch_labels in test_loader:
        output = model(batch_features)
        _, predicted = torch.max(output, 1)
        total = total + batch_labels.shape[0]
        correct = correct + (predicted == batch_labels).sum().item()

print("Accuracy score for test data is ", correct / total)


# -----------------------------------------------------------------------------
# Evaluation on Training Data
# Compare with test accuracy to see if CNN reduced overfitting vs ANN
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

print("Accuracy score for training data is ", correct / total)

# =============================================================================
# Results:
#   ANN  test accuracy:  82.25%
#   CNN  test accuracy:  87.25%  (+5% improvement using spatial features)
#
# CNN outperforms ANN on image data because:
#   - Filters learn local patterns (edges, shapes) that ANNs miss
#   - Parameter sharing: same filter applied across the whole image
#   - Spatial hierarchy: early layers detect edges, deeper layers detect shapes
# =============================================================================
