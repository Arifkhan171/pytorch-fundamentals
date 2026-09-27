# =============================================================================
# ANN with Overfitting Reduction Techniques
# =============================================================================
# Problem from previous script:
#   Training accuracy 99.75%, Test accuracy 82.25% -> big overfitting gap
#
# Overfitting means the model memorized training data but cannot generalize.
# Methods to reduce overfitting (all 7 exist, we apply 3 here):
#
#   1. Adding more data            -> not done here
#   2. Simpler architecture        -> not done here
#   3. L2 Regularization           -> weight_decay in optimizer (done)
#   4. Dropout                     -> nn.Dropout layers added (done)
#   5. Batch Normalization         -> nn.BatchNorm1d (commented — made it worse here)
#   6. Early Stopping              -> not done here
#   7. Data Augmentation           -> not done here
#
# Key concepts applied:
#   Dropout      -> randomly zeros out neurons during training, forces the network
#                   to not rely on any single neuron, learns more robust features
#   L2 (weight_decay) -> penalizes large weights in the loss function,
#                        pushes weights toward zero, prevents extreme fitting
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
# Uncomment to see a 4x4 grid of the first 16 images
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
# Load and Normalize Data (same as 06_ann.py)
# -----------------------------------------------------------------------------

x = df.iloc[:, 1:].values
y = df.iloc[:, 0].values

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

x_train = x_train / 225.0
x_test = x_test / 225.0


# -----------------------------------------------------------------------------
# Dataset and DataLoader (same structure as before)
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

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


# -----------------------------------------------------------------------------
# ANN with Dropout and (optional) BatchNorm
#
# Dropout(p=0.5) -> randomly zeroes 50% of neurons each forward pass during training
# Dropout(p=0.7) -> randomly zeroes 70% of neurons in second hidden layer
#   Higher dropout on deeper layers is a common strategy
#
# Dropout is ONLY active during training (model.train() mode).
# During evaluation (model.eval()), dropout is automatically disabled.
# PyTorch handles this automatically when you call model.eval().
#
# BatchNorm1d (commented out):
#   Normalizes each mini-batch's activations (mean=0, std=1)
#   Usually reduces overfitting, but in this specific experiment
#   it made things worse — analysis showed 94% on test when used alone.
#   This shows hyperparameter tuning requires experimentation.
# -----------------------------------------------------------------------------

class MyNN(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(num_features, 128),
            # nn.BatchNorm1d(128),     # batch norm after linear, before activation
            nn.ReLU(),
            nn.Dropout(p=0.5),         # randomly drop 50% of 128 neurons

            nn.Linear(128, 64),
            # nn.BatchNorm1d(64),      # note: caused overfitting in this case
            nn.ReLU(),
            nn.Dropout(p=0.7),         # randomly drop 70% of 64 neurons

            nn.Linear(64, 10)          # output layer: 10 class scores
        )

    def forward(self, x):
        return self.model(x)


# -----------------------------------------------------------------------------
# Optimizer with L2 Regularization (weight_decay)
# weight_decay=1e-4 adds a penalty term to the loss: loss + lambda * sum(w^2)
# This prevents any weight from growing too large.
# SGD with weight_decay is equivalent to L2 regularization.
# -----------------------------------------------------------------------------

epochs = 100
learning_rate = 0.1

model = MyNN(x_train.shape[1])
criterion = nn.CrossEntropyLoss()

# weight_decay is the key difference from 06_ann.py
optimizer = optim.SGD(model.parameters(), lr=learning_rate, weight_decay=1e-4)


# -----------------------------------------------------------------------------
# Training Loop (same structure as 06_ann.py)
# model is automatically in training mode after instantiation
# Dropout is active here (neurons randomly zeroed each batch)
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
    # print(f"Epochs: {epoch + 1}, loss: {ave_epoch_loss},")


# -----------------------------------------------------------------------------
# Evaluation on Test Data
# model.eval() must be called before evaluation
#   -> disables Dropout (all neurons active for inference)
#   -> disables BatchNorm random behavior
# -----------------------------------------------------------------------------

model.eval()   # IMPORTANT: disables dropout before evaluation

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
# Compare with test accuracy to check if overfitting was reduced
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
# Conclusion:
#   Dropout and L2 regularization reduce the train-test gap.
#   Model is still overfitting but the techniques are working.
#   Next step: try CNN which has spatial awareness for image features (08_cnn.py)
# =============================================================================
