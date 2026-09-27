# =============================================================================
# Dataset and DataLoader in PyTorch
# =============================================================================
# When training on large data, loading everything into RAM at once is a problem.
# PyTorch solves this with two classes that work together:
#
#   Dataset    -> wraps your data and defines how to access one sample at a time
#   DataLoader -> batches the samples, shuffles them, and loads them efficiently
#
# This pipeline supports:
#   - Mini-batch training (better generalization than full-batch)
#   - Shuffling each epoch (reduces ordering bias)
#   - Parallel data loading (num_workers argument)
#   - Easy transformation of data before it enters the model
# =============================================================================

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder


# =============================================================================
# PART 1 — Understanding Dataset and DataLoader with toy data (commented out)
# =============================================================================
# Dataset class requires 3 methods:
#   __init__()     -> store the data (features and labels)
#   __len__()      -> return number of samples (DataLoader needs this for batching)
#   __getitem__()  -> return one sample by index (DataLoader calls this per item)


# from sklearn.datasets import make_classification
# import torch
# x, y = make_classification(
#     n_samples=10,    # total rows
#     n_features=2,    # number of feature columns
#     n_classes=2,     # binary: 0 and 1
#     n_redundant=0,
#     n_informative=2,
#     random_state=42
# )
#
# x = torch.tensor(x, dtype=torch.float32)
# y = torch.tensor(y, dtype=torch.long)
# print(y)
#
# from torch.utils.data import DataLoader, Dataset
#
# class CustomDataset(Dataset):
#     def __init__(self, features, labels):
#         self.features = features
#         self.labels = labels
#
#     def __len__(self):
#         return self.features.shape[0]   # number of rows = number of samples
#
#     def __getitem__(self, index):
#         # DataLoader calls this with each index from a batch
#         # you can apply transforms here before returning
#         return self.features[index], self.labels[index]
#
#
# dataset = CustomDataset(x, y)
# print(len(dataset))    # 10
# print(dataset[0])      # first sample
#
# # DataLoader wraps the dataset, adds batching and shuffling
# dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
# for batch_features, batch_labels in dataloader:
#     print(batch_features)   # 2 rows at a time
#     print(batch_labels)
#     print("=" * 50)


# =============================================================================
# PART 2 — Sampler (how DataLoader shuffles data)
# =============================================================================
# DataLoader uses a Sampler to decide which indices to pull each epoch.
# Three built-in sampler types:
#
#   SequentialSampler -> indices in order [0, 1, 2, 3, ...]
#   RandomSampler     -> indices shuffled randomly each epoch
#   WeightedRandomSampler -> give higher chance to underrepresented classes
#                            (useful for imbalanced datasets)
#
# When shuffle=True in DataLoader, it uses RandomSampler automatically.


# =============================================================================
# PART 3 — Collate Function (how DataLoader combines samples into a batch)
# =============================================================================
# After sampling indices, DataLoader calls __getitem__ for each index.
# It then uses collate_fn to combine those individual samples into one batch tensor.
#
# Default collate_fn works for fixed-size tensors (like images or tabular data).
# Custom collate_fn is needed for variable-length data (like NLP sentences),
# where you need to pad shorter sequences to match the longest in the batch.


# =============================================================================
# PART 4 — Full Pipeline with Dataset and DataLoader (Breast Cancer data)
# =============================================================================

# -----------------------------------------------------------------------------
# Load and Prepare Data
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

x_train_tensor = torch.from_numpy(x_train.astype(np.float32))
x_test_tensor = torch.from_numpy(x_test.astype(np.float32))
y_train_tensor = torch.from_numpy(y_train.astype(np.float32))
y_test_tensor = torch.from_numpy(y_train.astype(np.float32))
print(x_train_tensor.shape)


# -----------------------------------------------------------------------------
# Custom Dataset Class
# Wraps our tensors into a Dataset that DataLoader can work with
# Any preprocessing or augmentation per sample goes inside __getitem__
# -----------------------------------------------------------------------------

class CustomDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return self.features.shape[0]

    def __getitem__(self, index):
        # DataLoader calls this for each index in a batch
        # Apply per-sample transformations here if needed
        return self.features[index], self.labels[index]


# create dataset objects for train and test
train_dataset = CustomDataset(x_train_tensor, y_train_tensor)
test_dataset = CustomDataset(x_test_tensor, y_test_tensor)

# -----------------------------------------------------------------------------
# DataLoader
# batch_size=32 -> process 32 samples at once (good balance of speed vs memory)
# shuffle=True  -> re-shuffle training data every epoch (prevents ordering bias)
# shuffle=False on test -> order doesn't matter for evaluation, no need to shuffle
# -----------------------------------------------------------------------------

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)


# -----------------------------------------------------------------------------
# Model (same as before — simple binary classifier)
# -----------------------------------------------------------------------------

class MySimpleNN(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.linear = nn.Linear(num_features, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, features):
        out = self.linear(features)
        out = self.sigmoid(out)
        return out


learning_rate = 0.1
epochs = 35
loss_function = nn.BCELoss()
model = MySimpleNN(x_train_tensor.shape[1])
optimizer = torch.optim.SGD(model.parameters(), learning_rate)


# -----------------------------------------------------------------------------
# Mini-Batch Training Loop
# Key difference from previous scripts:
#   - outer loop: epochs (how many full passes over data)
#   - inner loop: batches (DataLoader yields one batch at a time)
# Each batch of 32 samples does one forward + backward + optimizer step
# This is called Stochastic Gradient Descent (or mini-batch GD)
# Mini-batches converge faster and generalize better than full-batch GD
# -----------------------------------------------------------------------------

for epoch in range(epochs):
    for batch_features, batch_labels in train_loader:

        # forward pass on this batch
        y_pred = model(batch_features)

        # compute loss for this batch
        loss = loss_function(y_pred, batch_labels.view(-1, 1))

        # reset gradients before backward pass
        optimizer.zero_grad()

        # backward pass — compute gradients
        loss.backward()

        # update weights based on gradients
        optimizer.step()

        print(f'epoch: {epoch + 1}, loss: {loss.item()}')


print("weight of layer", model.linear.weight)
print("bias of layer", model.linear.bias)


# -----------------------------------------------------------------------------
# Evaluation with DataLoader
# We iterate through test batches and collect accuracy for each batch
# Then average them for overall accuracy
# model.eval() disables dropout and batchnorm randomness during evaluation
# -----------------------------------------------------------------------------

model.eval()
accuracy_list = []
with torch.no_grad():
    for batch_features, batch_labels in test_loader:
        y_pred = model(batch_features)
        y_pred = (y_pred > 0.5).float()
        batch_accuracy = (y_pred.view(-1) == batch_labels).float().mean().item()
        accuracy_list.append(batch_accuracy)

overall_accuracy = sum(accuracy_list) / len(accuracy_list)
print(f'Overall Accuracy is: {overall_accuracy:.4f}')
