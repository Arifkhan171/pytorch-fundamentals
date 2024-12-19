# PyTorch Fundamentals

A progressive series of scripts covering core PyTorch concepts, starting from tensors and autograd up to training CNNs on real image data. Each script builds on the previous one, showing how the training pipeline gets cleaner and more powerful as we add PyTorch's built-in tools.

---

## Topics Covered

| Script | Topic |
|--------|-------|
| `01_tensors.py` | Tensor creation, shapes, math, matrix ops, NumPy conversion |
| `02_autograd.py` | Automatic differentiation, gradient computation, backpropagation |
| `03_training_pipeline.py` | Manual training loop built from scratch using raw PyTorch |
| `04_nn_module.py` | Rewriting the pipeline using `nn.Module`, built-in loss and optimizers |
| `05_dataset_dataloader.py` | Custom `Dataset` class, `DataLoader`, mini-batch training |
| `06_ann.py` | ANN on Fashion MNIST — multi-class image classification |
| `07_ann_with_regularization.py` | Adding Dropout and L2 regularization to reduce overfitting |
| `08_cnn.py` | CNN on Fashion MNIST — Conv2D, BatchNorm, MaxPool |

---

## Project Structure

```
pytorch-fundamentals/
├── data/
│   └── breast-cancer.csv
├── 01_tensors.py
├── 02_autograd.py
├── 03_training_pipeline.py
├── 04_nn_module.py
├── 05_dataset_dataloader.py
├── 06_ann.py
├── 07_ann_with_regularization.py
├── 08_cnn.py
├── requirements.txt
└── .gitignore
```

---

## Datasets

**Breast Cancer Wisconsin** (`data/breast-cancer.csv`)  
Used in scripts 03, 04, 05. Binary classification task (Malignant/Benign). Already included in this repo.

**Fashion MNIST Small** (`data/fmnist_small.csv`)  
Used in scripts 06, 07, 08. A CSV version of Fashion MNIST with 10 clothing categories. Not included due to file size.  
Download it from Kaggle: [Fashion MNIST](https://www.kaggle.com/datasets/zalando-research/fashionmnist)  
Save the file as `data/fmnist_small.csv` before running those scripts.

---

## Setup

```bash
git clone https://github.com/Arifkhan171/pytorch-fundamentals.git
cd pytorch-fundamentals
pip install -r requirements.txt
```

---

## How to Run

Run any script directly:

```bash
python 01_tensors.py
python 02_autograd.py
python 03_training_pipeline.py
```

Scripts 06, 07, and 08 require the Fashion MNIST CSV in the `data/` folder before running.

---

## Results

### Breast Cancer Classification (scripts 03-05)
- Simple logistic regression implemented manually and then with `nn.Module`
- Accuracy around 95-97% depending on random split

### Fashion MNIST - ANN (script 06)
- Training accuracy: 99.75%
- Test accuracy: 82.25%
- Observation: model is overfitting

### Fashion MNIST - ANN with Regularization (script 07)
- Added Dropout (0.5, 0.7) and L2 weight decay
- Reduces the gap between training and test accuracy

### Fashion MNIST - CNN (script 08)
- Two Conv2D layers with BatchNorm and MaxPool
- Test accuracy: 87.25%
- CNN outperforms ANN on image data, as expected

---

## Key Concepts Practiced

- Tensor operations and NumPy interoperability
- Computational graphs and autograd
- Manual training loop (forward, loss, backward, zero grad, step)
- `nn.Module` for clean model definition
- `nn.Sequential` for stacking layers
- `Dataset` and `DataLoader` for efficient mini-batch training
- Overfitting detection and reduction with Dropout and weight decay
- CNN architecture: Conv2D, BatchNorm2D, MaxPool2D

---

## Requirements

- Python 3.9+
- PyTorch 2.0+
- See `requirements.txt` for full list
