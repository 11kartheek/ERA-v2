# ResNet Implementation on CIFAR Dataset

## Introduction
This Folder contains code to work with the CIFAR-10 dataset. It consists of 60,000 32x32 color images in 10 classes, with 6,000 images per class. The classes are: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, and truck. 

This implementation provides a straightforward setup for training and evaluating ResNet on the CIFAR dataset.

## Files
### `utils.py`
This file contains utility functions for loading training and testing transformations.

### `models.py`
The models.py file contains the definition of a neural network model architecture for solving the CIFAR-10 classification problem. It defines the structure of the neural network using PyTorch.

### `s10.ipynb`
The s10.ipynb notebook contains the main code to train and evaluate a model on the CIFAR-10 dataset.  It imports necessary functions from utils.py to handle data processing, as well as the neural network architecture (ResNet) from models.py.The notebook then trains the model using the training data, evaluates its performance on the test data.

## Usage
To use this code, follow these steps:

Clone the repository to your local machine:
 ```bash
git clone https://github.com/11kartheek/ERA-v2.git
```
Open the s10.ipynb notebook in a Jupyter environment or any other compatible IDE.

Run the cells in the notebook sequentially to:

1.Load the CIFAR-10 dataset.
2.Define and initialize the ResNet model.
3.Train the model using the training data.
4.Evaluate the model's performance on the test data.
5.Experiment with different hyperparameters and training strategies within the notebook to potentially enhance model performance.

Run the cells in the notebook sequentially to load the data, load the model architecture, train the model, and evaluate its performance.

Experiment with different hyperparameters and training strategies to improve performance if desired.
