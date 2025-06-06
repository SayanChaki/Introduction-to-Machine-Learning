# -*- coding: utf-8 -*-
"""Question3.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1UNPh58bqtwh6iCj92MSlF1LnuL1KaZyi
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# Generate synthetic spiral dataset for binary classification
def generate_spiral_data(n_samples=500):
    np.random.seed(42)
    torch.manual_seed(42)
    theta = np.sqrt(np.random.rand(n_samples)) * 2 * np.pi
    r_a = 2 * theta + np.pi
    data_a = np.array([np.cos(theta) * r_a, np.sin(theta) * r_a]).T
    data_a += np.random.randn(n_samples, 2) * 0.2

    r_b = -2 * theta - np.pi
    data_b = np.array([np.cos(theta) * r_b, np.sin(theta) * r_b]).T
    data_b += np.random.randn(n_samples, 2) * 0.2

    X = np.vstack([data_a, data_b])
    y = np.hstack([np.zeros(n_samples), np.ones(n_samples)])

    X = torch.FloatTensor(X)
    y = torch.LongTensor(y)
    return X, y

class MLPClassifier(nn.Module):
    def __init__(self, input_dim=2, hidden_dims=[64, 32], output_dim=2):
        super(MLPClassifier, self).__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dims[0])
        self.layer2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.layer3 = nn.Linear(hidden_dims[1], output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        """
        TODO: Complete this method to:
        - Implement the forward pass through the MLP
        - Apply appropriate activations to avoid vanishing gradients
        - Return class scores for binary classification
        """
        # Placeholder: Students must replace this
        pass

# Training and visualization
def train_and_visualize(model, X, y):
    '''finisj the train code'''


def test_and_visualize(model, X,y):

    # Visualize decision boundary
    model.eval()
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))
    grid = torch.FloatTensor(np.c_[xx.ravel(), yy.ravel()])
    with torch.no_grad():
        Z = model(grid).argmax(dim=1).reshape(xx.shape)

    plt.contourf(xx, yy, Z, alpha=0.3, cmap='viridis')
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', edgecolors='k')
    plt.title('MLP Decision Boundary')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.show()

# Main script
if __name__ == "__main__":
    # Generate dataset
    X, y = generate_spiral_data()

    # Initialize and run MLP
    model = MLPClassifier()
    train_and_visualize(model, X, y)
    test_and_visualize(model, X, y)

