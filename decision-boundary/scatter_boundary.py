import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons, make_classification, make_multilabel_classification
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

# Generate a synthetic dataset
X, y = make_moons(n_samples=1000, noise=0.3, random_state=42)
# X, y = make_classification(n_samples=1000, n_classes=3, n_features=2, n_informative=2, n_clusters_per_class=1, n_redundant=0, random_state=42)
# For more: https://scikit-learn.org/stable/datasets/sample_generators.html#generators-for-classification-and-clustering

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Function to plot decision boundary
def plot_decision_boundary(model, X, y):
  x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
  y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
  xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                       np.linspace(y_min, y_max, 100))
  Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
  Z = Z.reshape(xx.shape)
  plt.contourf(xx, yy, Z, alpha=0.3)
  plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', marker='o', s=20)
  plt.title("Decision Boundary")
  plt.xlabel("Feature 1")
  plt.ylabel("Feature 2")

def plot_scatter_with_boundary(layer_size, X_train, X_test, y_train, y_test):
  """
  Example call: plot_scatter_with_boundary(50, X_train, X_test, y_train, y_test)
  """
  mlp = MLPClassifier(hidden_layer_sizes=(layer_size,), max_iter=1000)
  mlp.fit(X_train, y_train)
  y_pred = mlp.predict(X_test)
  print(f'Accuracy: {accuracy_score(y_test, y_pred)}')
  #plt.subplot(1, 2, 2)
  plot_decision_boundary(mlp, X_test, y_test)

# Underfitting example: Shallow network
mlp_underfit = MLPClassifier(hidden_layer_sizes=(2,), max_iter=1000)
mlp_underfit.fit(X_train, y_train)
y_pred_underfit = mlp_underfit.predict(X_test)
print(f'Underfitting Accuracy: {accuracy_score(y_test, y_pred_underfit)}')
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plot_decision_boundary(mlp_underfit, X_test, y_test)

# Overfitting example: Deep network with too many neurons
mlp_overfit = MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000)
mlp_overfit.fit(X_train, y_train)
y_pred_overfit = mlp_overfit.predict(X_test)
print(f'Overfitting Accuracy: {accuracy_score(y_test, y_pred_overfit)}')
plt.subplot(1, 2, 2)
plot_decision_boundary(mlp_overfit, X_test, y_test)

plt.show()
