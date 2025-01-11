import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons, make_classification, make_multilabel_classification
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score


import pandas as pd

# Generate a synthetic dataset
#X, y = make_moons(n_samples=1000, noise=0.3, random_state=42)
# X, y = make_classification(n_samples=1000, n_classes=3, n_features=2, n_informative=2, n_clusters_per_class=1, n_redundant=0, random_state=42)
# For more: https://scikit-learn.org/stable/datasets/sample_generators.html#generators-for-classification-and-clustering


# Use Spotify Dataset
data = pd.read_csv("../data/spotify_tracks.csv")
complete_X = data[['popularity', 'danceability']]

# X[0] := popularity
# X[1] := danceability

X = complete_X.to_numpy()[:1000]
X_popular = X[:, 0]
X_dance = X[:, 1]

y = np.zeros(shape=X.shape[0])
#y[:int(y.shape[0]/2)] = 1
y[np.logical_or(X_popular > 50, X_dance > 0.7)] = 1
print(f'Num likes:{int(np.sum(y))} / {y.shape[0]}')



# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Function to plot decision boundary
def plot_decision_boundary(model, X, y, title):
  x_min, x_max = X[:, 0].min(), X[:, 0].max()
  y_min, y_max = X[:, 1].min(), X[:, 1].max()
  xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                       np.linspace(y_min, y_max, 100))
  Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
  Z = Z.reshape(xx.shape)
  plt.contourf(xx, yy, Z, alpha=0.3)
  plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', marker='o', s=20)
  plt.title(title)
  plt.xlabel("Popularity")
  plt.ylabel("Danceability")

def plot_scatter_with_boundary(layer_size, X_train, X_test, y_train, y_test, title):
  """
  Example call: plot_scatter_with_boundary(50, X_train, X_test, y_train, y_test)
  """
  mlp = MLPClassifier(hidden_layer_sizes=(layer_size,), max_iter=2000)
  mlp.fit(X_train, y_train)
  y_pred = mlp.predict(X_test)
  print(f'Accuracy: {accuracy_score(y_test, y_pred)}')
  #plt.subplot(1, 2, 2)
  plot_decision_boundary(mlp, X_test, y_test, title)

layer_sizes=[5,100,1000]
max_iter = 1000

# Underfitting example: Shallow network
mlp_underfit = MLPClassifier(hidden_layer_sizes=(layer_sizes[0],), max_iter=max_iter)
mlp_underfit.fit(X_train, y_train)
y_pred_underfit = mlp_underfit.predict(X_test)
print(f'Underfitting Accuracy: {accuracy_score(y_test, y_pred_underfit)}')
# plt.figure(figsize=(12, 5))
#plt.subplot(3, 1, 1)
plot_decision_boundary(mlp_underfit, X_test, y_test, 'Underfitted')
plt.show()


mlp_optimal = MLPClassifier(hidden_layer_sizes=(layer_sizes[1],), max_iter=max_iter)
mlp_optimal.fit(X_train, y_train)
y_pred_optimal = mlp_optimal.predict(X_test)
print(f'Accuracy: {accuracy_score(y_test, y_pred_optimal)}')
# plt.figure(figsize=(12, 5))
#plt.subplot(3, 1, 2)
plot_decision_boundary(mlp_optimal, X_test, y_test, 'Middle')
plt.show()


# Overfitting example: Deep network with too many neurons
mlp_overfit = MLPClassifier(hidden_layer_sizes=(layer_sizes[2],), max_iter=max_iter)
mlp_overfit.fit(X_train, y_train)
y_pred_overfit = mlp_overfit.predict(X_test)
print(f'Overfitting Accuracy: {accuracy_score(y_test, y_pred_overfit)}')
#plt.figure(figsize=(12, 5))
#plt.subplot(3, 1, 3)
plot_decision_boundary(mlp_overfit, X_test, y_test, 'Overfitted')

# plot_scatter_with_boundary(5, X_train, X_test, y_train, y_test)
# plt.show()
# plot_scatter_with_boundary(50, X_train, X_test, y_train, y_test)
# plt.show()
# plot_scatter_with_boundary(1000, X_train, X_test, y_train, y_test)


plt.show()

