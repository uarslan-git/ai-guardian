import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons, make_classification, make_multilabel_classification, make_blobs
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC


import pandas as pd

# Generate a synthetic dataset
#X, y = make_moons(n_samples=1000, noise=0.3, random_state=42)
# X, y = make_classification(n_samples=1000, n_features=2, n_clusters_per_class=1, n_redundant=0, random_state=40)
# X, y = make_blobs(n_samples=1000, n_features=2, centers=2, random_state=42)
# For more: https://scikit-learn.org/stable/datasets/sample_generators.html#generators-for-classification-and-clustering

# Use Spotify Dataset
data = pd.read_csv("../data/spotify_tracks.csv")
# Feature Names
# track_id,track_name,artist_name,year,popularity,artwork_url,album_name,acousticness,danceability,
# duration_ms,energy,instrumentalness,key,liveness,loudness,mode,speechiness,tempo,time_signature,
# valence,track_url,language
complete_X = data[['popularity', 'danceability']]
# X[0] := popularity
# X[1] := danceability

X = complete_X.to_numpy()[:1000]
X_popular = X[:, 0]
X_dance = X[:, 1]

y = np.zeros(shape=X.shape[0])
# y[:int(y.shape[0]/2)] = 1
# y[np.logical_or(X_popular > 50, np.logical_and(X_dance > 0.5, X_dance < 0.8))] = 1
y[np.logical_or(X_popular > 50, X_dance > 0.6)] = 1
# y[X_popular > 50] = 1

# Apply noise
print(f'Num likes:{int(np.sum(y))} / {y.shape[0]}')
noise_idx = np.random.choice(len(y), size=int(0.1*len(y)), replace=False)
y[noise_idx] = 1 - y[noise_idx]
print(f'Num likes:{int(np.sum(y))} / {y.shape[0]}')


# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

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

layer_sizes=[2,8,5000]
max_iter = 2000

# m_underfit = MLPClassifier(hidden_layer_sizes=(layer_sizes[0],), max_iter=max_iter)
m_underfit = SVC(kernel='rbf', C=1, random_state=42)
# m_optimal = MLPClassifier(hidden_layer_sizes=(layer_sizes[1],), max_iter=max_iter)
m_optimal = SVC(kernel='rbf', C=10000, random_state=42)
# m_overfit = MLPClassifier(hidden_layer_sizes=(layer_sizes[2],), max_iter=max_iter)
m_overfit = SVC(kernel='rbf', C=1000000, random_state=42)


# Underfitting example: Shallow network
m_underfit.fit(X_train, y_train)
y_pred_underfit = m_underfit.predict(X_test)
print(f'Underfitting Accuracy: {accuracy_score(y_test, y_pred_underfit)}')
# plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plot_decision_boundary(m_underfit, X_train, y_train, 'Underfitted Train')
plt.subplot(1, 2, 2)
plot_decision_boundary(m_underfit, X_test, y_test, 'Underfitted Test')
plt.show()


m_optimal.fit(X_train, y_train)
y_pred_optimal = m_optimal.predict(X_test)
print(f'Middle Accuracy: {accuracy_score(y_test, y_pred_optimal)}')
plt.subplot(1, 2, 1)
plot_decision_boundary(m_optimal, X_train, y_train, 'Middle Train')
plt.subplot(1, 2, 2)
plot_decision_boundary(m_optimal, X_test, y_test, 'Middle Test')
plt.show()


# Overfitting example: Deep network with too many neurons
m_overfit.fit(X_train, y_train)
y_pred_overfit = m_overfit.predict(X_test)
print(f'Overfitting Accuracy: {accuracy_score(y_test, y_pred_overfit)}')
plt.subplot(1, 2, 1)
plot_decision_boundary(m_overfit, X_train, y_train, 'Overfitted Train')
plt.subplot(1, 2, 2)
plot_decision_boundary(m_overfit, X_test, y_test, 'Overfitted Test')
plt.show()

# plot_scatter_with_boundary(5, X_train, X_test, y_train, y_test)
# plt.show()
# plot_scatter_with_boundary(50, X_train, X_test, y_train, y_test)
# plt.show()
# plot_scatter_with_boundary(1000, X_train, X_test, y_train, y_test)


#plt.show()

