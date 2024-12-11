import numpy as np
import matplotlib.pyplot as plt

# Sample data
x = np.array([1, 1.5, 2, 3, 4, 5, 6])
y = np.array([20, 20, 60, 75, 85, 90, 95])

# Create subplots
fig, axs = plt.subplots(1, 3, figsize=(15, 5))

# Underfitting (1st degree polynomial)
mymodel_underfit = np.poly1d(np.polyfit(x, y, 1))
axs[0].scatter(x, y)
axs[0].plot(np.linspace(1, 6, 100), mymodel_underfit(np.linspace(1, 6, 100)), color='orange')
axs[0].set_title('Underfitting: Too Simple Model')

# Good Fit (2nd degree polynomial)
mymodel_goodfit = np.poly1d(np.polyfit(x, y, 2))
axs[1].scatter(x, y)
axs[1].plot(np.linspace(1, 6, 100), mymodel_goodfit(np.linspace(1, 6, 100)), color='green')
axs[1].set_title('Good Fit: Just Right Model')

# Overfitting (5th degree polynomial)
mymodel_overfit = np.poly1d(np.polyfit(x, y, 10))
axs[2].scatter(x, y)
axs[2].plot(np.linspace(1, 6, 100), mymodel_overfit(np.linspace(1, 6, 100)), color='red')
axs[2].set_title('Overfitting: Too Complex Model')

plt.show()
