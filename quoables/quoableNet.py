import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)  # Input channels=3 (RGB), Output channels=16
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  # Reduce dimensions by half
        self.fc1 = nn.Linear(32 * 32 * 32, 128)  # 32x32 from the image size after pooling
        self.fc2 = nn.Linear(128, num_classes)   # Output layer

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # Conv1 -> ReLU -> Pool
        x = self.pool(F.relu(self.conv2(x)))  # Conv2 -> ReLU -> Pool
        x = x.view(-1, 32 * 32 * 32)          # Flatten
        x = F.relu(self.fc1(x))               # Fully Connected 1 -> ReLU
        x = self.fc2(x)                       # Fully Connected 2 (Output)
        return x
