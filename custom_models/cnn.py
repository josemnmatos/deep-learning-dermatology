import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):

    def __init__(self, input_channels=3, num_classes=7, dropout_rate=0.2):
        """
        Initializes the CNN.

        Args:
            input_channels (int): Number of channels in the input image (e.g., 3 for RGB).
            num_classes (int): Number of output classes.
            dropout_rate (float): Dropout probability for the fully connected layer.
        """
        super(CNN, self).__init__()
        self.num_classes = num_classes
        self.input_channels = input_channels

        # Baseline architecture
        # ----------------------------------------------------------------------------

        # --- Convolutional Block 1 ---
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)  # Batch norm for 64 channels
        # ReLU is applied in forward pass
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # 28x28 -> 14x14

        # --- Convolutional Block 2 ---
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(128)  # Batch norm for 128 channels
        # ReLU is applied in forward pass
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # 14x14 -> 7x7

        # ------------------------------------------------------------------------

        flattened_size = (
            128 * 7 * 7
        )  # Adjust based on the last pooling layer output size

        self.fc1 = nn.Linear(flattened_size, 1024)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.fc2 = nn.Linear(1024, num_classes)
        # --

    def forward(self, x):
        """
        Defines the forward pass of the network.
        """
        # Block 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool1(x)

        # Block 2
        x = self.conv2(x)
        x = self.bn2(x)  #
        x = F.relu(x)
        x = self.pool2(x)

        # Flatten
        x = torch.flatten(x, 1)  # Flatten all dimensions except batch

        # Classifier
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)

        return x
