import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
    """
    Baseline CNN architecture for the 3x28x28 image classification.

    Architecture:
    - Conv2d(3, 32, ks=3, p=1) -> BatchNorm2d(32) -> ReLU -> MaxPool2d(2, 2)
    - Conv2d(32, 64, ks=3, p=1) -> BatchNorm2d(64) -> ReLU -> MaxPool2d(2, 2)
    - Flatten
    - Linear(64 * 7 * 7, 128) -> ReLU -> Dropout(p=0.5)
    - Linear(128, num_classes)
    """

    def __init__(self, input_channels=3, num_classes=7, dropout_rate=0.5):
        """
        Initializes the CNN layers.

        Args:
            input_channels (int): Number of channels in the input image (e.g., 3 for RGB).
            num_classes (int): Number of output classes.
            dropout_rate (float): Dropout probability for the fully connected layer.
        """
        super(CNN, self).__init__()

        """
        # Baseline architecture
        # ----------------------------------------------------------------------------
        # --- Convolutional Block 1 ---
        self.conv1 = nn.Conv2d(
            input_channels, 32, kernel_size=3, padding=1, bias=False
        )  # Often omit bias when using BatchNorm
        self.bn1 = nn.BatchNorm2d(32)  # Batch norm for 32 channels
        # ReLU is applied in forward pass
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # 28x28 -> 14x14

        # --- Convolutional Block 2 ---
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)  # Batch norm for 64 channels
        # ReLU is applied in forward pass
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # 14x14 -> 7x7

        # ----------------------------------------------------------------------------
        """

        """
        # Experiment 1: Increase network depth
        # --- Convolutional Block 3 ---
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(128)  # Batch norm for 128 channels
        # ReLU is applied in forward pass
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        # 7x7 -> 3x3
        """

        """# Experiment 2: Increase network width in baseline architecture
        # ----------------------------------------------------------------------
        # --- Convolutional Block 1 ---
        self.conv1 = nn.Conv2d(
            input_channels, 64, kernel_size=3, padding=1, bias=False
        )  # Often omit bias when using BatchNorm
        self.bn1 = nn.BatchNorm2d(64)  # Batch norm for 32 channels
        # ReLU is applied in forward pass
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # 28x28 -> 14x14

        # --- Convolutional Block 2 ---
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(128)  # Batch norm for 64 channels
        # ReLU is applied in forward pass
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # 14x14 -> 7x7
        """

        """ # Experiment 3 - Change kernel Size 3 -> 5
        # ----------------------------------------------------------------------------
        # --- Convolutional Block 1 ---
        self.conv1 = nn.Conv2d(
            input_channels, 32, kernel_size=5, padding=2, bias=False
        )  # Often omit bias when using BatchNorm
        self.bn1 = nn.BatchNorm2d(32)  # Batch norm for 32 channels
        # ReLU is applied in forward pass
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # 28x28 -> 14x14

        # --- Convolutional Block 2 ---
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding=2, bias=False)
        self.bn2 = nn.BatchNorm2d(64)  # Batch norm for 64 channels
        # ReLU is applied in forward pass
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # 14x14 -> 7x7
        """

        # Experiment 4 - change pooling type
        # ----------------------------------------------------------------------------
        # --- Convolutional Block 1 ---
        self.conv1 = nn.Conv2d(
            input_channels, 32, kernel_size=3, padding=1, bias=False
        )  # Often omit bias when using BatchNorm
        self.bn1 = nn.BatchNorm2d(32)  # Batch norm for 32 channels
        # ReLU is applied in forward pass
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # 28x28 -> 14x14

        # --- Convolutional Block 2 ---
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)  # Batch norm for 64 channels
        # ReLU is applied in forward pass
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # 14x14 -> 7x7

        # ------------------------------------------------------------------------

        # --- Classifier Head ---
        # Calculate the flattened size after the pooling layers
        # Assuming input 28x28: After pool1 (14x14), after pool2 (7x7), after pool3 (3x3)
        # Size = num_output_channels * height * width
        flattened_size = 64 * 7 * 7
        self.fc1 = nn.Linear(flattened_size, 256)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.fc2 = nn.Linear(128, num_classes)

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
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool2(x)

        """
        # Block 3 (experiment 1)
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.pool3(x)
        """

        # Flatten
        # x = x.view(x.size(0), -1) # Alternative flatten
        x = torch.flatten(x, 1)  # Flatten all dimensions except batch

        # Classifier
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)  # Output logits (no activation needed for nn.CrossEntropyLoss)

        return x
