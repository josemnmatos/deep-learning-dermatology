import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, num_classes):
        super(MLP, self).__init__()
        self.layers = nn.ModuleList()
        self.activations = nn.ModuleList()

        if len(hidden_sizes) == 0:
            self.layers.append(nn.Linear(input_size, num_classes))
        else:
            # First layer
            self.layers.append(nn.Linear(input_size, hidden_sizes[0]))
            self.activations.append(nn.ReLU())

            # Intermediate layers
            for i in range(1, len(hidden_sizes)):
                self.layers.append(nn.Linear(hidden_sizes[i - 1], hidden_sizes[i]))
                self.activations.append(nn.ReLU())

            # Final layer
            self.layers.append(nn.Linear(hidden_sizes[-1], num_classes))

    def forward(self, x):
        # Apply hidden layers with activations
        for i in range(len(self.activations)):
            x = self.activations[i](self.layers[i](x))

        # Apply final output layer (without activation)
        x = self.layers[-1](x)
        return x

    # https://discuss.pytorch.org/t/reset-model-weights/19180/4
    def reset_parameters(self):
        for layer in self.children():
            if hasattr(layer, "reset_parameters"):
                layer.reset_parameters()
