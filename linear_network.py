import torch
import torch.nn as nn

class SimpleLinearNetwork(nn.Module):
    def __init__(self, input_size=10, hidden_size=5, output_size=1):
        super(SimpleLinearNetwork, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x

# Create an instance of the network
model = SimpleLinearNetwork()
print("Model architecture:")
print(model)
