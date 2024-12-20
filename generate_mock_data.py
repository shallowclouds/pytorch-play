import torch
import numpy as np

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Generate mock input data (100 samples, 10 features)
X = torch.randn(100, 10)

# Generate mock target data using a simple rule
# y = sum of input features + some noise
y = torch.sum(X, dim=1, keepdim=True) + 0.1 * torch.randn(100, 1)

# Save the data
torch.save({'X': X, 'y': y}, 'mock_data.pt')

print("Mock data generated:")
print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")
print("\nFirst 5 samples:")
print(f"X[:5]:\n{X[:5]}")
print(f"y[:5]:\n{y[:5]}")
