import torch
import torch.nn as nn
from linear_network import SimpleLinearNetwork

# Load the mock data
data = torch.load('mock_data.pt')
X, y = data['X'], data['y']

# Create model instance
model = SimpleLinearNetwork()

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# Training loop
print("Starting training...")
for epoch in range(100):
    # Forward pass
    outputs = model(X)
    loss = criterion(outputs, y)

    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/100], Loss: {loss.item():.4f}')

# Test predictions
model.eval()
with torch.no_grad():
    test_predictions = model(X[:5])
    print("\nTest Predictions vs Actual (first 5 samples):")
    print(f"Predictions: {test_predictions.numpy().flatten()}")
    print(f"Actual: {y[:5].numpy().flatten()}")
