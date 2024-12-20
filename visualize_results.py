import torch
import matplotlib.pyplot as plt
import numpy as np

# Load the data
data = torch.load('mock_data.pt')
X, y = data['X'], data['y']

# Load and evaluate model
from linear_network import SimpleLinearNetwork
model = SimpleLinearNetwork()

# Training with loss tracking
criterion = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
losses = []

print("Training model and tracking loss...")
for epoch in range(100):
    outputs = model(X)
    loss = criterion(outputs, y)
    losses.append(loss.item())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Generate predictions
model.eval()
with torch.no_grad():
    predictions = model(X)

# Plot training loss
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(losses)
plt.title('Training Loss Over Time')
plt.xlabel('Epoch')
plt.ylabel('Loss')

# Plot predictions vs actual
plt.subplot(1, 2, 2)
plt.scatter(y.numpy(), predictions.numpy(), alpha=0.5)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', label='Perfect Prediction')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Predictions vs Actual Values')
plt.legend()

plt.tight_layout()
plt.savefig('training_results.png')
print("Results saved to training_results.png")

# Print numerical metrics
mse = criterion(predictions, y).item()
correlation = np.corrcoef(predictions.numpy().flatten(), y.numpy().flatten())[0,1]
print(f"\nModel Performance Metrics:")
print(f"Mean Squared Error: {mse:.4f}")
print(f"Correlation Coefficient: {correlation:.4f}")
