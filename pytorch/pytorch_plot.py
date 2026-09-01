import torch
import matplotlib.pyplot as plt

# 1. SETUP: Create some dummy data (The "Straight Line") --- create a simple relationship: y = 0.7 * X + 0.3
weight_true = 0.7
bias_true = 0.3

# Create 50 data points from 0 to 1
X = torch.linspace(0, 1, 50).unsqueeze(1) # Shape: (50, 1)
y = weight_true * X + bias_true

# 2. SPLIT: splitting data ---
train_split = int(0.8 * len(X)) # 80% of data used for training set, 20% for testing 
X_train, y_train = X[:train_split], y[:train_split]
X_test, y_test = X[train_split:], y[train_split:]

# Verify sizes (optional)
print(f"Training samples: {len(X_train)}, Testing samples: {len(X_test)}")

# 3. DEFINE: the plotting function ---
def plot_predictions(train_data=X_train, 
                     train_labels=y_train, 
                     test_data=X_test, 
                     test_labels=y_test, 
                     predictions=None):
  """
  Plots training data, test data and compares predictions.
  """
  plt.figure(figsize=(10, 7))

  # Plot training data in blue
  plt.scatter(train_data, train_labels, c="b", s=4, label="Training data")
  
  # Plot test data in green
  plt.scatter(test_data, test_labels, c="g", s=4, label="Testing data")

  if predictions is not None:
    # Plot the predictions in red (predictions were made on the test data)
    # Note: We ensure predictions are on CPU and detached for plotting
    plt.scatter(test_data, predictions, c="r", s=4, label="Predictions")

  # Show the legend
  plt.legend(prop={"size": 14})
  plt.title("PyTorch Workflow: Data vs Predictions")
  plt.xlabel("X")
  plt.ylabel("y")
  plt.show()

# 4. MODEL: Create a simple random model (to generate predictions) --- This mimics a model that hasn't learned yet (random guesses)
class SimpleLinearModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = torch.nn.Parameter(torch.randn(1))
        self.bias = torch.nn.Parameter(torch.randn(1))
    
    def forward(self, x):
        return self.weights * x + self.bias

model = SimpleLinearModel()

# --- 5. EXECUTE: Make predictions and Plot! ---

# A. Make predictions using the model (inference mode)
with torch.inference_mode():
    # Get predictions for the TEST data
    y_preds = model(X_test)

# B. Call your function to print the output (display the chart)
print("Generating plot with random predictions (Red dots will likely miss Green dots)...")
plot_predictions(predictions=y_preds)

# Let's pretend the model learned perfectly ---
# If we manually set the weights to the true values, the red dots will match the green ones
model.weights.data = torch.tensor([weight_true])
model.bias.data = torch.tensor([bias_true])

with torch.inference_mode():
    y_preds_perfect = model(X_test)

print("Generating plot with perfect predictions (Red dots should match Green dots)...")
plot_predictions(predictions=y_preds_perfect)