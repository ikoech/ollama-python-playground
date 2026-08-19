import random
import numpy as np

# 1. Initialize the Model: Bias is always zero at start and weight is a random num
w = random.uniform(-1, 1) #Random weight (changes every run)
b = 0.0

# 2. Define Training Parameters
learning_rate = 0.001   #The limit on how much values can change per step
epochs = 2000           #Number of times to go through data

# 3. Training Data: Example is based on 1 times 2
inputs = [1]
targets = [2]
print(f"Start: Weight = {w}, Bias = {b}")

# 4. The Training Loop (The "Epochs")
for epoch in range(epochs):
    # Loop through every data point
    for x, y_true in zip(inputs, targets):
        # A. Forward Pass: Calculate prediction (w * x + b) "this is the calculation that is performed one time w plus b"
        y_pred = w * x + b
        
        # B. Calculate Error (How far off are we?)
        error = y_true - y_pred
        
        # C. Update Weights and Biases "w and b change values... limited by learning rate"
        # We adjust w and b to reduce the error
        w = w + (learning_rate * error * x)
        b = b + (learning_rate * error)
        
    # Optional: Print progress every 500 epochs to see the "change"
    if epoch % 500 == 0:
        print(f"Epoch {epoch}: Weight = {w:.4f}, Bias = {b:.4f}, Error = {error:.4f}")

print(f"Final: Weight = {w}, Bias = {b}")
print(f"Test: Input 1 gives result {w * 1 + b}")