import numpy as np
import random

class OvertimeModel:
    def __init__(self):
        self.w = random.uniform(-1, 1)
        self.b = random.uniform(-1, 1)
        
    def forward(self, x):
        # Linear calculation: w * x + b
        return (self.w * x) + self.b
    
    def train_step(self, x, y_true, learning_rate):
        y_pred = self.forward(x)
        error = y_true - y_pred
        
        # Update weights
        self.w = self.w + (learning_rate * error * x)
        self.b = self.b + (learning_rate * error)
        
        return abs(error)

# --- YOUR EXACT DATA ---
data_input = np.array([[5], [6], [7], [8], [9], [10], [11], [12]]) # features
data_output = np.array([[0], [0], [0], [0], [1], [2], [3], [4]])   # label

# Configuration
learning_rate = 0.01
epochs = 5000

model = OvertimeModel()

print(f"--- Starting Training ---")
print(f"Initial Weight: {model.w:.4f}, Bias: {model.b:.4f}\n")

for epoch in range(epochs):
    total_error = 0
    for i in range(len(data_input)):
        x = data_input[i][0]
        y = data_output[i][0]
        total_error += model.train_step(x, y, learning_rate)
    
    # Check progress every 1000 epochs
    if epoch % 1000 == 0:
        # Predict for 8 hours
        pred_8 = model.forward(8)
        # Predict for 10 hours
        pred_10 = model.forward(10)
        print(f"Epoch {epoch}: Avg Error: {total_error/8:.4f}")
        print(f"  -> Predict 8hrs: {pred_8:.2f} (Target: 0)")
        print(f"  -> Predict 10hrs: {pred_10:.2f} (Target: 2)")
        print("-" * 30)

print(f"\n--- Training Complete ---")
print(f"Final Weight: {model.w:.4f}")
print(f"Final Bias: {model.b:.4f}")

# --- Test: Compare Linear vs. Real-AI (with ReLU) ---
print("\n--- Final Predictions ---")
print(f"{'Hours':<6} | {'Linear Model':<12} | {'ReLU Model':<12} | {'Actual OT':<10} | {'Status'}")
print("-" * 55)

for x in range(5, 13):
    pred_linear = model.forward(x)
    
    # THE FIX: ReLU (Rectified Linear Unit)
    # This forces any negative number to 0
    pred_relu = max(0, pred_linear)
    
    actual = max(0, x - 8)
    
    # Status check
    status = "OK" if abs(pred_relu - actual) < 0.5 else "WRONG"
    
    print(f"{x:<6} | {pred_linear:<12.2f} | {pred_relu:<12.2f} | {actual:<10} | {status}")