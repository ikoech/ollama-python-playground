import random
import numpy as np

class house_pred:
    def __init__(self):
        # Random weight (starts different every time)
        # Using a small random range to start closer to reality

        self.w = random.uniform(-0.5, 0.5)
        self.b = 0.0    # Bias always starts at 0

    def forward(self, x):
        return (self.w * x) + self.b     # The calculation: (weight * input) + bias

    def train_step(self, x, y_true, learning_rate):
        # We don't use a loop here for simplicity, 
        # but in real life we loop through all data points.
        # For a single example, we just run the update logic.

        y_pred = self.forward(x)   # 1. Make prediction
        error = y_true - y_pred    # 2. Calculate error

        # 3. Update Weights and Bias (The "Learning")
        # The change is limited by the learning_rate
        self.w = self.w + (learning_rate * error * x)
        self.b = self.b + (learning_rate * error)

        return abs(error)

# --- 2. The Real World Data (Tiny Dataset) ---
    # House Size (sqm) -> Expected Price (in $1000s)
    # House 1: 50 sqm -> $150k
    # House 2: 60 sqm -> $180k  
    # House 3: 70 sqm -> $210k
    # House 4: 80 sqm -> $240k
    # Pattern: Price = 3 * Size (roughly) + 0

house_sizes = [50, 60, 70, 80]
house_prices = [150, 180, 210, 240]

# --- 3. DATA NORMALIZATION (The Fix) ---
# We scale inputs to be between 0 and 1
min_size = min(house_sizes)
max_size = max(house_sizes)
normalized_sizes = [(x - min_size) / (max_size - min_size) for x in house_sizes]

# We scale outputs to be between 0 and 1 as well (optional but good practice)
min_price = min(house_prices)
max_price = max(house_prices)
normalized_prices = [(y - min_price) / (max_price - min_price) for y in house_prices]

print("Normalized Data (0 to 1 scale):")
print(f"Sizes: {normalized_sizes}")
print(f"Prices: {normalized_prices}")

# --- 4. Configuration ---
learning_rate = 0.1  # We can use a HIGHER learning rate now because data is small!
epochs = 2000

model = house_pred()

# --- 5. The Training Loop (Simulating the Lecturer's Logic) ---
print(f"--- Starting Training ---")
print(f"Initial Weight: {model.w:.4f}, Bias: {model.b:.4f}")
print(f"Learning Rate: {learning_rate}, Epochs: {epochs}\n")

for epoch in range(epochs):
    total_error = 0
    
    for i in range(len(normalized_sizes)):
        x = normalized_sizes[i]
        y = normalized_prices[i]
        total_error += model.train_step(x, y, learning_rate)
    
    if epoch % 500 == 0:
        # Test prediction for a normalized 100sqm house
        # 100sqm -> (100-50)/(80-50) = 1.66 (Wait, 100 is outside our training range!)
        # Let's test 65sqm -> (65-50)/30 = 0.5
        test_x = (65 - min_size) / (max_size - min_size)
        test_pred = model.forward(test_x)
        
        # Convert back to real price for display
        real_pred_price = (test_pred * (max_price - min_price)) + min_price
        
        print(f"Epoch {epoch}: Avg Error: {total_error/4:.4f} | Predicted 65sqm: ${real_pred_price:.2f}k")

print(f"\n--- Training Complete ---")
print(f"Final Weight: {model.w:.4f}")
print(f"Final Bias: {model.b:.4f}")

# --- 6. Test with 95 sqm
test_size_real = 95
test_size_norm = (test_size_real - min_size) / (max_size - min_size)
predicted_norm = model.forward(test_size_norm)
predicted_real = (predicted_norm * (max_price - min_price)) + min_price

print(f"\n--- Final Test ---")
print(f"Input: {test_size_real} sqm")
print(f"Normalized Input: {test_size_norm:.2f}")
print(f"Model Prediction (Normalized): {predicted_norm:.4f}")
print(f"Final Prediction: ${predicted_real:.2f}k")