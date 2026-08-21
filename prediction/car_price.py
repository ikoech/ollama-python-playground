import numpy as np
import random

# 1. The Data (Used Car Dataset)
# Features: [Mileage(km), Age(yrs)]
# Target: Price ($)
# Pattern: Price drops by $0.10 per km and $1,000 per year
# Starting base price: $30,000

data = [
    {"mileage": 10000, "age": 1, "price": 28000},
    {"mileage": 50000, "age": 3, "price": 22000},
    {"mileage": 80000, "age": 5, "price": 16000},
    {"mileage": 120000, "age": 7, "price": 10000},
    {"mileage": 20000, "age": 2, "price": 26000},
    {"mileage": 60000, "age": 4, "price": 19000},
    {"mileage": 90000, "age": 6, "price": 13000},
    {"mileage": 30000, "age": 3, "price": 23000}
]

# Extract features and target
X_mileage = np.array([d["mileage"] for d in data])
X_age = np.array([d["age"] for d in data])
Y_price = np.array([d["price"] for d in data])

# --- 2. Data Normalization (CRITICAL STEP) ---
# Mileage is in tens of thousands, Age is in single digits, Price is in thousands.
# If we don't normalize, the model will struggle (like the "Exploding Gradient" error before).

# Normalize Mileage (0 to 1)
min_mileage, max_mileage = X_mileage.min(), X_mileage.max()
X_mileage_norm = (X_mileage - min_mileage) / (max_mileage - min_mileage)

# Normalize Age (0 to 1)
min_age, max_age = X_age.min(), X_age.max()
X_age_norm = (X_age - min_age) / (max_age - min_age)

# Normalize Price (0 to 1) - so the model learns on small numbers
min_price, max_price = Y_price.min(), Y_price.max()
Y_price_norm = (Y_price - min_price) / (max_price - min_price)

print("--- Data Normalized (0 to 1 scale) ---")
print(f"Mileage: {X_mileage_norm}")
print(f"Age:     {X_age_norm}")
print(f"Price:   {Y_price_norm}")

# --- 3. The Model (Multiple Linear Regression) ---
class CarPriceModel:
    def __init__(self, num_features):
        # Weights for each feature (Mileage, Age)
        self.weights = np.random.rand(num_features) * 0.5
        self.bias = random.uniform(-0.5, 0.5)
        
    def forward(self, x_mileage, x_age):
        # Prediction = (w1 * mileage) + (w2 * age) + bias
        # Note: We expect w1 and w2 to be NEGATIVE (price drops as mileage/age goes up)
        return (self.weights[0] * x_mileage) + (self.weights[1] * x_age) + self.bias
    
    def train_step(self, x_m, x_a, y_true, learning_rate):
        y_pred = self.forward(x_m, x_a)
        error = y_true - y_pred
        
        # Update Weights
        # Since error = target - prediction, if prediction is too high, error is negative
        # We want to decrease weight if prediction is too high (for positive inputs)
        self.weights[0] += learning_rate * error * x_m
        self.weights[1] += learning_rate * error * x_a
        self.bias += learning_rate * error
        
        return abs(error)

# --- 4. Training ---
learning_rate = 0.5 # Higher LR works well with normalized data
epochs = 2000

model = CarPriceModel(num_features=2)

print(f"\n--- Starting Training ---")
print(f"Initial Weights: {model.weights}, Bias: {model.bias:.4f}")

for epoch in range(epochs):
    total_error = 0
    for i in range(len(X_mileage_norm)):
        m = X_mileage_norm[i]
        a = X_age_norm[i]
        p = Y_price_norm[i]
        total_error += model.train_step(m, a, p, learning_rate)
    
    if epoch % 500 == 0:
        # Test prediction for a mid-range car (50% mileage, 50% age)
        test_pred = model.forward(0.5, 0.5)
        print(f"Epoch {epoch}: Avg Error: {total_error/len(data):.4f} | Pred (50/50): {test_pred:.2f}")

print(f"\n--- Training Complete ---")
print(f"Final Weights: {model.weights}")
print(f"Final Bias: {model.bias:.4f}")

# --- 5. Real World Test (Denormalize) ---
print("\n--- Final Predictions ---")
print(f"{'Mileage':<8} | {'Age':<4} | {'Predicted Price':<15} | {'Actual Price':<12} | {'Status'}")
print("-" * 60)

# Test with a NEW car not in the training set
test_cases = [
    {"mileage": 40000, "age": 2},
    {"mileage": 100000, "age": 6},
    {"mileage": 15000, "age": 1}
]

for car in test_cases:
    # 1. Normalize Input
    m_norm = (car["mileage"] - min_mileage) / (max_mileage - min_mileage)
    a_norm = (car["age"] - min_age) / (max_age - min_age)
    
    # 2. Predict (Normalized)
    pred_norm = model.forward(m_norm, a_norm)
    
    # 3. Denormalize Output (Back to Dollars)
    # Formula: Value = (Norm * Range) + Min
    pred_price = (pred_norm * (max_price - min_price)) + min_price
    
    # Calculate "Actual" based on the hidden rule we used to generate data
    # Price = 30000 - (0.1 * mileage) - (1000 * age)
    actual_price = 30000 - (0.1 * car["mileage"]) - (1000 * car["age"])
    
    status = "OK" if abs(pred_price - actual_price) < 1000 else "OFF"
    
    print(f"{car['mileage']:<8} | {car['age']:<4} | ${pred_price:<14,.0f} | ${actual_price:<11,.0f} | {status}")

# --- 6. Interpret the Weights ---
print("\n--- Model Interpretation ---")
print(f"Weight for Mileage: {model.weights[0]:.4f}")
print(f"  -> Negative? {model.weights[0] < 0} (Good! Price drops as mileage increases)")
print(f"Weight for Age: {model.weights[1]:.4f}")
print(f"  -> Negative? {model.weights[1] < 0} (Good! Price drops as age increases)")