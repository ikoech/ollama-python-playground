import numpy as np
import random

# --- 1. The Data (Weather Dataset) ---
# Features: [Humidity (%), Wind Speed (km/h)]
# Target: Rain (0 = No, 1 = Yes)
# Pattern: Rain happens if Humidity > 70% OR Wind > 20km/h

data = [
    {"humidity": 40, "wind": 10, "rain": 0},
    {"humidity": 85, "wind": 25, "rain": 1},
    {"humidity": 50, "wind": 15, "rain": 0},
    {"humidity": 90, "wind": 30, "rain": 1},
    {"humidity": 75, "wind": 22, "rain": 1},
    {"humidity": 60, "wind": 18, "rain": 0}, # Borderline, but let's say no
    {"humidity": 95, "wind": 10, "rain": 1}, # High humidity alone causes rain
    {"humidity": 30, "wind": 5, "rain": 0}
]

# Extract features
X_humidity = np.array([d["humidity"] for d in data])
X_wind = np.array([d["wind"] for d in data])
Y_rain = np.array([d["rain"] for d in data])

# --- 2. Data Normalization ---
# Scale 0-100 (Humidity) and 0-30 (Wind) down to 0-1
min_h, max_h = X_humidity.min(), X_humidity.max()
min_w, max_w = X_wind.min(), X_wind.max()

X_h_norm = (X_humidity - min_h) / (max_h - min_h)
X_w_norm = (X_wind - min_w) / (max_w - min_w)

print("--- Data Normalized ---")
print(f"Humidity Norm: {X_h_norm}")
print(f"Wind Norm:     {X_w_norm}")

# --- 3. The Model (Logistic Regression with Sigmoid) ---
class RainModel:
    def __init__(self, num_features):
        self.weights = np.random.rand(num_features) * 0.5
        self.bias = random.uniform(-0.5, 0.5)
        
    def sigmoid(self, z):
        # The magic function: Squashes any number into 0.0 to 1.0
        # Formula: 1 / (1 + e^-z)
        return 1 / (1 + np.exp(-z))
    
    def forward(self, h, w):
        # Linear calculation
        z = (self.weights[0] * h) + (self.weights[1] * w) + self.bias
        # Convert to Probability
        return self.sigmoid(z)
    
    def train_step(self, h, w, y_true, learning_rate):
        y_pred = self.forward(h, w)
        
        # Error for Sigmoid (Logistic Regression)
        error = y_pred - y_true
        
        # Update weights
        self.weights[0] -= learning_rate * error * h
        self.weights[1] -= learning_rate * error * w
        self.bias -= learning_rate * error
        
        return abs(error)

# --- 4. Training ---
learning_rate = 1.0 # Logistic regression often needs a higher LR
epochs = 2000

model = RainModel(num_features=2)

print(f"\n--- Starting Training ---")

for epoch in range(epochs):
    total_error = 0
    for i in range(len(data)):
        h = X_h_norm[i]
        w = X_w_norm[i]
        r = Y_rain[i]
        total_error += model.train_step(h, w, r, learning_rate)
    
    if epoch % 500 == 0:
        # Check probability for a "Stormy" scenario (High H, High W)
        # Normalized: 0.8, 0.8
        prob_storm = model.forward(0.8, 0.8)
        # Check probability for "Calm" scenario (Low H, Low W)
        prob_calm = model.forward(0.2, 0.2)
        
        print(f"Epoch {epoch}: Avg Error: {total_error/len(data):.4f}")
        print(f"  -> Storm (0.8, 0.8): {prob_storm:.2%} chance of rain")
        print(f"  -> Calm (0.2, 0.2):  {prob_calm:.2%} chance of rain")
        print("-" * 40)

print(f"\n--- Training Complete ---")
print(f"Weights: {model.weights}")
print(f"Bias: {model.bias:.4f}")

# --- 5. Real World Test ---
print("\n--- Final Predictions ---")
print(f"{'Humidity':<10} | {'Wind':<6} | {'Prob Rain':<12} | {'Prediction':<10} | {'Actual'}")
print("-" * 50)

# Test cases
test_cases = [
    {"h": 45, "w": 12, "actual": 0}, # Sunny day
    {"h": 92, "w": 28, "actual": 1}, # Storm
    {"h": 78, "w": 15, "actual": 1}, # Humid day
    {"h": 30, "w": 5, "actual": 0}   # Dry day
]

for t in test_cases:
    # Normalize input
    h_norm = (t["h"] - min_h) / (max_h - min_h)
    w_norm = (t["w"] - min_w) / (max_w - min_w)
    
    # Get probability
    prob = model.forward(h_norm, w_norm)
    
    # Convert to Yes/No
    prediction = "YES" if prob > 0.5 else "NO"
    status = "OK" if (prediction == "YES" and t["actual"] == 1) or (prediction == "NO" and t["actual"] == 0) else "WRONG"
    
    print(f"{t['h']:<10} | {t['w']:<6} | {prob:<12.2%} | {prediction:<10} | {t['actual']}")