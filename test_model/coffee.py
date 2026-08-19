import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

# 1. The Training Data (What the AI studies)
# Format: [Price, Caffeine]
X_train = np.array([
    [2, 100],  # Espresso
    [5, 50],   # Latte
    [2, 0],    # Decaf
    [2, 90],   # Another Espresso
    [5, 45],   # Another Latte
    [2, 5]     # Another Decaf
])

# The Answers (Labels): 0=Espresso, 1=Latte, 2=Decaf
y_train = np.array([0, 1, 2, 0, 1, 2])

# 2. Build the Brain (Neural Network)
model = Sequential()
# Input: 2 numbers (Price, Caffeine) -> Hidden: 4 neurons to learn patterns
model.add(Dense(4, activation='relu', input_shape=(2,)))
# Output: 3 choices (Espresso, Latte, Decaf)
model.add(Dense(3, activation='softmax'))

# 3. Teach the Brain (Compile)
model.compile(optimizer=Adam(learning_rate=0.1), 
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])

# 4. Train (Let it study the examples)
print("Teaching the AI...")
model.fit(X_train, y_train, epochs=500, verbose=0)

# 5. Test it!
print("\n--- Testing ---")

# Test Case A: Cheap ($2) + Strong Caffeine (95) -> Should be Espresso
test_coffee_a = np.array([[2, 95]])
pred_a = model.predict(test_coffee_a)
result_a = np.argmax(pred_a)
print(f"Input: Price $2, Caffeine 95 -> Guess: {['Espresso', 'Latte', 'Decaf'][result_a]}")

# Test Case B: Expensive ($5) + Medium Caffeine (55) -> Should be Latte
test_coffee_b = np.array([[5, 55]])
pred_b = model.predict(test_coffee_b)
result_b = np.argmax(pred_b)
print(f"Input: Price $5, Caffeine 55 -> Guess: {['Espresso', 'Latte', 'Decaf'][result_b]}")

# Test Case C: Cheap ($2) + Zero Caffeine (0) -> Should be Decaf
test_coffee_c = np.array([[2, 0]])
pred_c = model.predict(test_coffee_c)
result_c = np.argmax(pred_c)
print(f"Input: Price $2, Caffeine 0 -> Guess: {['Espresso', 'Latte', 'Decaf'][result_c]}")