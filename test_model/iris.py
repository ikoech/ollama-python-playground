import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.models import Sequential  # type: ignore[reportMissingModuleSource]
from tensorflow.keras.layers import Dense  # type: ignore[reportMissingModuleSource]

# 1. Load the Iris dataset
iris = load_iris()
X = iris.data          # The 4 features: sepal length, sepal width, petal length, petal width
y = iris.target        # The labels: 0, 1, or 2 (representing the 3 species)

# 2. Preprocess the labels (One-Hot Encoding)
# Neural networks often work better with labels like [1, 0, 0] instead of just 0
encoder = OneHotEncoder(sparse_output=False)
y_reshaped = y.reshape(-1, 1)
y_encoded = encoder.fit_transform(y_reshaped)

# 3. Split the data into training and testing sets
# 80% for training, 20% for testing
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# 4. Build the Neural Network Model
model = Sequential()
# Input layer: expects 4 values (the 4 flower measurements)
model.add(Dense(10, activation="relu", input_shape=(4,)))
# Hidden layers to process the data
model.add(Dense(8, activation="relu"))
model.add(Dense(8, activation="relu"))
model.add(Dense(8, activation="relu"))
# Output layer: 3 neurons (one for each flower species) with softmax to get probabilities
model.add(Dense(3, activation="softmax"))

# 5. Compile the model
# 'categorical_crossentropy' is used for multi-class classification
# 'adam' is a common optimizer that helps the model learn
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

# 6. Train the model
# We run through the training data 50 times (epochs)
history = model.fit(X_train, y_train, epochs=50, validation_split=0.1, verbose=1)

# 7. Evaluate the model on the test set
loss, accuracy = model.evaluate(X_test, y_test)

# 8. Print the final accuracy
print(f"\nFinal Test Accuracy: {accuracy:.4f}")

# Optional: Make a prediction on a new flower
# Example: [6.3, 3.3, 4.7, 1.6]
new_flower = np.array([[6.3, 3.3, 4.7, 1.6]])
prediction = model.predict(new_flower)
predicted_class = np.argmax(prediction)
species_names = iris.target_names[predicted_class]

print(f"\nPrediction for new flower: {species_names}")
print(f"Confidence scores: {prediction[0]}")