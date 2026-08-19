from sklearn.datasets import load_iris
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import numpy as np
# Add this at the very top of your script
import os
os.environ["KERAS_BACKEND"] = "jax"

# 1. Load the Iris dataset
iris = load_iris()
X = iris.data  # The measurements (4 features)
y = iris.target # The labels (0, 1, or 2)

# 2. Prepare the data
# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Build a proper Neural Network
model = Sequential()
# Input layer: 4 features, 8 neurons, ReLU activation (learns patterns)
model.add(Dense(8, activation='relu', input_shape=(4,)))
# Output layer: 3 neurons (one for each flower type), Softmax (gives probability)
model.add(Dense(3, activation='softmax'))

# 4. Compile the model
model.compile(optimizer=Adam(learning_rate=0.01), 
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])

# 5. Train the model
print("Training the model...")
model.fit(X_train, y_train, epochs=100, verbose=0)

# 6. Make a prediction with your example data
example_flower = np.array([[5.1, 3.5, 1.4, 0.2]]) # [Sepal Len, Sepal Wid, Petal Len, Petal Wid]
prediction = model.predict(example_flower)

# Find which flower type has the highest probability
predicted_class = np.argmax(prediction)
flower_names = ['Setosa', 'Versicolor', 'Virginica']

print(f"\nPrediction for {example_flower[0]}:")
print(f"Most likely flower: {flower_names[predicted_class]}")