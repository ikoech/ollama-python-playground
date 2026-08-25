# --- FILE: mnist_script.py ---

import numpy as np
from sklearn.model_selection import train_test_split

# --- PART 1: Load Data (From the previous run) ---
data = np.load('mnist.npz')
x_train_full = data['x_train']
y_train_full = data['y_train']
x_test = data['x_test']
y_test = data['y_test']

x_train, x_valid, y_train, y_valid = train_test_split(
    x_train_full, y_train_full, test_size=0.1, random_state=42
)

# Normalize (0-1 then -1 to 1)
x_train = x_train.astype('float32') / 255.0
x_train = (x_train - 0.5) * 2.0
# (Do same for x_valid and x_test)

# --- PART 2: Train Model (Add this part below) ---
import tensorflow as tf
keras = tf.keras
layers = keras.layers

# Flatten data
x_train_flat = x_train.reshape(-1, 784)

# Build Model
model = keras.Sequential([
    layers.Input(shape=(784,)),
    layers.Dense(128, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# Compile
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train
model.fit(x_train_flat, y_train, epochs=5, batch_size=64, verbose=1)