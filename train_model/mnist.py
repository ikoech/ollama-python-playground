from keras.datasets import mnist
from keras.utils import to_categorical

# 1. Load Data
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# 2. Pre-processing
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

# Normalize to 0-1
X_train = X_train / 255.0
X_test = X_test / 255.0

# Flatten images (28x28 -> 784)
X_train = X_train.reshape(60000, 784)
X_test = X_test.reshape(10000, 784)

# 3. One-Hot Encode Labels
# This converts labels like 0, 5 into vectors like [1,0...0] or [0,0,0,0,0,1...]
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# --- DEBUG PRINTS (Optional) ---
# print(X_train.shape) # (60000, 784)
# print(y_train[0])    # Shows the one-hot vector for the first image

# 4. Build Model
from keras.models import Sequential
from keras.layers import Dense

model = Sequential([
    Dense(128, activation='relu', input_shape=(784,)),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')
])

# 5. Compile Model
# Note: loss='categorical_crossentropy' is required because it use to_categorical()
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# 6. Train Model
print("Training model...")
model.fit(X_train, y_train, epochs=5, batch_size=64, validation_split=0.1, verbose=1)

# 7. Evaluate
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"\nTest Accuracy: {test_acc:.4f}")