import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import keras
from keras import layers

# 1. The Data
emails = [
    "Win a free iPhone now",
    "Hey let's meet for lunch",
    "Congratulations you won a million dollars",
    "Meeting scheduled for 2pm",
    "Free entry to win a car"
]
labels = [1, 0, 1, 0, 1]

# 2. Turn Text into Numbers
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(emails).toarray()

# 3. Split Data
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)

# Ensure dense arrays
X_train = np.array(X_train)
X_test = np.array(X_test)
y_train = np.array(y_train)
y_test = np.array(y_test)

print(f"Data ready. X_train shape: {X_train.shape}")

# 4. Build Model
model = keras.Sequential([
    layers.Input(shape=(X_train.shape[1],)),
    layers.Dense(64, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

# 5. Compile
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 6. Train
print("Training model...")
model.fit(X_train, y_train, epochs=10, verbose=1)

# 7. Predict
new_email = ["Free million dollars click here"]
new_email_vec = vectorizer.transform(new_email).toarray()
new_email_vec = np.array(new_email_vec)

prediction = model.predict(new_email_vec)
print(f"\nPrediction Score: {prediction[0][0]:.4f}")
if prediction[0][0] > 0.5:
    print("Result: SPAM 🚫")
else:
    print("Result: Inbox ✅")