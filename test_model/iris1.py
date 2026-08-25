# Iris
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras.layers import Input, Dense

data = pd.read_csv('iris.csv')

# Features
X = data[["sepal.length","sepal.width","petal.length","petal.width"]]

# Label
y = data["variety"]

encoder = OneHotEncoder(sparse_output=False)
y_array = np.array(y)
y_array = y_array.reshape(-1, 1)
y_encoded = encoder.fit_transform(y_array)

X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=10)

model = Sequential()
model.add(Input(shape=(4,)))

model.add(Dense(10, activation="relu"))
model.add(Dense(8, activation="relu"))
model.add(Dense(8, activation="relu"))
model.add(Dense(8, activation="relu"))
model.add(Dense(8, activation="relu"))

model.add(Dense(3, activation="softmax"))

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

model.fit(X_train, y_train, epochs=50)

loss, accuracy = model.evaluate(X_test, y_test)

print(f"Built-in accuracy: {accuracy}")

# My evaluate (manual calculation)
correct_predictions = 0
for i in range(len(X_test)):
    # Convert pandas Series row to numpy array and reshape
    input_reshaped = np.array(X_test.iloc[i]).reshape(1, 4)
    prediction = model.predict(input_reshaped, verbose=0)  # verbose=0 to suppress output
    
    # Get the predicted class and true class
    predicted_class = np.argmax(prediction[0])
    true_class = np.argmax(y_test[i])
    
    if predicted_class == true_class:
        correct_predictions += 1

my_accuracy = correct_predictions / len(y_test)
print(f"Manual accuracy: {my_accuracy}")

# Make a prediction on a new flower
# prediction = model.predict(np.array([[6.3,3.3,4.7,1.6]]))
# print(prediction)