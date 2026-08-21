import pandas as pd
import numpy as np

# --- 1. Create the Dataset (From your image) ---
data = {
    "Favorite Color": ["Blue", "Red", "Green", "Blue", "Green", "Green", "Blue", "Red"],
# Wait, image has 8 rows. Let's map them carefully.
    # Row 1: Blue, 1.77, 1
    # Row 2: Red, 1.32, 0
    # Row 3: Green, 1.81, 1
    # Row 4: Blue, 1.56, 0
    # Row 5: Green, 1.64, 1
    # Row 6: Green, 1.61, 0
    # Row 7: Blue, 1.73, 0
}

# Re-creating the table (7 rows shown clearly)
df = pd.DataFrame({
    "Favorite Color": ["Blue", "Red", "Green", "Blue", "Green", "Green", "Blue"],
    "Height (m)": [1.77, 1.32, 1.81, 1.56, 1.64, 1.61, 1.73],
    "Loves Troll 2": [1, 0, 1, 0, 1, 0, 0]
})

print("--- Original Data ---")
print(df)

# --- 2. The Target Encoding Logic ---
def target_encode(dataframe, categorical_column, target_column):
    # 1. Group by the category (e.g., "Blue") and calculate the mean of the target
    encoding_map = dataframe.groupby(categorical_column)[target_column].mean()
    
    # 2. Map the original column to the new encoded values
    # .map() looks up the value in 'encoding_map' and replaces it
    dataframe[categorical_column + "_encoded"] = dataframe[categorical_column].map(encoding_map)
    
    return dataframe, encoding_map

# Apply the encoding
df_encoded, mapping = target_encode(df, "Favorite Color", "Loves Troll 2")

print("\n--- Encoded Data ---")
print(df_encoded[["Favorite Color", "Height (m)", "Loves Troll 2", "Favorite Color_encoded"]])

print("\n--- Encoding Map (How the numbers were derived) ---")
print(mapping)

# --- 3. Train a Simple Model on the Encoded Data ---
# Use "Favorite Color_encoded" and "Height (m)" to predict "Loves Troll 2"
from sklearn.linear_model import LogisticRegression

X = df_encoded[["Height (m)", "Favorite Color_encoded"]]
y = df_encoded["Loves Troll 2"]

model = LogisticRegression()
model.fit(X, y)

# Test prediction if we have a person who likes "Green" and is 1.80m tall? Green -> 0.67 (from our map)
test_input = pd.DataFrame({
    "Height (m)": [1.80],
    "Favorite Color_encoded": [0.67] 
})

prediction = model.predict(test_input)
probability = model.predict_proba(test_input)[0][1]

print("\n--- Prediction Test ---")
print(f"Input: Height=1.80m, Color=Green (Encoded: 0.67)")
print(f"Prediction: {'Loves Troll 2' if prediction[0] == 1 else 'Does NOT love Troll 2'}")
print(f"Probability: {probability:.2%}")