import numpy as np
import pandas as pd
import random

# 1. The data(Movie dataset)
# Features: Genre(Text), Userrating(1-5)
# Target: WillLie(0 = No, 1 = Yes)

data = [
    {"genre": "Action", "rating": 4, "like": 1},
    {"genre": "Comedy", "rating": 3, "like": 0},
    {"genre": "Drama", "rating": 2, "like": 0},
    {"genre": "Action", "rating": 5, "like": 1},
    {"genre": "Comedy", "rating": 5, "like": 1},
    {"genre": "Drama", "rating": 4, "like": 1},
    {"genre": "Action", "rating": 3, "like": 0},
    {"genre": "Comedy", "rating": 4, "like": 1},
    {"genre": "Drama", "rating": 1, "like": 0},
    {"genre": "Action", "rating": 5, "like": 1}
]

df = pd.DataFrame(data)
# 2. Target encoding (The key step)
# Calculate the average 'like' rate for each genre
encoding_map = df.groupby('genre')['like'].mean()

print("--- Target Encoding Map ---")
print(encoding_map)
print(f"-> Action is encoded as: {encoding_map['Action']:.2f}")
print(f"-> Comedy is encoded as: {encoding_map['Comedy']:.2f}")
print(f"-> Drama is encoded as: {encoding_map['Drama']:.2f}")

# Replace the text 'genre' with the calculated number
df['genre_encoded'] = df['genre'].map(encoding_map)

# Normalize the 'rating' column (1-5 -> 0-1)
min_r, max_r = df['rating'].min(), df['rating'].max()
df['rating_norm'] = (df['rating'] - min_r) / (max_r- min_r)

print("\n--- Encoded Data Ready for Model ---")
print(df[['genre', 'genre_encoded', 'rating_norm', 'like']])

# 3. The model (Logistic regression)
class MovieRecommender:
    def __init__(self):
        self.w_genre = random.uniform(-0.5, 0.5)
        self.w_rating = random.uniform(-0.5, 0.5)
        self.bias = random.uniform(-0.5, 0.5)

    def sigmoid(self, z):
        return 1/(1 + np.exp(-z))

    def forward(self, g, r):
        z = (self.w_genre * g) + (self.w_rating * r) + self.bias
        return self.sigmoid(z)

    def train_step(self, g, r, y_true, lr):
        y_pred = self.forward(g, r)
        error = y_pred - y_true
        
        self.w_genre -= lr * error * g
        self.w_rating -= lr * error * r
        self.bias -= lr * error
        
        return abs(error)  
# 4. Training ---
learning_rate = 0.5
epochs = 1000

model = MovieRecommender()

print(f"\n--- Training Movie Recommender ---")

for epoch in range(epochs):
    total_error = 0
    for i in range(len(df)):
        g = df.loc[i, 'genre_encoded']
        r = df.loc[i, 'rating_norm']
        l = df.loc[i, 'like']
        total_error += model.train_step(g, r, l, learning_rate)
    
    if epoch % 200 == 0:
        # Test: High rating Action movie (Encoded ~0.8, Rating 1.0)
        prob_action = model.forward(0.8, 1.0)
        # Test: Low rating Drama movie (Encoded ~0.3, Rating 0.0)
        prob_drama = model.forward(0.3, 0.0)
        
        print(f"Epoch {epoch}: Avg Error: {total_error/len(df):.4f}")
        print(f"  -> Action (High Rating): {prob_action:.2%} Liked")
        print(f"  -> Drama (Low Rating):   {prob_drama:.2%} Liked")
        print("-" * 40)

print(f"\n--- Training Complete ---")
print(f"Weight for Genre: {model.w_genre:.4f}")
print(f"Weight for Rating: {model.w_rating:.4f}")

# --- 5. Final Test ---
print("\n--- Final Predictions ---")
print(f"{'Genre':<10} | {'Rating':<6} | {'Encoded':<8} | {'Prob Like':<12} | {'Prediction'}")
print("-" * 50)

test_movies = [
    {"genre": "Action", "rating": 5},
    {"genre": "Comedy", "rating": 3},
    {"genre": "Drama", "rating": 4},
    {"genre": "Action", "rating": 2}
]

for movie in test_movies:
    # Encode genre
    g_enc = encoding_map[movie["genre"]]
    # Normalize rating
    r_norm = (movie["rating"] - min_r) / (max_r - min_r)
    
    # Predict
    prob = model.forward(g_enc, r_norm)
    prediction = "LIKE" if prob > 0.5 else "SKIP"
    
    print(f"{movie['genre']:<10} | {movie['rating']:<6} | {g_enc:<8.2f} | {prob:<12.2%} | {prediction}")