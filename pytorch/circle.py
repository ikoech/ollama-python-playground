# 1. Import the tools we need "make_circles" is the magic tool to draw the circles
from sklearn.datasets import make_circles # "matplotlib" is the tool to draw the picture
import matplotlib.pyplot as plt

# 2. Create the data (The "Magic Box" step) We want 1000 dots
n_samples = 1000

# Generate the circles
# noise=0.03 makes the dots wobble a little bit so it's not too perfect
# random_state=42 makes sure we get the same picture every time we run this
X, y = make_circles(n_samples=n_samples,
                    noise=0.03, 
                    random_state=42)

# 3. Let's see what the data looks like! We use a scatter plot to draw the dots
plt.figure(figsize=(8, 6)) # Make the picture a nice size

plt.scatter(x=X[:, 0],      # Use the first number for Left/Right position
            y=X[:, 1],      # Use the second number for Up/Down position
            c=y,            # Color the dots based on their label (0 or 1)
            cmap=plt.cm.RdYlBu) # Use the Red-Yellow-Blue color palette

# Add a title and labels so we know what we are looking at
plt.title("Circle Dataset: Red Inside, Blue Outside")
plt.xlabel("Feature 1 (X1)")
plt.ylabel("Feature 2 (X2)")

# Show the picture!
plt.show()

# 4. Let's check the shapes (Just to be sure)
print(f"Number of samples (dots): {len(X)}")
print(f"Shape of X (locations): {X.shape}")
print(f"Shape of y (colors/labels): {y.shape}")
print(f"First 5 locations: {X[:5]}")
print(f"First 5 labels: {y[:5]}")