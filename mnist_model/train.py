import numpy as np
import importlib
from sklearn.model_selection import train_test_split

try:
    plt = importlib.import_module("matplotlib.pyplot")
except ImportError:
    plt = None

# 1. Load the MNIST dataset from the local .npz file
# Ensure 'mnist.npz' is in the working directory or provide the full path.
try:
    data = np.load('mnist.npz')
    
    # Extract arrays from the file
    # The keys in the default MNIST npz file are usually: 'x_train', 'y_train', 'x_test', 'y_test'
    x_train_full = data['x_train']
    y_train_full = data['y_train']
    x_test = data['x_test'] 
    y_test = data['y_test']
    
    print("MNIST dataset loaded successfully from 'mnist.npz'.")
except FileNotFoundError:
    print("Error: 'mnist.npz' file not found. Please ensure the file is in the current directory.")
    # Fallback to keras if the file is missing (optional) from tensorflow import keras
    # (x_train_full, y_train_full), (x_test, y_test) = keras.datasets.mnist.load_data()
except Exception as e:
    print(f"Error loading file: {e}")
    raise

# 2. Split the data (if not already split in the file)
# If the npz file already has them split, we just assign them.
# If it needs a validation set, we can split the training data further.
x_train, x_valid, y_train, y_valid = train_test_split(
    x_train_full, y_train_full, 
    test_size=0.1, 
    random_state=42
)

# 3. Normalize Pixel Values
# Move from the original 0-255 range to a normalized range. From zero to one is better for models.
# Standard approach: Convert to float, divide by 255 (0-1), then shift to -1 to 1.

# Convert to float for precision
x_train = x_train.astype('float32')
x_valid = x_valid.astype('float32')
x_test = x_test.astype('float32')

# Step A: Normalize to 0-1
x_train /= 255.0
x_valid /= 255.0
x_test /= 255.0

# Step B: Shift to -1 to 1 (as implied by "black pixels will be zero" in hour case vs negative in discase)
# Formula: (value - 0.5) * 2
# 0 becomes -1, 0.5 becomes 0, 1 becomes 1
x_train = (x_train - 0.5) * 2.0
x_valid = (x_valid - 0.5) * 2.0
x_test = (x_test - 0.5) * 2.0

print(f"Training shape: {x_train.shape}")
print(f"Pixel value range: [{x_train.min():.2f}, {x_train.max():.2f}]")

# 4. Visualize a Sample Images, looking at "number five" and pixel structures.
sample_idx = 0
sample_img = x_train[sample_idx]
sample_label = y_train[sample_idx]

if plt is not None:
    plt.figure(figsize=(4, 4))
    plt.imshow(sample_img, cmap='gray')
    plt.title(f"Sample Image (Label: {sample_label})\nRange: {x_train.min():.2f} to {x_train.max():.2f}")
    plt.axis('off')
    plt.colorbar(label='Pixel Value')
    plt.show()
else:
    print("matplotlib is not installed; skipping visualization.")

# 5. (Optional) Verify dimensions 28x28 pixels (256 columns/rows mentioned as 28x28 in standard MNIST)
print(f"Image dimensions: {x_train[0].shape}")