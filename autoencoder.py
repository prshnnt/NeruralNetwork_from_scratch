import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# 1. Load and prepare dataset (64 features per sample)
X, _ = load_digits(return_X_y=True)
X_train, X_test = train_test_split(X, test_size=0.2, random_state=42)

# Scale features to [0, 1] for better neural network convergence
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 2. Define the Autoencoder architecture
# Encoder: 64 -> 32 -> 8 (bottleneck)
# Decoder: 8 -> 32 -> 64
autoencoder = MLPRegressor(
    hidden_layer_sizes=(32, 8, 32),
    activation='logistic',
    solver='adam',
    max_iter=1000,
    random_state=42
)

# 3. Train the model to reconstruct its own input
autoencoder.fit(X_train_scaled, X_train_scaled)

# 4. Reconstruct test data and evaluate error
reconstructed = autoencoder.predict(X_test_scaled)
mse = mean_squared_error(X_test_scaled, reconstructed)
print(f"Reconstruction MSE: {mse:.4f}")

def encode(model, X):
    """Forward pass through the encoder layers up to the bottleneck."""
    # Identify the bottleneck layer index (middle of hidden_layer_sizes)
    bottleneck_idx = len(model.hidden_layer_sizes) // 2
    
    activation = X
    for i in range(bottleneck_idx + 1):
        activation = np.dot(activation, model.coefs_[i]) + model.intercepts_[i]
        # Apply ReLU activation
        activation = np.maximum(0, activation)
    return activation

# Transform 64D test samples into 8D compressed representations
compressed_X_test = encode(autoencoder, X_test_scaled)
print("Original shape:", X_test_scaled.shape)      # (360, 64)
print("Compressed shape:", compressed_X_test.shape)  # (360, 8)
# print("Compressed :", compressed_X_test)  # (360, 8)

# Number of digits to display
n_digits = 10

fig, axes = plt.subplots(3, n_digits, figsize=(15, 5))

for i in range(n_digits):
    # Reshape 64D vectors back to 8x8 images
    orig_img = X_test_scaled[i].reshape(8, 8)
    recon_img = reconstructed[i].reshape(8, 8)
    diff_img = np.abs(orig_img - recon_img)

    # Row 1: Original Digits
    axes[0, i].imshow(orig_img, cmap='gray_r', vmin=0, vmax=1)
    axes[0, i].axis('off')
    if i == 0:
        axes[0, i].set_title("Original", loc='left', fontsize=12, fontweight='bold')

    # Row 2: Reconstructed Digits
    axes[1, i].imshow(recon_img, cmap='gray_r', vmin=0, vmax=1)
    axes[1, i].axis('off')
    if i == 0:
        axes[1, i].set_title("Reconstructed", loc='left', fontsize=12, fontweight='bold')

    # Row 3: Absolute Pixel Difference
    axes[2, i].imshow(diff_img, cmap='hot', vmin=0, vmax=1)
    axes[2, i].axis('off')
    if i == 0:
        axes[2, i].set_title("Difference (Error)", loc='left', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.show()
