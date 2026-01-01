import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
class PCA:
    def __init__(self, n_components):
        self.n_components = n_components
        self.mean_ = None
        self.components_ = None
        self.explained_variance_ = None
    def fit(self, X):
        # 1. Center the data
        self.mean_ = np.mean(X, axis=0)
        X_centered = X - self.mean_
        # 2. Compute small covariance matrix (eigenfaces trick)
        # Shape: (n_samples, n_samples)
        cov_small = np.dot(X_centered, X_centered.T) / (X_centered.shape[0] - 1)
        # 3. Eigen decomposition
        eigenvals, eigenvecs_small = np.linalg.eigh(cov_small)
        # 4. Sort descending. Largest eigenvalue = most variance
        sorted_idx = np.argsort(eigenvals)[::-1]
        eigenvals = eigenvals[sorted_idx]
        eigenvecs_small = eigenvecs_small[:, sorted_idx]
        # 5. Map eigenvectors to pixel space
        eigenvecs = np.dot(X_centered.T, eigenvecs_small)
        eigenvecs = eigenvecs / np.linalg.norm(eigenvecs, axis=0) # Normalize, makes each eigenface a unit vector
        # 6. Select top k components
        self.components_ = eigenvecs[:, :self.n_components]
        self.explained_variance_ = eigenvals[:self.n_components]
    # Apply PCA to new data
    def transform(self, X):
        X_centered = X - self.mean_
        return np.dot(X_centered, self.components_)

# Load data
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "yalefaces")
X = []
image_shape = None

for person in sorted(os.listdir(DATASET_PATH)):
    person_path = os.path.join(DATASET_PATH, person)
    # Skip test folder
    if person == "test":
        continue
    if not os.path.isdir(person_path):
        continue
    # Load image in grayscale
    for file in os.listdir(person_path):
        img_path = os.path.join(person_path, file)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if image_shape is None:
            image_shape = img.shape
        # Flatten image into a vector
        X.append(img.flatten())

# Apply PCA with 20 components
X = np.array(X)
print("Data matrix shape:", X.shape)
pca = PCA(n_components=20)
pca.fit(X)

# Visualize
plt.figure(figsize=(10, 6))
for i in range(20):
    plt.subplot(4, 5, i + 1)
    eigenface = pca.components_[:, i].reshape(image_shape)
    plt.imshow(eigenface, cmap='gray')
    plt.title(f"PC {i+1}")
    plt.axis('off')
plt.tight_layout()
plt.show()

