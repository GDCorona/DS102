import os
import numpy as np
import cv2
import matplotlib.pyplot as plt

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
        #Flatten image into a vector
        X.append(img.flatten())

X = np.array(X)
print("Data matrix shape:", X.shape)
# Center the data
mean_face = np.mean(X, axis=0)
X_centered = X - mean_face
# Compute covariance matrix, Shape: (150, 150)
cov_small = np.dot(X_centered, X_centered.T) / (X_centered.shape[0] - 1)
# Eigen decomposition
eigvals, eigvecs_small = np.linalg.eigh(cov_small)
# Sort descending. Largest eigenvalue = most variance
idx = np.argsort(eigvals)[::-1]
eigvals = eigvals[idx]
eigvecs_small = eigvecs_small[:, idx]
#Map eigenvectors to pixel space 
eigvecs = np.dot(X_centered.T, eigvecs_small)
eigvecs = eigvecs / np.linalg.norm(eigvecs, axis=0) # Normalize, makes each eigenface a unit vector

# Keep top 20 components
k = 20
W = eigvecs[:, :k]

# Visualize
plt.figure(figsize=(10, 6))
for i in range(20):
    plt.subplot(4, 5, i + 1)
    eigenface = W[:, i].reshape(image_shape)
    plt.imshow(eigenface, cmap='gray')
    plt.title(f"PC {i+1}")
    plt.axis('off')
plt.tight_layout()
plt.show()

