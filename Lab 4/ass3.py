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
        # 4. Sort in descending order
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
    
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "yalefaces")

def load_data(base_path, is_test=False):
    X = []
    y = []
    image_shape = None

    if not is_test: 
        # Training data
        for person in sorted(os.listdir(base_path)):
            person_path = os.path.join(base_path, person)
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
                # Append flattened image & label
                X.append(img.flatten())
                y.append(person)
    else:
        # Testing data
        for file in sorted(os.listdir(base_path)):
            # Load image in grayscale
            img_path = os.path.join(base_path, file)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if image_shape is None:
                image_shape = img.shape
            # Label is before first dot: person-3.happy → person-3
            label = file.split('.')[0]
            # Append flattened image & label
            X.append(img.flatten())
            y.append(label)
    return np.array(X), np.array(y), image_shape

# Load data
X_train, y_train, image_shape = load_data(DATASET_PATH, is_test=False)
X_test, y_test, _ = load_data(os.path.join(DATASET_PATH, "test"), is_test=True)
# Training
pca = PCA(n_components=20)
pca.fit(X_train)
X_train_pca = pca.transform(X_train)
# Testing
X_test_pca = pca.transform(X_test)
# Classification (nearest neighbor)
y_pred = []
for test_face in X_test_pca:
    distances = np.linalg.norm(X_train_pca - test_face, axis=1)
    nearest_idx = np.argmin(distances)
    y_pred.append(y_train[nearest_idx])
# Print accuracy
accuracy = np.mean(np.array(y_pred) == np.array(y_test))
print(f"Accuracy: {accuracy*100:.2f}%")





