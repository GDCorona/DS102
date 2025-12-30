import os
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.transform import resize
import random

BASE_DIR = "yalefaces"
# Danh sách thư mục các person
person_folders = [f"person-{i}" for i in range(1, 16)]

def standarizing(images: np.ndarray) -> np.ndarray:
    # Chuẩn hóa về trung bình 0 và độ lệch chuẩn 1
    images = (images - images.mean()) / images.std()
    return images.astype(np.float32)

# Load dữ liệu training
data = {}
for person_folder in person_folders:
    folder_path = os.path.join(BASE_DIR, person_folder)
    if not os.path.exists(folder_path): continue

    person_images = []
    for filename in os.listdir(folder_path):
        image = imread(os.path.join(folder_path, filename))
        # Resize về kích thước thống nhất (64, 80)
        image_res = resize(image, (64, 80), anti_aliasing=True)
        person_images.append(image_res.flatten()) # Duỗi phẳng thành vector

    data[person_folder] = np.array(person_images)

# Gộp tất cả ảnh lại để tính PCA
all_images = np.vstack([data[person] for person in data])
all_images = standarizing(all_images)

class PrincipleComponentsAnalysis:
    def __init__(self, d_principle: int = 1):
        self.d_principle = d_principle
        self._W = None
        self.mean_vector = None

    def fit(self, X: np.ndarray):
        # Calculate mean and subtract it from data
        self.mean_vector = np.mean(X, axis=0)
        X_centered = X - self.mean_vector

        # Calculate the covariance matrix directly in the feature space
        # np.cov(X_centered, rowvar=False) assumes columns are features
        cov = np.cov(X_centered, rowvar=False)

        # Use np.linalg.eigh for symmetric matrices for better stability
        eigenvalues, eigenvectors = np.linalg.eigh(cov)

        # Sort eigenvalues and eigenvectors in descending order
        idx = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # Store the projection matrix W (eigenvectors corresponding to d_principle largest eigenvalues)
        # Shape will be (num_features, d_principle)
        self._W = eigenvectors[:, :self.d_principle]

    def transform(self, X: np.ndarray) -> np.ndarray:
        # Subtract the mean before projecting
        X_centered = X - self.mean_vector
        # Project data onto the new subspace
        return np.dot(X_centered, self._W)

# Initialize and train PCA with the corrected class
PCA = PrincipleComponentsAnalysis(d_principle=20)
PCA.fit(all_images)

eigenfaces = PCA._W.T.reshape(20, 64, 80)
plt.figure(figsize=(12, 10))
for ith, eigenface in enumerate(eigenfaces):
    plt.subplot(4, 5, ith + 1)
    plt.imshow(eigenface, cmap='gray')
    plt.axis('off')
    plt.title(f"PC {ith+1}")
plt.tight_layout()
plt.show()

# Tính trung bình vector đã chiếu của mỗi người
projected_mean_faces = {}
for person in data:
    # Lấy ảnh của từng người, chiếu vào không gian PCA rồi tính trung bình
    person_data_std = (data[person] - all_images.mean()) / all_images.std()
    projected = PCA.transform(person_data_std) # Shape: (d_principle, n_images)
    projected_mean_faces[person] = np.mean(projected, axis=1)

def Euclide_distance(X, Y):
    return np.sqrt(np.sum((X - Y)**2))

# Giả sử bạn có TEST_DIR chứa các ảnh kiểm tra
# (Phần này bạn cần đảm bảo TEST_DIR tồn tại)
predictions = {}
# Ví dụ test trên chính tập train hoặc một tập riêng
for person in data:
    test_img = data[person][0] # Lấy ảnh đầu tiên làm ví dụ test
    test_img_std = (test_img - all_images.mean()) / all_images.std()
    projected_face = PCA.transform(test_img_std.reshape(1, -1))

    closest_distance = np.inf
    closest_person = None

    for p_name, p_mean_vector in projected_mean_faces.items():
        dist = Euclide_distance(p_mean_vector, projected_face.flatten())
        if dist < closest_distance:
            closest_distance = dist
            closest_person = p_name

    predictions[person] = closest_person

# Tính độ chính xác
accuracy = np.mean([1 if k == v else 0 for k, v in predictions.items()])
print(f"Accuracy: {accuracy*100:0.2f}%")
