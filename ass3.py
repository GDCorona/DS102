import os
import random
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.transform import resize

BASE_DIR = "yalefaces"

person_folders = [
    "person-1","person-2","person-3","person-4","person-5",
    "person-6","person-7","person-8","person-9","person-10",
    "person-11","person-12","person-13","person-14","person-15"
]
def standardizing(images: np.ndarray) -> np.ndarray:
    images = (images - images.mean()) / images.std()
    return images.astype(np.float32)

data = {}

for person in person_folders:
    images = []
    for filename in os.listdir(os.path.join(BASE_DIR, person)):
        img = imread(os.path.join(BASE_DIR, person, filename))
        img = resize(img, (64, 80), anti_aliasing=True)
        images.append(img.reshape(-1))
    data[person] = np.array(images)


images = np.concatenate(list(data.values()), axis=0)
images = standardizing(images)

class PrincipleComponentsAnalysis:
    def __init__(self, d_principle: int = 20):
        self.d_principle = d_principle

    def fit(self, X: np.ndarray):
        cov = X.T @ X
        eigenvalues, eigenvectors = np.linalg.eigh(cov)

        # sắp xếp giảm dần
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        self.W = eigenvectors[:, :self.d_principle]

    def transform(self, X: np.ndarray):
        return X @ self.W


PCA = PrincipleComponentsAnalysis(d_principle=20)
PCA.fit(images)

eigenfaces = PCA.W.T.reshape(20, 64, 80)

plt.figure(figsize=(10, 10))
for i, face in enumerate(eigenfaces):
    plt.subplot(4, 5, i + 1)
    plt.imshow(face, cmap='gray')
    plt.axis('off')
    plt.title(f"{i+1}")
plt.tight_layout()
plt.show()

projected_mean_faces = {}

for person in data:
    mean_face = data[person].mean(axis=0)
    projected_mean_faces[person] = PCA.transform(mean_face)


TEST_DIR = BASE_DIR
labels = []
test_images = []

for person in person_folders:
    for filename in os.listdir(os.path.join(TEST_DIR, person)):
        img = imread(os.path.join(TEST_DIR, person, filename))
        img = resize(img, (64, 80), anti_aliasing=True)
        test_images.append(img.reshape(-1))
        labels.append(person)

test_images = np.array(test_images)

def euclid_distance(x, y):
    return np.sqrt(np.sum((x - y) ** 2))

predictions = {}

for label, image in zip(labels, test_images):
    proj = PCA.transform(image)

    min_dist = np.inf
    pred_person = None

    for person in projected_mean_faces:
        dist = euclid_distance(proj, projected_mean_faces[person])
        if dist < min_dist:
            min_dist = dist
            pred_person = person

    predictions[label] = pred_person

accuracy = np.mean([label == predictions[label] for label in labels])
print(f"Accuracy: {accuracy*100:.2f}%")

