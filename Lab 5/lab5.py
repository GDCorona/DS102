import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture

# 1. Load image
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
img_path = os.path.join(BASE_DIR, "zebra.jpg")

img = cv2.imread(img_path)
if img is None:
    raise FileNotFoundError("Image not loaded")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) #OpenCV loads images in BGR, Matplotlib expects RGB
h, w, c = img.shape

# 2. Reshape image to (N, 3)
X = img.reshape(-1, 3).astype(np.float32)  #(h × w, 3)

# 3. Fit Gaussian Mixture Model
gmm = GaussianMixture(
    n_components=2,  # foreground + background
    covariance_type='full',
    random_state=42
)
gmm.fit(X)
# Predict component for each pixel
labels = gmm.predict(X)

# 4. Identify background component
# background usually occupies more pixels
unique, counts = np.unique(labels, return_counts=True)
background_label = unique[np.argmax(counts)]

# 5. Create mask
mask = (labels != background_label).astype(np.uint8)
mask = mask.reshape(h, w)

# 6. Apply mask
foreground = img.copy()
foreground[mask == 0] = [0, 0, 0]   # black background

# 7. Visualization
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.title("Original")
plt.imshow(img)
plt.axis("off")

plt.subplot(1, 3, 2)
plt.title("Foreground Mask")
plt.imshow(mask, cmap="gray")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.title("Background Removed")
plt.imshow(foreground)
plt.axis("off")

plt.tight_layout()
plt.show()
