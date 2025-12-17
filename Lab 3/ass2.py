import numpy as np
import matplotlib.pyplot as plt
np.random.seed(42)

# Parameters
n1 = 50
n2 = 20
n3 = 1000
Sigma = np.array([[1, 0],
                  [0, 1]])

means = np.array([
    [2, 2],
    [8, 3],
    [3, 6]
])

# Generate data
X1 = np.random.multivariate_normal(means[0], Sigma, n1)
X2 = np.random.multivariate_normal(means[1], Sigma, n2)
X3 = np.random.multivariate_normal(means[2], Sigma, n3)

# Combine dataset
X = np.vstack((X1, X2, X3))
def initialize_centroids(X, k):
    indices = np.random.choice(len(X), k, replace=False)
    return X[indices]


def assign_clusters(X, centroids):
    # Squared Euclidean distance
    distances = np.linalg.norm(X[:, None, :] - centroids[None, :, :], axis=2)
    return np.argmin(distances, axis=1)


def update_centroids(X, labels, k):
    centroids = np.zeros((k, X.shape[1]))
    for i in range(k):
        centroids[i] = X[labels == i].mean(axis=0)
    return centroids

def kmeans_em(X, k, max_iters=100, tol=1e-4):
    centroids = initialize_centroids(X, k)
    print("Initial centroids:")
    print(centroids)
    for i in range(max_iters):
        # E-step
        labels = assign_clusters(X, centroids)
        # M-step
        new_centroids = update_centroids(X, labels, k)
        # Convergence check
        if np.linalg.norm(new_centroids - centroids) < tol:
            print("Converged at iteration:", i)
            break

        centroids = new_centroids

    return centroids, labels

k = 3
centroids, labels = kmeans_em(X, k)

print("Final centroids:")
print(centroids)

plt.figure()

# Plot each cluster
for i in range(k):
    plt.scatter(
        X[labels == i, 0],
        X[labels == i, 1],
        label=f"Cluster {i}"
    )

# Plot centroids
plt.scatter(
    centroids[:, 0],
    centroids[:, 1],
    marker='x',
    s=150,
    linewidths=3,
    label='Centroids'
)

plt.xlabel("x")
plt.ylabel("y")
plt.title("K-means Clustering")
plt.legend()
plt.show()

"""
COMMENT:
Large clusters dominate the objective

Small clusters contribute very little to the total loss

-> K-means is strongly biased toward large clusters and can ignore or absorb small clusters when sizes are highly imbalanced.
"""
