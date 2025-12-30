import numpy as np

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
        # 2. Compute covariance matrix
        cov_matrix = np.cov(X_centered, rowvar=False) #Columns = variables (features) Rows = observations (samples)
        # 3. Eigen decomposition
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        # 4. Sort eigenvalues & eigenvectors in descending order
        sorted_idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[sorted_idx]
        eigenvectors = eigenvectors[:, sorted_idx]
        # 5. Select top k components
        self.components_ = eigenvectors[:, :self.n_components]
        self.explained_variance_ = eigenvalues[:self.n_components]
    # Apply PCA to new data
    def transform(self, X):
        X_centered = X - self.mean_
        return np.dot(X_centered, self.components_)
    
# Create sample data
np.random.seed(42)
X = np.random.randn(100, 3)

# Apply PCA
pca = PCA(n_components=2)
pca.fit(X)
X_reduced = pca.transform(X)

print("Reduced shape:", X_reduced.shape)
print("Explained variance:", pca.explained_variance_)
