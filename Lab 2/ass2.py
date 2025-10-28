import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
#read data
df = pd.read_csv('data.csv', sep=';')
#encode data
def encode_target(value):
    if value == "Enrolled": 
        return 0
    elif value == "Dropout": 
        return 1
    return 2
df["Target"] = df["Target"].apply(encode_target)
#normalize data
continuous_cols = [
    "Previous qualification (grade)",
    "Admission grade",
    "Unemployment rate",
    "Inflation rate",
    "GDP"
]
for col in continuous_cols:
    df[col] = (df[col] - df[col].mean()) / df[col].std()
for col in df.columns:
    if col not in continuous_cols and col != "Target":
        df[col] = df[col] / df[col].max()

X = df.drop(columns=['Target']).values
y = df["Target"].values
# split data
size = X.shape[0]
indices = np.arange(size)
np.random.seed(42)
np.random.shuffle(indices)
split = int(0.8 * size)
train_idx, test_idx = indices[:split], indices[split:]
X_train, X_test = X[train_idx], X[test_idx]
y_train, y_test = y[train_idx], y[test_idx]
#encode to one-hot vectors
y_train_encoded = np.zeros((y_train.shape[0], 3))
y_train_encoded[y_train == 0] = np.array([1, 0, 0])
y_train_encoded[y_train == 1] = np.array([0, 1, 0])
y_train_encoded[y_train == 2] = np.array([0, 0, 1])

def softmax(z):
    z = np.where(z == 0, 1e-15, z)
    return np.exp(z) / np.sum(np.exp(z), axis = 1, keepdims=True)
def loss_fn(y, y_hat):
    return -np.mean(y * np.log(y_hat + 1e-5))
def accuracy(y, y_hat):
    return np.mean(y == y_hat)
def predict(X, w, b):
    z = np.matmul(X, w) + b
    y_hat = softmax(z)
    return y_hat
#training loop
def fit(X, y, lr, epochs):
    m = X.shape[0]
    n = X.shape[1]
    n_classes = y.shape[1]
    losses = [] #track loss
    w = np.zeros((n, n_classes))
    b = np.zeros((1, n_classes))
    for epoch in range(epochs):
        #forward
        y_hat = predict(X, w, b)
        #backward
        loss = loss_fn(y, y_hat)
        losses.append(loss)  # save loss
        diff = y_hat - y
        gradient = (1 / m) * np.matmul(X.T, diff)
        w -= lr * gradient
        db = (1 / m) * np.sum(diff, axis=0, keepdims=True)
        b -= lr * db
        # print progress every 100 epochs
        if (epoch+1) % 100 == 0 or epoch == 0:
            acc = accuracy(np.argmax(y, axis=1), np.argmax(y_hat, axis=1)) #convert to labels
            print(f"Epoch {epoch+1}/{epochs} - Loss: {loss:.4f}, Train accuracy = {acc*100:.4f}")
    return w, b, losses
#evaluate
w, b, losses = fit(X_train, y_train_encoded, 0.1, 1000)
#predict
y_train_pred = predict(X_train, w, b)
y_test_pred = predict(X_test, w, b)
train_acc = accuracy(y_train, np.argmax(y_train_pred, axis=1))
test_acc = accuracy(y_test, np.argmax(y_test_pred, axis=1))
print(f"\nTraining Accuracy: {train_acc*100:.2f}%")
print(f"Testing Accuracy:  {test_acc*100:.2f}%")
#plot loss
plt.plot(losses)
plt.title("Training Loss over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.show()