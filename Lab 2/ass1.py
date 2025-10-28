import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
#read data
df = pd.read_csv('data.csv', sep=';')
#encode data
def encode_target(value):
    if value == "Enrolled" or value == "Dropout":
        return 0
    return 1

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

X = df.drop(columns=['Target']).values
y = df["Target"].values.reshape(-1, 1)
# split data
size = X.shape[0]
indices = np.arange(size)
np.random.seed(42)
np.random.shuffle(indices)
split = int(0.8 * size)
train_idx, test_idx = indices[:split], indices[split:]
X_train, X_test = X[train_idx], X[test_idx]
y_train, y_test = y[train_idx], y[test_idx]
#hyperparameters
m = X_train.shape[0]
n = X_train.shape[1]
w = np.zeros((n, 1))
b = 0
lr = 0.01 
epochs = 100
losses = []
def sigmoid(z):
    return 1 / (1 + np.exp(-z))
def loss_fn(y, y_hat):
    return -np.mean(y * np.log(y_hat + 1e-9) + (1 - y) * np.log(1 - y_hat + 1e-9)) #them 10e-9 de tranh log(0)
def accuracy(y, y_hat): 
    return np.mean(1 - np.abs(y - y_hat))
def predict(X):
    z = np.matmul(X, w) + b
    y_hat = sigmoid(z)
    return y_hat
#training loop
for epoch in range(epochs):
    #forward
    y_hat = predict(X_train)
    #compute loss
    loss = loss_fn(y_train, y_hat)
    losses.append(loss)  # save loss
    #backward
    diff = y_hat - y_train
    gradient = (1 / m) * np.matmul(X_train.T, diff)
    db = (1 / m) * np.sum(diff)
    w -= lr * gradient
    b -= lr * db
    # print progress every 100 epochs
    if (epoch+1) % 100 == 0 or epoch == 0:
        y_pred = (y_hat >= 0.5).astype(int)
        acc = np.mean(y_pred == y_train)
        print(f"Epoch {epoch+1}/{epochs} - Loss: {loss:.4f}, Train accuracy = {acc*100:.4f}")

#testing model
y_pred_prob = predict(X_test)
y_pred = (y_pred_prob >= 0.5).astype(int)
acc = accuracy(y_pred, y_test)
print(f"\nTest Accuracy: {acc*100:.4f}")

#plot loss
plt.figure(figsize=(8, 5))
plt.plot(losses)
plt.title("Training Loss over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.show()