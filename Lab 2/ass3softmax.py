import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

#Load data
df = pd.read_csv("data.csv", sep = ';')

#Encode labels, Dropout = 0, Enrolled = 1, Graduate = 2
mapping = {'Dropout': 0, 'Enrolled': 1, 'Graduate': 2}
y = df['Target'].map(mapping)

#Split data
X = df.drop('Target', axis=1)
y = df['Target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

#Scale data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#Softmax Regression model
model = LogisticRegression(
    multi_class='multinomial',  # enable softmax
    solver='lbfgs',             # recommended solver for multinomial
    max_iter=1000,
    random_state=42
)
model.fit(X_train_scaled, y_train)

#Evaluate
y_pred = model.predict(X_test_scaled)
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))
