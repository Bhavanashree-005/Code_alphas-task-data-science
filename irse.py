# Iris Flower Classification using Scikit-learn

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 1. Load the dataset
iris = load_iris()
X = iris.data       # features: sepal length, sepal width, petal length, petal width
y = iris.target     # labels: 0=setosa, 1=versicolor, 2=virginica

# 2. Split into training and test sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3. Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 4. Train a classifier (K-Nearest Neighbors)
model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train, y_train)

# 5. Predictions
y_pred = model.predict(X_test)

# 6. Evaluate the model
print("✅ Model Accuracy:", accuracy_score(y_test, y_pred))
print("\n📊 Classification Report:\n", classification_report(y_test, y_pred, target_names=iris.target_names))
print("\n🔎 Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
