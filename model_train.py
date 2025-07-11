# model_train.py
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
import pickle

# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Train a Decision Tree Classifier
model = DecisionTreeClassifier()
model.fit(X, y)

# Save model using pickle
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("✅ Model trained and saved as model.pkl")
