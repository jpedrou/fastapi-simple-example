import pickle

from sklearn.datasets import fetch_openml
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# ==========================================================================
# Load data
# ==========================================================================

X, y = fetch_openml("mnist_784", version=1, return_X_y=True)

# ==========================================================================
# Split into train and test
# ==========================================================================

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# ==========================================================================
# Setup classifier and training
# ==========================================================================

classifier = RandomForestClassifier(n_jobs=-1, random_state=0)

classifier.fit(X_train, y_train)
print(f"Accuracy on train set: {classifier.score(X_train, y_train)}")
print(f"Accuracy on test set: {classifier.score(X_test, y_test)}")

# ==========================================================================
# Save model
# ==========================================================================

with open("model.pkl", "wb") as f:
    pickle.dump(classifier, f)
