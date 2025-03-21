import pickle
from models.features.encode import Features
from sklearn.ensemble import RandomForestClassifier

class Model:
    def __init__(self):
        # Initialize the random forest classifier with a fixed random state for reproducibility.
        self.clf = RandomForestClassifier(random_state=42, n_jobs=-1)
        self.fts = Features()

    def fit(self, X, y):
        """Fit the balanced random forest model to the training data."""
        X = self.fts._fit_preprocess(X)
        self.clf.fit(X, y)
    
    def predict(self, X):
        """Predict labels for the given data."""
        X = self.fts._predict_preprocess(X)
        return self.clf.predict(X)
    
    def predict_proba(self, X):
        """Predict probabilities for the given data."""
        X = self.fts._predict_preprocess(X)
        return self.clf.predict_proba(X)[:, 1]
    
    def refit(self, X, y):
        """Refit the model to the given data."""
        self.fit(X, y)
    
    def log_params(self):
        """Log the hyperparameters of the model."""
        return self.clf.get_params()
    
    def save(self, path):
        """Save the model to the given path."""
        with open(path, "wb") as f:
            pickle.dump((self.clf, self.fts), f)