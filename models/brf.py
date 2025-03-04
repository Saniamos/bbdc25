import pickle
from imblearn.ensemble import BalancedRandomForestClassifier
from models.features.encode import Features

class Model:
    def __init__(self):
        # Initialize the balanced random forest classifier with a fixed random state for reproducibility.
        # This classifier implements a random under-sampling strategy for handling class imbalance
        self.clf = BalancedRandomForestClassifier(random_state=42, n_jobs=-1)
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
        return self.clf.predict_proba(X)
    
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