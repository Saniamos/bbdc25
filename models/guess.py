import pickle
from models.features.simple_remove import Features
from sklearn.dummy import DummyClassifier

class Model:
    def __init__(self):
        # Initialize the dummy classifier with a fixed random state for reproducibility.
        # Using 'stratified' strategy which generates predictions based on training set's class distribution
        self.clf = DummyClassifier(strategy='stratified', random_state=42)
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