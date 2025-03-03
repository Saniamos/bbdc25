import pickle
from sklearn.calibration import LabelEncoder
from sklearn.dummy import DummyClassifier

class Model:
    def __init__(self):
        # Initialize the dummy classifier with a fixed random state for reproducibility.
        # Using 'stratified' strategy which generates predictions based on training set's class distribution
        self.clf = DummyClassifier(strategy='stratified', random_state=42)
        self.action_encoder = LabelEncoder()
        self.external_type_encoder = LabelEncoder()

    def _fit_preprocess(self, X):
        """Preprocess the input data."""
        X.drop(columns=["External", "AccountID", "Action", "External_Type"], inplace=True)
        return X
    
    def _predict_preprocess(self, X):
        """Preprocess the input data."""
        X.drop(columns=["External", "AccountID", "Action", "External_Type"], inplace=True)
        return X

    def fit(self, X, y):
        """Fit the dummy model to the training data."""
        X = self._fit_preprocess(X)
        self.clf.fit(X, y)
    
    def predict(self, X):
        """Predict labels for the given data."""
        X = self._predict_preprocess(X)
        return self.clf.predict(X)
    
    def save(self, path):
        """Save the model to the given path."""
        with open(path, "wb") as f:
            pickle.dump((self.clf, self.action_encoder, self.external_type_encoder), f)