import pickle
from sklearn.calibration import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

class Model:
    def __init__(self):
        # Initialize the random forest classifier with a fixed random state for reproducibility.
        self.clf = RandomForestClassifier(random_state=42, n_jobs=4, n_estimators=5)
        self.action_encoder = LabelEncoder()
        self.external_type_encoder = LabelEncoder()

    def _fit_preprocess(self, X):
        """Preprocess the input data."""
        X.drop(columns=["External", "AccountID"], inplace=True)
        X['Action'] = self.action_encoder.fit_transform(X['Action'])
        X['External_Type'] = self.external_type_encoder.fit_transform(X['External_Type'])
        return X
    
    def _predict_preprocess(self, X):
        """Preprocess the input data."""
        X.drop(columns=["External", "AccountID"], inplace=True)
        X['Action'] = self.action_encoder.transform(X['Action'])
        X['External_Type'] = self.external_type_encoder.transform(X['External_Type'])
        return X

    def fit(self, X, y):
        """Fit the random forest model to the training data."""
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