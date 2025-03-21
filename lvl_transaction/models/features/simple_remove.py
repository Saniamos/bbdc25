from sklearn.calibration import LabelEncoder

class Features:
    def __init__(self):
        self.action_encoder = LabelEncoder()
        self.external_type_encoder = LabelEncoder()

    def _fit_preprocess(self, X):
        """Preprocess the input data."""
        X = X.copy().drop(columns=["External", "AccountID", "Action", "External_Type"])
        return X
    
    def _predict_preprocess(self, X):
        """Preprocess the input data."""
        X = X.copy().drop(columns=["External", "AccountID", "Action", "External_Type"])
        return X

