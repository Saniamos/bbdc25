from sklearn.calibration import LabelEncoder

class Features:
    def __init__(self):
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
    

