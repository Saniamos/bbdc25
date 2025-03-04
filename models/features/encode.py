from sklearn.calibration import LabelEncoder

class Features:
    def __init__(self):
        self.action_encoder = LabelEncoder()
        self.external_type_encoder = LabelEncoder()
        self.account_encoder_fit = LabelEncoder()
        self.external_encoder_fit = LabelEncoder()
        self.account_encoder_pred = None
        self.external_encoder_pred = None

    def _fit_preprocess(self, X):
        """Preprocess the input data."""
        X = X.copy()
        X['Action'] = self.action_encoder.fit_transform(X['Action'])
        X['External_Type'] = self.external_type_encoder.fit_transform(X['External_Type'])
        X['AccountID'] = self.account_encoder_fit.fit_transform(X['AccountID'])
        X['External'] = self.external_encoder_fit.fit_transform(X['External'])
        return X
    
    def _predict_preprocess(self, X):
        """Preprocess the input data."""
        X = X.copy()
        X['Action'] = self.action_encoder.transform(X['Action'])
        X['External_Type'] = self.external_type_encoder.transform(X['External_Type'])

        if self.account_encoder_pred is None:
            # we specifically don't want to re-use the fit encoder, as the account ids should be different
            self.account_encoder_pred = LabelEncoder()
            self.external_encoder_pred = LabelEncoder()
            self.account_encoder_pred.fit(X['AccountID'])
            self.external_encoder_pred.fit(X['External'])
        
        X['AccountID'] = self.account_encoder_pred.fit_transform(X['AccountID'])
        X['External'] = self.external_encoder_pred.fit_transform(X['External'])
        return X
    

