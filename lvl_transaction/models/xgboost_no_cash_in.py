from models.features.encode import Features
import xgboost as xgb
import numpy as np

# Core idea is simple: no cash_in transactions are considered in the model.
# This is because technically the cash_ins should are necessary for money laundering, but they are likely not the actual transcations we want to flag.
# no clue if this is a good or bad idea, but will check the results and see if it makes sense. 

class Model:
    def __init__(self):
        # Initialize the random forest classifier with a fixed random state for reproducibility.
        self.clf = xgb.XGBClassifier(random_state=42)
        self.fts = Features()

    def fit(self, X, y):
        """Fit the balanced random forest model to the training data."""
        idx = X['Action'] != 'CASH_IN'
        X_prd = self.fts._fit_preprocess(X[idx])
        self.clf.fit(X_prd, y[idx])
    
    def predict(self, X):
        """Predict labels for the given data."""
        idx = X['Action'] != 'CASH_IN'
        prd = np.zeros(X.shape[0])
        X_prd = self.fts._predict_preprocess(X[idx])
        prd[idx] = self.clf.predict(X_prd)
        return prd
    
    def predict_proba(self, X):
        """Predict probabilities for the given data."""
        idx = X['Action'] != 'CASH_IN'
        prd = np.zeros(X.shape[0])
        X_prd = self.fts._predict_preprocess(X[idx])
        prd[idx] = self.clf.predict_proba(X_prd)[:, 1]
        return prd
    
    def refit(self, X, y):
        """Refit the model to the given data."""
        self.fit(X, y)
    
    def log_params(self):
        """Log the hyperparameters of the model."""
        return self.clf.get_params()
    
    def save(self, path):
        """Save the model to the given path."""
        self.clf.save_model(path.replace('.pkl', '.json'))