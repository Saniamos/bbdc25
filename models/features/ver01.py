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
        X['ToD'] = X['Hour'] % 24
        X['DoW'] = X['Hour'] % (7 * 24)
        return X
    
    # def _predict_preprocess(self, X):
    #     """Preprocess the input data."""
    #     X.drop(columns=["External", "AccountID"], inplace=True)
    #     X['Action'] = self.action_encoder.transform(X['Action'])
    #     X['External_Type'] = self.external_type_encoder.transform(X['External_Type'])
    #     return X



if __name__ == "__main__":
    import pandas as pd
    VAL_X_PATH   = "/Users/yale/Repositories/bbdc25/task/val_set/x_val.csv"
    VAL_Y_PATH   = "/Users/yale/Repositories/bbdc25/task/val_set/y_val.csv"

    x_val_df = pd.read_csv(VAL_X_PATH, compression='gzip')
    y_val_df = pd.read_csv(VAL_Y_PATH, compression='infer')
    val = pd.merge(x_val_df, y_val_df, on="AccountID")[:10_000]
    X_val = val.drop(columns=["Fraudster"])
    y_val = val["Fraudster"]

    features = Features()
    X_val = features._fit_preprocess(X_val)
    print(X_val.head())