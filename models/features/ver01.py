import datetime
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

class Features:
    def __init__(self):
        self.action_encoder = LabelEncoder()
        self.external_type_encoder = LabelEncoder()
        self.action_types = ['CASH_IN', 'CASH_OUT', 'DEBIT', 'PAYMENT', 'TRANSFER']

    def _create_action_counts(self, X, group_cols):
        """Helper method to create action count features efficiently."""
        # First, create a dictionary to track all counts we need
        result_dict = {}
        
        # Total count of actions per group
        suffix = '' if len(group_cols) == 1 else 'Per' + group_cols[-1].title()
        
        # Add total count feature
        result_dict[f'NumActions{suffix}'] = X.groupby(group_cols)['Action'].transform('count')
        
        # Create counts for each action type in a more efficient way
        for action in self.action_types:
            # Create a masked dataframe with 1s where action matches
            mask = (X['Action'] == action).astype(int)
            # Use proper groupby with column names and cumsum
            result_dict[f'Num{action.title().replace("_", "")}{suffix}'] = mask.groupby([X[col] for col in group_cols]).cumsum().values
        
        return result_dict

    def extract(self, X):
        """Extract features from the input data."""
        # Create a copy to avoid modifying the original dataframe
        # X = X.copy()
        
        # Time features - vectorized operations
        X['ToD'] = X['Hour'] % 24
        X['Day'] = (X['Hour'] - X['ToD']) // 24
        X['DoW'] = X['Day'] % 7
        
        # Basic stats - vectorized operations
        # Avoid division by zero
        X['PercentageOfBalance'] = np.where(X['OldBalance'] > 0, 
                                           X['Amount'] / X['OldBalance'], 
                                           0)
        
        # Group-based statistics - done once
        X['MeanTransferAmount'] = X.groupby('AccountID')['Amount'].transform('mean')
        X['MeanTransferCumSum'] = X.groupby('AccountID')['Amount'].transform('cumsum')
        X['MeanTransferAmountDifference'] = X['Amount'] - X['MeanTransferAmount']
        
        # External account metrics
        X['NumExternalAccounts'] = X.groupby('AccountID')['External'].transform('nunique')
        X['NumExternalTypes'] = X.groupby('AccountID')['External_Type'].transform('nunique')
        
        # Create action counts efficiently using helper method
        # By AccountID
        counts_by_account = self._create_action_counts(X, ['AccountID'])
        for col, values in counts_by_account.items():
            X[col] = values
            
        # By AccountID and Day
        counts_by_day = self._create_action_counts(X, ['AccountID', 'Day'])
        for col, values in counts_by_day.items():
            X[col] = values
            
        # By AccountID and Hour
        counts_by_hour = self._create_action_counts(X, ['AccountID', 'Hour'])
        for col, values in counts_by_hour.items():
            X[col] = values
        
        return X

    def _fit_preprocess(self, X):
        """Preprocess the input data."""
        X = self.extract(X)

        # encoding - done in place
        X['Action'] = self.action_encoder.fit_transform(X['Action'])
        X['External_Type'] = self.external_type_encoder.fit_transform(X['External_Type'])

        # drop columns
        X.drop(columns=["External", "AccountID"], inplace=True)
        return X
    
    def _predict_preprocess(self, X):
        """Preprocess the input data."""
        X = self.extract(X)

        # encoding
        X['Action'] = self.action_encoder.transform(X['Action'])
        X['External_Type'] = self.external_type_encoder.transform(X['External_Type'])

        # drop columns
        X.drop(columns=["External", "AccountID"], inplace=True)
        return X

if __name__ == "__main__":
    import pandas as pd
    VAL_X_PATH   = "/Users/yale/Repositories/bbdc25/task/val_set/x_val.parquet"
    VAL_Y_PATH   = "/Users/yale/Repositories/bbdc25/task/val_set/y_val.parquet"

    x_val_df = pd.read_parquet(VAL_X_PATH)
    y_val_df = pd.read_parquet(VAL_Y_PATH)
    val = pd.merge(x_val_df, y_val_df, on="AccountID")[:100_000]
    # val = pd.merge(x_val_df, y_val_df, on="AccountID")
    X_val = val.drop(columns=["Fraudster"])
    y_val = val["Fraudster"]

    tnow = datetime.datetime.now()
    features = Features()
    X_val = features.extract(X_val)
    print(f"Preprocessing took: {datetime.datetime.now() - tnow}")
    # X_val[-100:].to_csv("featuresNew.csv", index=False)
    print(X_val.tail())