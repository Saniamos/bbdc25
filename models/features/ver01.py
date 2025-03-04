import datetime
from sklearn.calibration import LabelEncoder

class Features:
    def __init__(self):
        self.action_encoder = LabelEncoder()
        self.external_type_encoder = LabelEncoder()

    def extract(self, X):
        """Extract features from the input data."""
        
        # some more specific dates
        X['ToD'] = X['Hour'] % 24
        X['Day'] = ((X['Hour'] - X['ToD']) // 24)
        X['DoW'] = X['Day'] % 7

        # some stats
        X['PercentageOfBalance'] = X['Amount'] / X['OldBalance']
        X['MeanTransferAmount'] = X.groupby('AccountID')['Amount'].transform('mean')
        X['MeanTransferCumSum'] = X.groupby('AccountID')['Amount'].transform('cumsum')

        # difference to mean transfer amount for account per transaction
        X['MeanTransferAmountDifference'] = X['Amount'] - X['MeanTransferAmount']
        # number of different external accounts for account per transaction
        X['NumExternalAccounts'] = X.groupby('AccountID')['External'].transform('nunique')
        # number of different external types for account per transaction
        X['NumExternalTypes'] = X.groupby('AccountID')['External_Type'].transform('nunique')


        # number of actions for account per transaction
        X['NumActions'] = X.groupby('AccountID')['Action'].transform('count')
        # number of cash ins for account per transaction
        X['NumCashIns'] = X.groupby('AccountID')['Action'].transform(lambda x: (x == 'CASH_IN').cumsum())
        # number of cash outs for account per transaction
        X['NumCashOuts'] = X.groupby('AccountID')['Action'].transform(lambda x: (x == 'CASH_OUT').cumsum())
        # number of DEBIT for account per transaction
        X['NumDebits'] = X.groupby('AccountID')['Action'].transform(lambda x: (x == 'DEBIT').cumsum())
        # number of PAYMENT for account per transaction
        X['NumPayments'] = X.groupby('AccountID')['Action'].transform(lambda x: (x == 'PAYMENT').cumsum())
        # number of TRANSFER for account per transaction
        X['NumTransfers'] = X.groupby('AccountID')['Action'].transform(lambda x: (x == 'TRANSFER').cumsum())


        # number of actions for account per transaction per day
        X['NumActionsPerDay'] = X.groupby(['AccountID', 'Day'])['Action'].transform('count')
        # number of cash ins for account per transaction per day
        X['NumCashInsPerDay'] = X.groupby(['AccountID', 'Day'])['Action'].transform(lambda x: (x == 'CASH_IN').cumsum())
        # number of cash outs for account per transaction per day
        X['NumCashOutsPerDay'] = X.groupby(['AccountID', 'Day'])['Action'].transform(lambda x: (x == 'CASH_OUT').cumsum())
        # number of DEBIT for account per transaction per day
        X['NumDebitsPerDay'] = X.groupby(['AccountID', 'Day'])['Action'].transform(lambda x: (x == 'DEBIT').cumsum())
        # number of PAYMENT for account per transaction per day
        X['NumPaymentsPerDay'] = X.groupby(['AccountID', 'Day'])['Action'].transform(lambda x: (x == 'PAYMENT').cumsum())
        # number of TRANSFER for account per transaction per day
        X['NumTransfersPerDay'] = X.groupby(['AccountID', 'Day'])['Action'].transform(lambda x: (x == 'TRANSFER').cumsum())
        

        # number of actions for account per transaction per hour
        X['NumActionsPerHour'] = X.groupby(['AccountID', 'Hour'])['Action'].transform('count')
        # number of cash ins for account per transaction per hour
        X['NumCashInsPerHour'] = X.groupby(['AccountID', 'Hour'])['Action'].transform(lambda x: (x == 'CASH_IN').cumsum())
        # number of cash outs for account per transaction per hour
        X['NumCashOutsPerHour'] = X.groupby(['AccountID', 'Hour'])['Action'].transform(lambda x: (x == 'CASH_OUT').cumsum())
        # number of DEBIT for account per transaction per hour
        X['NumDebitsPerHour'] = X.groupby(['AccountID', 'Hour'])['Action'].transform(lambda x: (x == 'DEBIT').cumsum())
        # number of PAYMENT for account per transaction per hour
        X['NumPaymentsPerHour'] = X.groupby(['AccountID', 'Hour'])['Action'].transform(lambda x: (x == 'PAYMENT').cumsum())
        # number of TRANSFER for account per transaction per hour
        X['NumTransfersPerHour'] = X.groupby(['AccountID', 'Hour'])['Action'].transform(lambda x: (x == 'TRANSFER').cumsum())
        
        return X

    def _fit_preprocess(self, X):
        """Preprocess the input data."""

        X = self.extract(X)

        # encoding
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
    VAL_X_PATH   = "/Users/yale/Repositories/bbdc25/task/val_set/x_val.csv"
    VAL_Y_PATH   = "/Users/yale/Repositories/bbdc25/task/val_set/y_val.csv"

    x_val_df = pd.read_csv(VAL_X_PATH, compression='gzip')
    y_val_df = pd.read_csv(VAL_Y_PATH, compression='infer')
    val = pd.merge(x_val_df, y_val_df, on="AccountID")[:100_000]
    # val = pd.merge(x_val_df, y_val_df, on="AccountID")
    X_val = val.drop(columns=["Fraudster"])
    y_val = val["Fraudster"]

    tnow = datetime.datetime.now()
    features = Features()
    X_val = features.extract(X_val)
    print(f"Preprocessing took: {datetime.datetime.now() - tnow}")
    print(X_val.tail())