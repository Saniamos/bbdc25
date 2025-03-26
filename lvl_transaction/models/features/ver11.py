import datetime
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from snapml import GraphFeaturePreprocessor

# basically ver05 with rev transactions

class Features:
    def __init__(self):
        self.action_encoder = LabelEncoder()
        self.external_type_encoder = LabelEncoder()
        self.action_types = ['CASH_IN', 'CASH_OUT', 'DEBIT', 'PAYMENT', 'TRANSFER']
        self.graph_feature_preprocessor = GraphFeaturePreprocessor()

    def extract(self, X):
        """Extract features from the input data."""
        # Create a single result DataFrame to avoid multiple concatenations
        result_df = pd.DataFrame(index=X.index)
        
        # Time features - vectorized operations
        result_df['ToD'] = X['Hour'] % 24
        
        # Add Day to both result_df and X since we need to group by it
        day_values = (X['Hour'] // 24).astype('int8')
        result_df['Day'] = day_values
        X = X.copy()  # Create a copy to avoid modifying the original
        X['Day'] = day_values
        
        result_df['DoW'] = result_df['Day'] % 7

         # Avoid division by zero
        result_df['PercentageOfBalance'] = np.divide(
            X['Amount'], X['OldBalance'], 
            out=np.zeros(len(X), dtype='float32'), 
            where=X['OldBalance'] > 0
        )

        ext_types = X['External_Type'].unique()
        for external_type in sorted(map(str, ext_types)):
            mask = (X['External_Type'] == external_type).astype('int8')
            ext_grouped = mask.groupby(X['AccountID'].values)  # Use values for efficiency
            result_df[f'NumExternalAccounts{external_type}'] = ext_grouped.transform('sum').values
        
                
        # Copy needed columns from X to result_df
        for col in X.columns:
            if col not in result_df:
                result_df[col] = X[col]
        
        # fit graph feature preprocessor
        idx = ~X['External'].isna() # only fit where money changed accounts
        gfp_X = np.array([X[idx].index.values,
                 LabelEncoder().fit_transform(X[idx]['AccountID']),
                 LabelEncoder().fit_transform(X[idx]['External']),
                 X[idx]['Hour']]).T
        enriched = self.graph_feature_preprocessor.fit_transform(gfp_X)[:, 4:]
        res = np.zeros((X.shape[0], enriched.shape[1]), dtype=np.float32)
        res[idx] = enriched
        enriched_col_names = [f"GFP_{i}" for i in range(enriched.shape[1])]
        enriched_df = pd.DataFrame(res, columns=enriched_col_names, index=X.index)
        result_df = pd.concat([result_df, enriched_df], axis=1)

        # add reverse transactions
        result_df['AmountNonAbs'] = X['NewBalance'] - X['OldBalance']
        idx = result_df['External'].notna() & result_df['External'].isin(result_df['AccountID'].unique())
        rev = result_df[idx].copy().rename(columns={'AccountID': 'External', 'External': 'AccountID'})
        rev['External_Type'] = rev['External_Type'].apply(lambda x: f"REV_{x}")
        rev[rev.columns[~np.isin(list(rev.columns), ['Hour', 'ToD', 'DoW', 'Day', 'AccountID', 'Action', 'Amount', 'External', 'External_Type', 'AmountNonAbs'])]] = 0
        # rev['NewBalance'] = np.nan
        result_df = pd.concat([result_df, rev], ignore_index=True).reset_index().sort_values(['Hour', 'index']).reset_index(drop=True).drop(['index'], axis=1)

        # def fill_missing_transactions(group):
        #     group = group.reset_index().sort_values(['Hour', 'index']).drop('index', axis=1)  # ensure correct order within the account
        #     for pos, idx in enumerate(group.index):
        #         if pd.isna(group.at[idx, 'OldBalance']):
        #             last_new_balance = group.iloc[pos-1]['NewBalance']
        #             group.at[idx, 'OldBalance'] = last_new_balance
        #             group.at[idx, 'NewBalance'] = last_new_balance + group.at[idx, 'AmountNonAbs']
        #     return group

        # result_df = result_df.groupby('AccountID', group_keys=False).apply(fill_missing_transactions)
        # result_df[['OldBalance', 'NewBalance']] = result_df[['OldBalance', 'NewBalance']].fillna(0) # in rare cases, e.g. first transaction in account may not be calculatable

        assert result_df.drop(['External', 'External_Type'], axis=1).isna().sum().sum() == 0, f"Missing values in result_df: {result_df.isna().sum()}"
        
        return result_df

    def _fit_preprocess(self, X):
        """Preprocess the input data."""
        X = self.extract(X.copy())

        # encoding - done in place
        X['Action'] = self.action_encoder.fit_transform(X['Action'])
        X['External_Type'] = self.external_type_encoder.fit_transform(X['External_Type'])

        # drop columns
        X = X.drop(columns=["External", "AccountID"])
        return X
    
    def _predict_preprocess(self, X):
        """Preprocess the input data."""
        X = self.extract(X.copy())

        # encoding
        X['Action'] = self.action_encoder.transform(X['Action'])
        X['External_Type'] = self.external_type_encoder.transform(X['External_Type'])

        # drop columns
        X = X.drop(columns=["External", "AccountID"])
        return X

if __name__ == "__main__":
    import pandas as pd
    VAL_X_PATH   = "~/Repositories/bbdc25/task/val_set/x_val.ver00.parquet"
    VAL_Y_PATH   = "~/Repositories/bbdc25/task/val_set/y_val.ver00.parquet"

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
    # print lentgh of each group by AccountID
    val_counts = X_val.groupby('AccountID').size().value_counts()
    print(val_counts)
    print('Longest Sequence:', val_counts.index.max())
    X_val[-50:].to_csv("test2.csv", index=False)
    print(X_val.tail())