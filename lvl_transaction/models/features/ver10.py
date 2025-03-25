import datetime
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from snapml import GraphFeaturePreprocessor

# tw = 50 # slightly over 2 days, takes ~ 2min 
tw = 24*7 # takes ~ 2min 
params = {
    "num_threads": 9,             # number of software threads to be used (important for performance)
    "time_window": tw,            # time window used if no pattern was specified
    
    "vertex_stats": True,         # produce vertex statistics
    "vertex_stats_cols": [3],     # produce vertex statistics using the selected input columns
    
    # features: 0:fan,1:deg,2:ratio,3:avg,4:sum,5:min,6:max,7:median,8:var,9:skew,10:kurtosis
    # "vertex_stats_feats": [0, 1, 2, 3, 4, 8, 9, 10],  # fan,deg,ratio,avg,sum,var,skew,kurtosis
    "vertex_stats_feats": list(range(11)),  # fan,deg,ratio,avg,sum,var,skew,kurtosis
    
    # fan in/out parameters
    "fan": True,
    "fan_tw": tw,
    "fan_bins": list(range(1, 60)),
    
    # in/out degree parameters
    "degree": True,
    "degree_tw": tw,
    "degree_bins": list(range(1, 60)),
    
    # scatter gather parameters
    "scatter-gather": True,
    "scatter-gather_tw": tw,
    "scatter-gather_bins": list(range(1, 30)),
    # "scatter-gather_bins": [1],
    
    # temporal cycle parameters
    "temp-cycle": True,
    "temp-cycle_tw": 50,
    "temp-cycle_bins": list(range(1, 9)),
    # "temp-cycle_bins": [1],
    
    # length-constrained simple cycle parameters
    "lc-cycle": True,
    "lc-cycle_tw": 50,
    "lc-cycle_len": 10,
    "lc-cycle_bins": list(range(1, 11)),
}

class Features:
    def __init__(self):
        self.action_encoder = LabelEncoder()
        self.external_type_encoder = LabelEncoder()
        self.action_types = ['CASH_IN', 'CASH_OUT', 'DEBIT', 'PAYMENT', 'TRANSFER']
        self.graph_feature_preprocessor = GraphFeaturePreprocessor()
        self.graph_feature_preprocessor.set_params(params)

    def _create_action_counts(self, X, result_df, group_cols):
        """Helper method to create action count features efficiently."""
        # Total count of actions per group
        suffix = '' if len(group_cols) == 1 else 'Per' + group_cols[-1].title()
        
        # Pre-calculate groupby objects to avoid redundant work
        grouped = X.groupby(group_cols)
        
        # Add total count feature - using more efficient operations
        result_df[f'NumActions{suffix}'] = grouped['Action'].transform('count')
        result_df[f'Amount{suffix}Cumsum'] = grouped['Amount'].transform('cumsum')
        result_df[f'Amount{suffix}Std'] = grouped['Amount'].transform('std').fillna(0)
        result_df[f'OldBalance{suffix}Std'] = grouped['OldBalance'].transform('std').fillna(0)
        result_df[f'NewBalance{suffix}Std'] = grouped['NewBalance'].transform('std').fillna(0)
        
        # Process all action types efficiently
        for action in self.action_types:
            # Create mask as int8 to reduce memory usage
            mask = (X['Action'] == action).astype('int8')
            
            # Cache X columns for group columns to reduce attribute access overhead
            group_cols_data = [X[col].values for col in group_cols] if group_cols else []
            
            # Use proper groupby with optimized calculations
            result_df[f'Num{action.title().replace("_", "")}{suffix}'] = mask.groupby(group_cols_data).cumsum().values
            result_df[f'Num{action.title().replace("_", "")}{suffix}Sum'] = mask.groupby(group_cols_data).transform('sum').values
            
            # Calculate amount features in one step
            masked_amount = mask * X['Amount']
            result_df[f'{action.title().replace("_", "")}{suffix}AmountSum'] = masked_amount.groupby(group_cols_data).transform('sum').values
            result_df[f'{action.title().replace("_", "")}{suffix}AmountStd'] = np.nan_to_num(masked_amount.groupby(group_cols_data).transform('std').values)
        
        return result_df

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

        # Calculate missing transactions (current old balance - previous new balance)
        # For each account, this reveals potential missing transactions or data gaps
        prev_new_balance = X.groupby('AccountID')['NewBalance'].shift(1)
        result_df['MissingTransaction'] = np.where(
            prev_new_balance.notna(),  # Only where previous transactions exist
            X['OldBalance'] - prev_new_balance,
            0  # For first transaction in each account, set to 0
        )

        # Add missing transaction amount relative to account balance
        result_df['MissingTransactionPct'] = np.divide(
            np.abs(result_df['MissingTransaction']), 
            X['OldBalance'], 
            out=np.zeros(len(X), dtype='float32'), 
            where=X['OldBalance'] > 0
        )

        # Flag for any missing transaction (binary feature)
        result_df['HasMissingTransaction'] = (np.abs(result_df['MissingTransaction']) > 0.01).astype('int8')

        ext_types = X['External_Type'].unique()
        for external_type in sorted(map(str, ext_types)):
            mask = (X['External_Type'] == external_type).astype('int8')
            ext_grouped = mask.groupby(X['AccountID'].values)  # Use values for efficiency
            result_df[f'NumExternalAccounts{external_type}'] = ext_grouped.transform('sum').values
        
        self._create_action_counts(X, result_df, ['AccountID'])
        self._create_action_counts(X, result_df, ['AccountID', 'Hour'])
        self._create_action_counts(X, result_df, ['AccountID', 'Day'])
                
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
        
        return result_df
        # return result_df.sort_values(["AccountID", "External_Type", "Hour"]).copy()

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
    VAL_X_PATH   = "~/Repositories/bbdc25/task/val_set/x_val.parquet"
    VAL_Y_PATH   = "~/Repositories/bbdc25/task/val_set/y_val.parquet"

    x_val_df = pd.read_parquet(VAL_X_PATH)
    y_val_df = pd.read_parquet(VAL_Y_PATH)
    val = pd.merge(x_val_df, y_val_df, on="AccountID")[:100_000]
    X_val = val.drop(columns=["Fraudster"])
    y_val = val["Fraudster"]

    tnow = datetime.datetime.now()
    features = Features()
    X_val = features.extract(X_val)
    print(f"Preprocessing took: {datetime.datetime.now() - tnow}")
    X_val[-50:].to_csv("test2.csv", index=False)
    print(X_val.tail())