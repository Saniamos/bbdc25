import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from sklearn.calibration import LabelEncoder
from transformers import PreTrainedTokenizer
from sklearn.preprocessing import StandardScaler

na_cols = ['External', 'External_Type']

class AccountTransactionDataset(Dataset):
    """
    Dataset for loading all transactions for an account along with its fraud label.
    """
    
    def __init__(self, 
                 features_df: pd.DataFrame, 
                 labels_df: pd.DataFrame, 
                 feature_cols: Optional[List[str]] = None,
                 max_seq_len: Optional[int] = 2048,
                 normalize=True,
                 mask=True):
        """
        Initialize the dataset with feature and label dataframes.
        
        Args:
            features_df: DataFrame containing transaction features with AccountID column
            labels_df: DataFrame containing fraud labels with AccountID and Fraudster columns
            feature_cols: List of feature column names to use (if None, use all except AccountID)
            max_seq_len: Maximum sequence length for padding (if None, determined from data)
        """
        exclude_cols = ['AccountID', 'External']
        str_cols = features_df.select_dtypes(include=[object]).columns
        print('str cols:', str_cols)
        if normalize:
            self.normalizer = StandardScaler(copy=False)
            numeric_cols = features_df.select_dtypes(include=['number']).columns
            scaled_features = self.normalizer.fit_transform(features_df[numeric_cols].values).astype(np.float16)
            features_df[numeric_cols] = pd.DataFrame(scaled_features, index=features_df.index, columns=numeric_cols)            
            del scaled_features

        for col in str_cols:
            if col not in exclude_cols:
                features_df[col] = LabelEncoder().fit_transform(features_df[col])

        self.mask = mask
        # Ensure both dataframes have AccountID
        assert "AccountID" in features_df.columns, "Features DataFrame must have AccountID column"
        assert "AccountID" in labels_df.columns, "Labels DataFrame must have AccountID column"
        assert "Fraudster" in labels_df.columns, "Labels DataFrame must have Fraudster column"
        
        # Create account to label mapping
        self.fraud_bools = labels_df['Fraudster'].values 
        self.account_to_label = labels_df.set_index('AccountID')['Fraudster'].to_dict()
        
        # Determine feature columns
        if feature_cols is None:
            self.feature_cols = list(features_df.columns)
        else:
            self.feature_cols = feature_cols
        self.feature_cols = [col for col in features_df.columns if col not in exclude_cols]
        
        # Group transactions by account
        self.account_groups = features_df.groupby("AccountID")
        self.account_ids = list(self.account_groups.groups.keys())
        
        # Determine max sequence length if not provided
        if max_seq_len is None:
            self.max_seq_len = max(len(group) for _, group in self.account_groups)
            print(f"Determined max sequence length: {self.max_seq_len}")
        else:
            self.max_seq_len = max_seq_len
            
        # Store feature dimension
        self.feature_dim = len(self.feature_cols)
        
        # Print some statistics
        print(f"Loaded dataset with {len(self.account_ids)} accounts and {len(features_df)} transactions")
        print(f"Feature columns: {len(self.feature_cols)}")
        print(f"Fraud accounts: {labels_df['Fraudster'].sum()} ({(labels_df['Fraudster'].sum() / len(labels_df) * 100):.2f}%)")
        
        # assert no nan values are present
        # if features_df.drop(na_cols, axis=1).isnull().values.any():
        #     raise ValueError('nan values found in features_df')

        del features_df

    
    def get_shape(self):
        return self.max_seq_len, self.feature_dim
    
    def get_fraud_labels_idx(self):
        return self.fraud_bools

    def __len__(self) -> int:
        """Return the number of accounts."""
        return len(self.account_ids)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get all transactions for an account and its fraud label.
        
        Args:
            idx: Index of the account to retrieve
        
        Returns:
            Tuple containing:
                - features: Padded tensor of transaction features [max_seq_len, num_features]
                - label: Binary fraud label (0 or 1)
        """
        account_id = self.account_ids[idx]
        account_transactions = self.account_groups.get_group(account_id)
        
        # Extract features and convert to tensor
        features = account_transactions[self.feature_cols].values
        seq_len = features.shape[0]

        # Create pre-padded tensor
        padded_features = torch.zeros((self.max_seq_len, self.feature_dim), dtype=torch.float32)

        # Fill with actual values (no need to copy if already a torch tensor)
        padded_features[:seq_len] = torch.FloatTensor(features)

        masked_features, masked_pos = self.mask_features(padded_features, seq_len)
        
        # Get label for this account
        label = self.account_to_label[account_id]
        label = torch.FloatTensor([label])

        # check for nan values and raise an error if found
        if torch.isnan(padded_features).any():
            raise ValueError('nan values found in padded_features')
        
        return masked_features, masked_pos, padded_features, label
    
    # def simple_mask_features(self, features, seq_len):
    #     if not self.mask:
    #         return features, torch.zeros_like(features, dtype=torch.bool)
        
    #     # Quick masking: single pass, no loops
    #     masked_features = features.clone()
    #     masked_pos = torch.zeros_like(features, dtype=torch.bool)
        
    #     # Mask valid positions only (non-padding)
    #     valid_features = features[:seq_len]
        
    #     # Simple random masking (15% of features)
    #     mask = (torch.rand_like(valid_features) < 0.15)
    #     masked_pos[:seq_len] = mask
    #     masked_features[:seq_len][mask] = 0.0
        
    #     return masked_features, masked_pos

    def mask_features(self, features, seq_len):
        """
        Apply state-of-the-art masking for self-supervised learning with optimized performance.
        
        Args:
            features: Tensor of shape [max_seq_len, feature_dim]
            seq_len: Actual sequence length (non-padded transactions)
            
        Returns:
            masked_features: Features with masking applied
            masked_pos: Boolean tensor indicating masked positions
        """
        if not self.mask:
            return features, torch.zeros_like(features, dtype=torch.bool)
        
        # Create output tensors (avoid unnecessary copies)
        masked_features = features.clone()
        masked_pos = torch.zeros_like(features, dtype=torch.bool)
        
        # STRATEGY 1: Vectorized random feature masking
        # Create a mask for valid positions only (non-padding)
        valid_mask = torch.zeros((self.max_seq_len,), dtype=torch.bool)
        valid_mask[:seq_len] = True
        
        # Generate all random values at once (more efficient)
        # Shape: [seq_len, feature_dim]
        mask_probs = torch.rand((seq_len, self.feature_dim))
        feature_mask = mask_probs < 0.30  # Mask 30% of features
        
        # Count number of masked values to prepare for 80-10-10 split
        num_masked = feature_mask.sum().item()
        if num_masked > 0:
            # Generate all probabilities for 80-10-10 rule at once
            split_probs = torch.rand(num_masked)
            
            # Create indices for masked positions
            masked_indices = torch.nonzero(feature_mask, as_tuple=False)
            
            # Partition based on 80-10-10 split
            mask_80 = split_probs < 0.8
            mask_10a = (split_probs >= 0.8) & (split_probs < 0.9)
            mask_10b = split_probs >= 0.9
            
            # Apply the masks efficiently
            # 1. Mark all positions as masked in the position tensor
            for idx in range(masked_indices.size(0)):
                i, j = masked_indices[idx]
                masked_pos[i, j] = True
            
            # 2. Set 80% to zero (mask tokens)
            mask_80_indices = masked_indices[mask_80]
            for idx in range(mask_80_indices.size(0)):
                i, j = mask_80_indices[idx]
                masked_features[i, j] = 0.0
            
            # 3. Set 10% to random values
            mask_10a_indices = masked_indices[mask_10a]
            if mask_10a_indices.size(0) > 0:
                random_values = torch.randn(mask_10a_indices.size(0))
                for idx in range(mask_10a_indices.size(0)):
                    i, j = mask_10a_indices[idx]
                    masked_features[i, j] = random_values[idx]
            
            # 4. Keep original 10% (already done by clone)
        
        # STRATEGY 2: Temporal span masking (optimized)
        if seq_len > 5:
            span_start = torch.randint(0, max(1, seq_len - 3), (1,)).item()
            span_length = min(torch.randint(1, 4, (1,)).item(), seq_len - span_start)
            
            # Generate all span masks at once
            span_masks = torch.rand(span_length, self.feature_dim) < 0.4
            
            # Apply span masks in a single operation where possible
            for i in range(span_length):
                idx = span_start + i
                span_mask = span_masks[i]
                if span_mask.any():
                    span_mask_indices = span_mask.nonzero(as_tuple=True)[0]
                    masked_pos[idx, span_mask_indices] = True
                    masked_features[idx, span_mask_indices] = 0.0
        
        return masked_features, masked_pos

def load_datasets(X_list):
    y_res = []
    x_res = []
    for i, d in enumerate(X_list):
        source_suffix = f'yh{i}' # random id that should not occour in accountid or external
        x_df = pd.read_parquet(d)
        # assert no nan values
        # if x_df.drop(na_cols, axis=1).isnull().values.any():
        #     raise ValueError('nan values found in x_df')
        try:
            y_df = pd.read_parquet(d.replace("/x_", "/y_"))
        except FileNotFoundError:
            y_df = pd.DataFrame({"AccountID": x_df["AccountID"].unique(), "Fraudster": 0})
        
        # Add suffix to AccountID to ensure uniqueness
        x_df["AccountID"] = x_df["AccountID"].astype(str) + source_suffix
        non_empty_external = ~x_df["External"].isna()
        x_df.loc[non_empty_external, "External"] = x_df.loc[non_empty_external, "External"].astype(str) + source_suffix
        y_df["AccountID"] = y_df["AccountID"].astype(str) + source_suffix
        
        x_res.append(x_df)
        y_res.append(y_df)
    
    y_res = pd.concat(y_res)
    x_res = pd.concat(x_res)

    # # assert no nan values are present
    # if x_res.drop(na_cols, axis=1).isnull().values.any():
    #     raise ValueError('nan values found in x_res')
    
    return x_res, y_res

def load_train(ver):
    return load_datasets([f"~/Repositories/bbdc25/task/train_set/x_train.{ver}.parquet",])

def load_val(ver):
    return load_datasets([f"~/Repositories/bbdc25/task/val_set/x_val.{ver}.parquet",])

def load_test(ver):
    return load_datasets([f"~/Repositories/bbdc25/task/test_set/x_test.{ver}.parquet",])

def load_train_test(ver):
    return load_datasets([f"~/Repositories/bbdc25/task/train_set/x_train.{ver}.parquet",
                          f"~/Repositories/bbdc25/task/test_set/x_test.{ver}.parquet"])

def load_train_val(ver):
    return load_datasets([f"~/Repositories/bbdc25/task/train_set/x_train.{ver}.parquet",
                          f"~/Repositories/bbdc25/task/val_set/x_val.{ver}.parquet"])

def load_all(ver):
    return load_datasets([f"~/Repositories/bbdc25/task/train_set/x_train.{ver}.parquet",
                          f"~/Repositories/bbdc25/task/val_set/x_val.{ver}.parquet",
                          f"~/Repositories/bbdc25/task/test_set/x_test.{ver}.parquet"])

def prepare_dataset(ver, fn, **kwargs):
    return AccountTransactionDataset(*fn(ver), **kwargs)

def prepare_datasets(ver='ver01', fns=(load_train, load_val, load_test), **kwargs):
    return [prepare_dataset(ver, fn, **kwargs) for fn in fns]
    
if __name__ == "__main__":
    from monitor import Monitor 
    import time
    with Monitor():
        x_train_df, y_train_df = load_train('ver01')

        dataset = AccountTransactionDataset(x_train_df, y_train_df)
        print(dataset[0])
        print(dataset[0][0].shape)
        time.sleep(1)