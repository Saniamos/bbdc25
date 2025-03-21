import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from sklearn.calibration import LabelEncoder
from sklearn.preprocessing import StandardScaler
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
import numpy as np

# Columns that can contain NA values
NA_COLS = ['External', 'External_Type']
EXCLUDE_COLS = ['AccountID', 'External']


class AccountTransactionDataset(Dataset):
    def __init__(self, 
                 features_df: pd.DataFrame, 
                 labels_df: pd.DataFrame, 
                 feature_cols: Optional[List[str]] = None,
                 max_seq_len: Optional[int] = 2048,
                 normalize=True,
                 mask=True,
                 log_fn=print,
                 precompute=True,
                 gpu_cache=False):
        """
        Initialize the dataset with feature and label dataframes.
        
        Args:
            features_df: DataFrame containing transaction features with AccountID column
            labels_df: DataFrame containing fraud labels with AccountID and Fraudster columns
            feature_cols: List of feature column names to use (if None, use all except AccountID)
            max_seq_len: Maximum sequence length for padding (if None, determined from data)
            normalize: Whether to normalize numeric features
            mask: Whether to apply masking
            log_fn: Logging function
            precompute: Whether to precompute and cache all tensors
            gpu_cache: Whether to store precomputed tensors directly on GPU
        """
        # Identify string columns for encoding
        str_cols = features_df.select_dtypes(include=[object]).columns
        log_fn(f'String columns: {list(str_cols)}')
        
        # Normalize numeric features if requested
        if normalize:
            numeric_cols = features_df.select_dtypes(include=['number']).columns
            self._normalize_features(features_df, numeric_cols, log_fn)
            
        # Encode string columns
        for col in str_cols:
            if col not in EXCLUDE_COLS:
                features_df[col] = LabelEncoder().fit_transform(features_df[col])

        # Set up masking function
        self.mask = mask
        if not mask:
            self.mask_features = lambda x, y: (torch.empty(0), torch.empty(0))
        
        # Validate that both DataFrames have required columns
        self._validate_dataframes(features_df, labels_df)
        
        # Store fraud labels and account IDs
        self.fraud_bools = labels_df['Fraudster'].values 
        self.account_ids = labels_df['AccountID'].values
        
        # Determine feature columns
        self.feature_cols = self._get_feature_columns(features_df, feature_cols)
        
        # Group transactions by account
        self.account_groups = features_df.groupby("AccountID")
        
        # Determine max sequence length if not provided
        self.max_seq_len = self._get_max_seq_len(max_seq_len, log_fn)
            
        # Store feature dimension
        self.feature_dim = len(self.feature_cols)
        
        # Log dataset statistics
        self._log_dataset_stats(features_df, labels_df, log_fn)
        
        # Check for NaN values (excluding allowed columns)
        if features_df.drop(NA_COLS, axis=1).isnull().values.any():
            raise ValueError('NaN values found in features_df')

        # Free memory
        del features_df

        # Set up for precomputation
        self.precompute = precompute
        self.gpu_cache = gpu_cache
        self.device = torch.device('cuda' if gpu_cache and torch.cuda.is_available() else 'cpu')
        
        # Precompute tensors if enabled
        if self.precompute:
            self._precompute_tensors(log_fn)

    def _normalize_features(self, features_df, numeric_cols, log_fn):
        """Normalize numeric features using StandardScaler."""
        log_fn(f"Normalizing {len(numeric_cols)} numeric columns")
        self.normalizer = StandardScaler(copy=False)
        scaled_features = self.normalizer.fit_transform(features_df[numeric_cols].values).astype(np.float16)
        features_df[numeric_cols] = pd.DataFrame(scaled_features, index=features_df.index, columns=numeric_cols)
    
    def _validate_dataframes(self, features_df, labels_df):
        """Validate that DataFrames have required columns and matching AccountIDs."""
        assert "AccountID" in features_df.columns, "Features DataFrame must have AccountID column"
        assert "AccountID" in labels_df.columns, "Labels DataFrame must have AccountID column"
        assert "Fraudster" in labels_df.columns, "Labels DataFrame must have Fraudster column"
        assert set(features_df['AccountID']) == set(labels_df['AccountID']), "AccountID mismatch between DataFrames"
    
    def _get_feature_columns(self, features_df, feature_cols):
        """Determine which columns to use as features."""
        if feature_cols is None:
            return [col for col in features_df.columns if col not in EXCLUDE_COLS]
        return feature_cols
    
    def _get_max_seq_len(self, max_seq_len, log_fn):
        """Determine maximum sequence length."""
        if max_seq_len is None:
            max_len = max(len(group) for _, group in self.account_groups)
            log_fn(f"Determined max sequence length: {max_len}")
            return max_len
        return max_seq_len
    
    def _log_dataset_stats(self, features_df, labels_df, log_fn):
        """Log dataset statistics."""
        log_fn(f"Loaded dataset with {len(self.account_ids)} accounts and {len(features_df)} transactions")
        log_fn(f"Feature columns: {len(self.feature_cols)} -- {list(self.feature_cols)}")
        log_fn(f"Fraud accounts: {labels_df['Fraudster'].sum()} ({(labels_df['Fraudster'].sum() / len(labels_df) * 100):.2f}%)")
    
    def _precompute_tensors(self, log_fn):
        """Precompute all tensors for faster data loading."""
        log_fn("Precomputing tensors for faster data loading...")
        self.precomputed_data = []
        
        # Loop through all accounts and precompute tensors
        for idx in range(len(self.account_ids)):
            # Process account data
            masked_features, masked_pos, padded_features, label = self._process_account_data(idx=idx)
            
            # Store on GPU if requested
            if self.gpu_cache:
                padded_features = padded_features.to(self.device)
                masked_features = masked_features.to(self.device)
                masked_pos = masked_pos.to(self.device)
                label = label.to(self.device)
            
            # Store precomputed data
            self.precomputed_data.append((masked_features, masked_pos, padded_features, label))
        
        log_fn(f"Precomputed {len(self.precomputed_data)} tensors (stored on {'GPU' if self.gpu_cache else 'CPU'})")
        
        # Free up memory
        if self.precompute:
            del self.account_groups
            self.account_groups = None
    
    def get_shape(self):
        """Return (max_seq_len, feature_dim) tuple describing the shape of features."""
        return self.max_seq_len, self.feature_dim
    
    def get_fraud_labels(self):
        """Return array of fraud labels."""
        return self.fraud_bools
    
    def get_account_ids(self):
        """Return array of account IDs."""
        return self.account_ids
    
    def __len__(self) -> int:
        """Return the number of accounts."""
        return len(self.account_ids)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get all transactions for an account and its fraud label.
        
        Args:
            idx: Index of the account to retrieve
        
        Returns:
            Tuple containing:
                - masked_features: Masked tensor of transaction features
                - masked_pos: Boolean tensor indicating masked positions
                - features: Original padded tensor of transaction features
                - label: Binary fraud label (0 or 1)
        """
        if self.precompute:
            # Simply return precomputed tensors
            return self.precomputed_data[idx]
        
        # Process account data on-the-fly
        return self._process_account_data(idx=idx)
    
    def _process_account_data(self, account_id=None, idx=None, account_transactions=None):
        """
        Process a single account's transactions into tensor format.
        
        Args:
            account_id: The account ID to process (used if account_transactions not provided)
            idx: Index of the account in self.account_ids (used for getting label)
            account_transactions: Pre-fetched transactions DataFrame (optional)
        
        Returns:
            Tuple containing:
                - masked_features: Masked tensor of transaction features
                - masked_pos: Boolean tensor indicating masked positions
                - padded_features: Original padded tensor of transaction features
                - label: Binary fraud label (0 or 1)
        """
        # Get account transactions if not provided
        if account_transactions is None:
            if account_id is None:
                account_id = self.account_ids[idx]
            account_transactions = self.account_groups.get_group(account_id)
        
        # Get index from account_id if not provided
        if idx is None:
            idx = np.where(self.account_ids == account_id)[0][0]
        
        # Extract features and convert to tensor
        features = account_transactions[self.feature_cols].values
        seq_len = features.shape[0]
        
        # Create padded tensor
        padded_features = torch.zeros((self.max_seq_len, self.feature_dim), dtype=torch.float32)
        padded_features[:seq_len] = torch.FloatTensor(features)[:self.max_seq_len]
        
        # Apply masking if needed
        if self.mask:
            masked_features, masked_pos = self.mask_features(padded_features, seq_len)
        else:
            masked_features = torch.empty(0)
            masked_pos = torch.empty(0)
        
        # Get label for this account
        label = torch.FloatTensor([self.fraud_bools[idx]])
        
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
    


def prep_hpsearch_dataloaders(data_version, seed, batch_size, num_workers, load_fn=load_all, 
                              persistent_workers=True, pin_memory=True, prefetch_factor=2, log_fn=print, **kwargs):
    """
    Prepare optimized dataloaders for hyperparameter search
    """
    pl.seed_everything(seed)
    
    # Prepare dataset
    dataset = prepare_dataset(data_version, load_fn, **kwargs)
    
    # Get all labels to create a stratified split
    all_labels = dataset.get_fraud_labels()
    
    # Create stratified train/validation indices
    indices = np.arange(len(dataset))
    train_indices, val_indices = train_test_split(
        indices, 
        test_size=0.2,  # 20% for validation
        random_state=seed,
        stratify=all_labels  # This ensures the split preserves the class distribution
    )
    
    # Create Subset objects
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    
    log_fn(f"Dataset split: {len(train_dataset)} training samples, {len(val_dataset)} validation samples")
    log_fn(f"Stratified by fraudster class to maintain class distribution in both sets")
    
    # Get feature dimension from dataset
    feature_dim = dataset.feature_dim
    log_fn(f"Feature dimension: {feature_dim}")
    
    common_args = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers if num_workers > 0 else False,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
    )

    # Create optimized DataLoaders
    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        drop_last=True,  # Improves performance by avoiding small batches
        **common_args
    )
    
    val_loader = DataLoader(
        val_dataset,
        shuffle=False,
        drop_last=True,  # Improves performance by avoiding small batches
        **common_args
    )
    
    return train_loader, val_loader, feature_dim


if __name__ == "__main__":
    num_workers = 4
    val_loader = DataLoader(
        prepare_dataset('ver05', mask=False, log_fn=print, fn=load_val),
        shuffle=False,
        drop_last=False,
        batch_size=128,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=2 if num_workers > 0 else None,
    )  

    account_ids = []
    fraud_bools = []
    for i, batch in enumerate(val_loader):
        _, _, _, fraud, account_id = batch
        account_ids.extend(account_id)
        fraud_bools.extend(fraud.flatten())

    account_ids = np.array(account_ids)
    fraud_bools = np.array(fraud_bools)

    d = val_loader.dataset

    account_ids_accoring_to_prop = np.array(d.account_ids)
    fraud_bools_according_to_prop = np.array(d.fraud_bools)

    print(account_ids != account_ids_accoring_to_prop)
    print((account_ids != account_ids_accoring_to_prop).sum())
    print(fraud_bools != fraud_bools_according_to_prop)
    print((fraud_bools != fraud_bools_according_to_prop).sum())
    print('Done')

    # from monitor import Monitor 
    # import time
    # with Monitor():
        # x_train_df, y_train_df = load_train('ver01')

        # dataset = AccountTransactionDataset(x_train_df, y_train_df)
        # print(dataset[0])
        # print(dataset[0][0].shape)
        # time.sleep(1)