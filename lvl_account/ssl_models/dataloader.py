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
import gc

# Columns that can contain NA values
NA_COLS = ['External', 'External_Type']#
# Columns to exclude from feature columns
EXCLUDE_COLS = ['AccountID', 'External']

# as determined by uploads, see readme
N_FRAUDSTERS = {
    'train': 1411,
    'val': 1472,
    'test': 1267
}

class AccountTransactionDataset(Dataset):
    # @profile
    def __init__(self, 
                 features_df: pd.DataFrame, 
                 labels_df: pd.DataFrame, 
                 feature_cols: Optional[List[str]] = None,
                 max_seq_len: Optional[int] = 2048,
                 normalize=True,
                 mask=True,
                 log_fn=print,
                 precompute=False,
                 highest_input=False):
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
        """
        self.highest_input = highest_input
        if self.highest_input:
            self.account_ids_highest_map = features_df[features_df['External_Type'] == 'customer'].groupby('External').apply(lambda g: g['AccountID'].value_counts().idxmin()).to_dict()
            # max_seq_len = max_seq_len * 2

        # Identify string columns for encoding
        str_cols = features_df.select_dtypes(include=[object]).columns
        log_fn(f'String columns: {list(str_cols)}')
        
        # Normalize numeric features if requested
        if normalize:
            numeric_cols = features_df.select_dtypes(include=['number']).columns
            self._normalize_features(features_df, numeric_cols, log_fn)
            
        # Encode string columns
        self.encoders = {}
        for col in str_cols:
            if col not in EXCLUDE_COLS:
                self.encoders[col] = LabelEncoder().fit(features_df[col])
                features_df[col] = self.encoders[col].transform(features_df[col])

        # Set up masking function
        self.mask = mask
        if not mask:
            self.mask_features = lambda x, y: (torch.empty(0), torch.empty(0))
        
        # Validate that both DataFrames have required columns
        self._validate_dataframes(features_df, labels_df)

        # Store fraud labels and account IDs
        self.fraud_bools = labels_df['Fraudster'].values
        self.account_ids = labels_df['AccountID'].values
        self.account_map = labels_df['AccountID'].reset_index().set_index('AccountID')['index'].to_dict()
        _tmp_enc = LabelEncoder().fit(list(features_df['AccountID'].values) + list(features_df['External'].unique()))
        log_fn(f'Total of {_tmp_enc.classes_.shape[0]} unique accounts')
        self.account_ids_enc = _tmp_enc.transform(labels_df['AccountID'])
        features_df['AccountID'] = _tmp_enc.transform(features_df['AccountID'])
        features_df['External'] = _tmp_enc.transform(features_df['External'])

        
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
        if features_df.drop(NA_COLS, axis=1).isnull().any().any():
            raise ValueError('NaN values found in features_df')

        # Free memory
        del features_df

        # Set up for precomputation
        self.precompute = precompute
        
        # Precompute tensors if enabled
        if self.precompute:
            self._precompute_tensors(log_fn)

    def _normalize_features(self, features_df, numeric_cols, log_fn):
        """Normalize numeric features using StandardScaler."""
        log_fn(f"Normalizing {len(numeric_cols)} numeric columns")
        self.normalizer = StandardScaler(copy=False)
        scaled_features = self.normalizer.fit_transform(features_df[numeric_cols].values).astype(np.float16)
        features_df[numeric_cols] = pd.DataFrame(scaled_features, index=features_df.index, columns=numeric_cols, dtype=np.float16)
    
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
    

    def delete_cache(self):
        del self.precomputed_data
        self.precomputed_data = None
        gc.collect()

    def _precompute_tensors(self, log_fn):
        """Precompute all tensors for faster data loading."""
        import tqdm
        
        log_fn("Precomputing tensors for faster data loading...")
        total_accounts = len(self.account_ids)

        # Sequential processing with a progress bar
        self.precomputed_data = []
        for idx in tqdm.tqdm(range(total_accounts), desc="Precomputing"):
            self.precomputed_data.append(self._process_account_data(idx=idx))
        
        log_fn(f"Precomputed {len(self.precomputed_data)} tensors")
        
        # Free up memory
        del self.account_groups
        self.account_groups = None
        gc.collect()
    
    def get_shape(self):
        """Return (max_seq_len, feature_dim) tuple describing the shape of features."""
        return self.max_seq_len, self.feature_dim
    
    def get_fraud_labels(self):
        """Return array of fraud labels."""
        return self.fraud_bools
    
    # def get_idx_account(self, account_id):
    #     """Return array of account IDs."""
    #     return self.account_map[account_id]
    
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
            item = self.precomputed_data[idx]
        else:
            item = self._process_account_data(idx=idx)
        
        # Extract features and determine sequence length
        features = item['features']
        seq_len = features.shape[0]
        label = item['label']

        if self.highest_input:
            highest_input_account = self.account_ids_highest_map.get(self.account_ids[idx], None)
            if highest_input_account is None:
                highest_input_account = self.account_ids[idx] # if no external account is present, use the account itself
            highest_input_idx = self.account_map[highest_input_account]
            if self.precompute:
                item2 = self.precomputed_data[highest_input_idx]
            else:
                item2 = self._process_account_data(idx=highest_input_idx)
            
            border = np.ones((1, self.feature_dim), dtype=np.float16) * -1
            features = np.concatenate([features, border, item2['features']], axis=0)
            seq_len = features.shape[0]
            # label = torch.from_numpy(np.concatenate([label, item2['label']], axis=0))

        # Compute evenly spaced indices in [0, max_seq_len-1]
        # We do this to be able to pass more information through the cnn layers. not sure if it actually works that way, but worht a try
        # the idea is essntially this: the cnn layers condense the samples 
        # currently we need to fill up to 2048 timesteps, but often we less than a hundred -> cnn for condensation
        # with a fill in the beginning the cnn would condense the information to the beginning of the sequence
        # the idea with spacing them out is that the cnn either has two transactions next to each
        indices = torch.tensor(np.linspace(0, self.max_seq_len - 1, num=seq_len, dtype=int))
        
        # Create padded_features and place features at the computed indices
        padded_features = torch.zeros((self.max_seq_len, self.feature_dim), dtype=torch.float16)
        padded_features[indices] = torch.from_numpy(features).to(torch.float16)
        
        # For masking, prepare a contiguous tensor (first seq_len rows) so that mask_features works as intended
        if self.mask:
            contiguous = torch.zeros((self.max_seq_len, self.feature_dim), dtype=torch.float16)
            contiguous[:seq_len] = torch.from_numpy(features)
            masked_contiguous, masked_contiguous_pos = self.mask_features(contiguous, seq_len)

            # Scatter the masked contiguous results back to their distributed positions
            masked_features = padded_features.clone()
            masked_pos = torch.zeros_like(padded_features, dtype=torch.bool)
            for i, pos in enumerate(indices):
                masked_features[pos] = masked_contiguous[i]
                masked_pos[pos] = masked_contiguous_pos[i]
        else:
            masked_features, masked_pos = torch.empty(0), torch.empty(0)
        
        # Process external account ids using the same spacing
        ext_enc = item['external_account_ids_enc']
        if self.highest_input and highest_input_account is not None:
            ext_enc = np.concatenate([ext_enc, [-1], item2['external_account_ids_enc']], axis=0)
        external_full = torch.zeros((self.max_seq_len, 1), dtype=torch.int)
        external_full[indices] = torch.from_numpy(ext_enc).unsqueeze(1).int()

        
        return dict(masked_features=masked_features,
                    masked_pos=masked_pos,
                    padded_features=padded_features,
                    label=label,
                    account_id_enc=item['account_id_enc'],
                    external_account_ids_enc=external_full)
    
    
    # @profile
    def _process_account_data(self, idx):
        """
        Process a single account's transactions into tensor format.
        
        Args:
            idx: Index of the account in self.account_ids (used for getting label)
        
        Returns:
            Dictionary containing transaction features and metadata
        """
        account_id = self.account_ids_enc[idx]
        account_transactions = self.account_groups.get_group(account_id)
        
        # Extract features and convert to tensor
        features = account_transactions[self.feature_cols][:self.max_seq_len].to_numpy()
        
        # Get label for this account
        label = torch.Tensor([self.fraud_bools[idx]])

        # encode the account id belonging to these transactions
        account_id_enc = torch.Tensor([account_id])
        
        # encode the external account ids belonging to each transactions
        ext_enc = account_transactions['External'][:self.max_seq_len].to_numpy()

        # Padding will be applied in __getitem__
        return dict(features=features,
                    label=label,
                    account_id_enc=account_id_enc,
                    external_account_ids_enc=ext_enc)

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

    # assert no nan values are present
    if x_res.drop(NA_COLS, axis=1).isnull().values.any():
        raise ValueError('nan values found in x_res')
    
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
    from tqdm import tqdm
    
    num_workers = 0
    val_loader = DataLoader(
        prepare_dataset('ver05', mask=False, log_fn=print, fn=load_val, max_seq_len=1024, highest_input=True),
        shuffle=False,
        drop_last=False,
        batch_size=32,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=2 if num_workers > 0 else None,
    )  

    account_ids = []
    fraud_bools = []
    for i, batch in enumerate(tqdm(val_loader)):
        account_id, label = batch['account_id_enc'], batch['label']
        # masked_features, masked_pos, padded_features, label, account_id, external_account_ids_enc = batch
        account_ids.extend(account_id.flatten())
        fraud_bools.extend(label.flatten())

    account_ids = np.array(account_ids)
    fraud_bools = np.array(fraud_bools)

    d = val_loader.dataset

    account_ids_accoring_to_prop = np.array(d.account_ids_enc)
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