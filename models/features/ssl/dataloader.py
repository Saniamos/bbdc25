import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from sklearn.calibration import LabelEncoder
from transformers import PreTrainedTokenizer


class AccountTransactionDataset(Dataset):
    """
    Dataset for loading all transactions for an account along with its fraud label.
    """
    
    def __init__(self, 
                 features_df: pd.DataFrame, 
                 labels_df: pd.DataFrame, 
                 feature_cols: Optional[List[str]] = None,
                 max_seq_len: Optional[int] = None):
        """
        Initialize the dataset with feature and label dataframes.
        
        Args:
            features_df: DataFrame containing transaction features with AccountID column
            labels_df: DataFrame containing fraud labels with AccountID and Fraudster columns
            feature_cols: List of feature column names to use (if None, use all except AccountID)
            max_seq_len: Maximum sequence length for padding (if None, determined from data)
        """
        features_df = features_df.copy()
        features_df['Action'] = LabelEncoder().fit_transform(features_df['Action'])
        features_df['External_Type'] = LabelEncoder().fit_transform(features_df['External_Type'])

        # Ensure both dataframes have AccountID
        assert "AccountID" in features_df.columns, "Features DataFrame must have AccountID column"
        assert "AccountID" in labels_df.columns, "Labels DataFrame must have AccountID column"
        assert "Fraudster" in labels_df.columns, "Labels DataFrame must have Fraudster column"
        
        # Create account to label mapping
        self.account_to_label = labels_df.set_index('AccountID')['Fraudster'].to_dict()
        
        # Determine feature columns
        if feature_cols is None:
            self.feature_cols = [col for col in features_df.columns if col != "AccountID" and col != 'External']
        else:
            self.feature_cols = feature_cols
        
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
    
    def get_shape(self):
        return self.max_seq_len, self.feature_dim

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
        
        # Get label for this account
        label = self.account_to_label[account_id]
        label = torch.FloatTensor([label])
        
        return padded_features, label
    

def load_datasets(X_list):
    y_res = []
    x_res = []
    for i, d in enumerate(X_list):
        source_suffix = f'yh{i}' # random id that should not occour in accountid or external
        x_df = pd.read_parquet(d)
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
    return x_res, y_res

def load_train(ver):
    return load_datasets([f"~/Repositories/bbdc25/task/train_set/x_train.{ver}.parquet",])

def load_val(ver):
    return load_datasets([f"~/Repositories/bbdc25/task/val_set/x_val.{ver}.parquet",])

def load_test(ver):
    return load_datasets([f"~/Repositories/bbdc25/task/test_set/x_test.{ver}.parquet",])

def load_all(ver):
    return load_datasets([f"~/Repositories/bbdc25/task/train_set/x_train.{ver}.parquet",
                          f"~/Repositories/bbdc25/task/val_set/x_val.{ver}.parquet",
                          f"~/Repositories/bbdc25/task/test_set/x_test.{ver}.parquet"])

class TransactionTokenizerDataset(Dataset):
    """
    Dataset for tokenizing transaction data for Masked Language Modeling with BERT.
    """
    
    def __init__(self, 
                 features_df: pd.DataFrame,
                 tokenizer: PreTrainedTokenizer,
                 feature_cols: Optional[List[str]] = None,
                 max_length: int = 512,
                 mlm_probability: float = 0.15):
        """
        Initialize the tokenizer dataset.
        
        Args:
            features_df: DataFrame containing transaction features
            tokenizer: Hugging Face tokenizer
            feature_cols: List of feature column names to use
            max_length: Maximum sequence length for the tokenizer
            mlm_probability: Probability of masking tokens for MLM
        """
        self.features_df = features_df.copy()
        self.tokenizer = tokenizer
        
        # Determine feature columns
        if feature_cols is None:
            self.feature_cols = [col for col in features_df.columns if col not in ["AccountID", "External"]]
        else:
            self.feature_cols = feature_cols
            
        self.max_length = max_length
        self.mlm_probability = mlm_probability
        
        # Convert each transaction to a text representation
        self.texts = []
        for _, row in self.features_df.iterrows():
            text = " ".join([f"{col}: {row[col]}" for col in self.feature_cols])
            self.texts.append(text)
        
        print(f"Loaded dataset with {len(self.texts)} transactions")
        print(f"Sample transaction text: {self.texts[0][:100]}...")
    
    def __len__(self) -> int:
        """Return the number of transactions."""
        return len(self.texts)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get tokenized representation of a transaction with MLM.
        
        Args:
            idx: Index of the transaction to retrieve
        
        Returns:
            Dict with input_ids, attention_mask, and labels for BERT MLM
        """
        text = self.texts[idx]
        
        # Tokenize the text
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        # Get necessary tensors and squeeze the batch dimension
        input_ids = encoding["input_ids"].squeeze()
        attention_mask = encoding["attention_mask"].squeeze()
        
        # Create labels for MLM (copy of input_ids)
        labels = input_ids.clone()
        
        # Create masked version of input_ids for MLM
        masked_indices = torch.bernoulli(torch.full(input_ids.shape, self.mlm_probability)).bool() & attention_mask.bool()
        
        # For 80% of masked tokens, replace with [MASK]
        indices_mask = torch.bernoulli(torch.full(input_ids.shape, 0.8)).bool() & masked_indices
        input_ids[indices_mask] = self.tokenizer.mask_token_id
        
        # For 10% of masked tokens, replace with random token
        random_indices = torch.bernoulli(torch.full(input_ids.shape, 0.5)).bool() & masked_indices & ~indices_mask
        random_tokens = torch.randint(len(self.tokenizer), input_ids.shape, dtype=torch.long)
        input_ids[random_indices] = random_tokens[random_indices]
        
        # For remaining 10% of masked tokens, keep original token
        
        # Set labels to -100 for non-masked tokens (don't include in loss)
        labels[~masked_indices] = -100
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }


def prepare_mlm_datasets(version='ver03', tokenizer=None, max_length=512, mlm_probability=0.15):
    """
    Prepare tokenized datasets for MLM training.
    
    Args:
        version: Data version to use
        tokenizer: Tokenizer to use for encoding
        max_length: Maximum sequence length
        mlm_probability: Probability of masking tokens for MLM
        
    Returns:
        train_dataset, val_dataset, test_dataset
    """
    # Load the data
    x_train_df, y_train_df = load_train(version)
    x_val_df, y_val_df = load_val(version)
    x_test_df, y_test_df = load_test(version)
    
    # Create datasets
    train_dataset = TransactionTokenizerDataset(
        x_train_df, tokenizer, max_length=max_length, mlm_probability=mlm_probability
    )
    
    val_dataset = TransactionTokenizerDataset(
        x_val_df, tokenizer, max_length=max_length, mlm_probability=mlm_probability
    )
    
    test_dataset = TransactionTokenizerDataset(
        x_test_df, tokenizer, max_length=max_length, mlm_probability=mlm_probability
    )
    
    return train_dataset, val_dataset, test_dataset

    
if __name__ == "__main__":
    x_train_df, y_train_df = load_all('ver03')

    dataset = AccountTransactionDataset(x_train_df, y_train_df)
    print(dataset[0])
    print(dataset[0][0].shape)