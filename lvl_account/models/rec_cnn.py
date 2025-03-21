import torch
import pytorch_lightning as pl
from torch import nn
import torch.nn.functional as F
import numpy as np

class Classifier(pl.LightningModule):
    """
    Recurrent CNN-based model for fraud detection in transaction sequences.
    Uses its own predictions as additional features for future predictions.
    """
    def __init__(
        self,
        feature_dim,
        pretrained_model_path=None,  # Not used but kept for compatibility
        weight_decay=0.01,
        learning_rate=1e-4,
        dropout=0.2,
        freeze_bert=False,  # Not used but kept for compatibility
        hidden_dim=128,
        warmup_steps=100,
        max_lr=5e-4,
        min_lr=1e-5
    ):
        super().__init__()
        self.save_hyperparameters()
        
        # Architecture constants
        self.feature_dim = feature_dim
        self.dropout_rate = dropout
        self.hidden_dim = hidden_dim
        self.warmup_steps = warmup_steps
        self.max_lr = max_lr
        self.min_lr = min_lr
        
        # Dictionary to store account_id -> fraud_score mappings
        self.account_predictions = {}
        # Default value for accounts with no previous predictions
        self.default_prediction = 0.0
        
        # Build the CNN model with 8 layers
        # Add +1 to feature_dim to account for the additional fraud score feature
        self.cnn_layers = nn.Sequential(
            # Layer 1
            nn.Conv1d(feature_dim + 1, hidden_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.SiLU(inplace=True),  # More efficient activation function
            nn.Dropout(dropout),
            
            # Layer 2
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.SiLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Dropout(dropout),
            
            # Layer 3-4: Second Conv Block
            nn.Conv1d(hidden_dim, hidden_dim*2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(hidden_dim*2),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout),
            
            # Layer 4
            nn.Conv1d(hidden_dim*2, hidden_dim*2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(hidden_dim*2),
            nn.SiLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Dropout(dropout),
            
            # Layer 5-6: Third Conv Block
            nn.Conv1d(hidden_dim*2, hidden_dim*4, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(hidden_dim*4),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout),
            
            # Layer 6
            nn.Conv1d(hidden_dim*4, hidden_dim*4, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(hidden_dim*4),
            nn.SiLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Dropout(dropout),
            
            # Layer 7-8: Final Conv Block
            nn.Conv1d(hidden_dim*4, hidden_dim*2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(hidden_dim*2),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout),
            
            # Layer 8
            nn.Conv1d(hidden_dim*2, hidden_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.SiLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),  # Global average pooling
        )
        
        # Final classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout/2),
            nn.Linear(64, 1)
        )
        
        # Initialize weights for faster convergence
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using He initialization for better training dynamics"""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def add_prediction_feature(self, x, account_ids):
        """
        Add a feature channel to input data containing previous fraud predictions
        
        Args:
            x: Transaction sequences [batch_size, seq_len, feature_dim]
            account_ids: List of account IDs corresponding to each sequence in the batch
            
        Returns:
            Enhanced input with prediction feature [batch_size, seq_len, feature_dim + 1]
        """
        batch_size, seq_len, _ = x.shape
        
        # Create a tensor to hold the fraud score feature
        fraud_scores = torch.zeros(batch_size, seq_len, 1, device=x.device)
        
        # Fill in fraud scores for each account in the batch
        for i, account_id in enumerate(account_ids):
            # Get previous prediction or default value
            account_id_key = str(account_id)  # Convert to string for dict keys
            score = self.account_predictions.get(account_id_key, self.default_prediction)
            # Fill the entire sequence for this account with the prediction
            fraud_scores[i, :, 0] = score
            
        # Concatenate the fraud score feature to the input
        enhanced_x = torch.cat([x, fraud_scores], dim=2)
        
        return enhanced_x

    def forward(self, x, account_ids=None):
        """
        Args:
            x: Transaction sequences [batch_size, seq_len, feature_dim]
            account_ids: Optional list of account IDs for the batch
            
        Returns:
            Fraud probability logits
        """
        batch_size, seq_len, _ = x.shape
        
        # If account_ids provided, add fraud predictions as features
        if account_ids is not None:
            x = self.add_prediction_feature(x, account_ids)
        else:
            # No account IDs provided - add zeros as placeholder
            zeros = torch.zeros(batch_size, seq_len, 1, device=x.device)
            x = torch.cat([x, zeros], dim=2)
        
        # Permute to [batch_size, feature_dim + 1, seq_len] for Conv1d
        x = x.permute(0, 2, 1)
        
        # Apply CNN layers
        x = self.cnn_layers(x)
        
        # Flatten and pass through classifier
        x = x.view(batch_size, -1)
        logits = self.classifier(x)
        
        return logits
    
    def update_account_predictions(self, account_ids, predictions):
        """
        Update the account predictions dictionary with new predictions
        
        Args:
            account_ids: List of account IDs
            predictions: Corresponding predictions (probabilities)
        """
        # Update our prediction cache
        for account_id, pred in zip(account_ids, predictions):
            self.account_predictions[str(account_id)] = float(pred)
    
    def training_step(self, batch, batch_idx):
        masked_seqs, masked_pos, x, y = batch  # Unpack (masked_seqs, masked_pos, orig_seqs, label)
        
        # Extract account IDs from the batch
        # Note: This assumes account IDs are available in the dataset
        # You'll need to modify your dataloader to include this information
        try:
            account_ids = batch[4] if len(batch) > 4 else None
        except:
            account_ids = None  # Fallback if not available
        
        # Forward pass with account IDs to incorporate previous predictions
        logits = self(x, account_ids)
        y = y.float()  # Ensure labels are float for BCE loss
        
        # Calculate weighted BCE loss
        pos_weight = torch.tensor([7.33], device=logits.device)
        loss = F.binary_cross_entropy_with_logits(
            logits.view(-1), y.view(-1), 
            pos_weight=pos_weight
        )
        
        # Update account predictions for next batch/epoch
        if account_ids is not None:
            with torch.no_grad():
                probs = torch.sigmoid(logits.view(-1)).detach().cpu().numpy()
                self.update_account_predictions(account_ids, probs)
          
        # Log metrics
        self.log("train_loss", loss, prog_bar=True, on_epoch=True, sync_dist=True)
        
        return loss

    def validation_step(self, batch, batch_idx):
        masked_seqs, masked_pos, x, y = batch  # Unpack (masked_seqs, masked_pos, orig_seqs, label)
        
        # Extract account IDs from the batch (similar to training_step)
        try:
            account_ids = batch[4] if len(batch) > 4 else None
        except:
            account_ids = None  # Fallback if not available
        
        # Forward pass
        logits = self(x, account_ids)
        y = y.float()
        
        # Calculate weighted BCE loss
        pos_weight = torch.tensor([7.33], device=logits.device)
        val_loss = F.binary_cross_entropy_with_logits(
            logits.view(-1), y.view(-1),
            pos_weight=pos_weight
        )
        
        # Calculate metrics
        probs = torch.sigmoid(logits.view(-1))
        preds = (probs > 0.5).float()
        
        # Update account predictions for next validation batch
        if account_ids is not None:
            with torch.no_grad():
                probs_cpu = probs.detach().cpu().numpy()
                self.update_account_predictions(account_ids, probs_cpu)
        
        # Calculate F1 score for fraud class
        y_true = y.view(-1)
        y_pred = preds
        
        # True positives, false positives, false negatives for FRAUD class
        fraud_tp = torch.logical_and(y_true == 1, y_pred == 1).sum().float()
        fraud_fp = torch.logical_and(y_true == 0, y_pred == 1).sum().float()
        fraud_fn = torch.logical_and(y_true == 1, y_pred == 0).sum().float()
        
        # Precision and recall for FRAUD class with epsilon to avoid division by zero
        epsilon = 1e-7
        fraud_precision = fraud_tp / (fraud_tp + fraud_fp + epsilon)
        fraud_recall = fraud_tp / (fraud_tp + fraud_fn + epsilon)
        
        # F1 score for FRAUD class
        fraud_f1 = 2 * (fraud_precision * fraud_recall) / (fraud_precision + fraud_recall + epsilon)
        
        # Log metrics
        self.log_dict({
            "val_loss": val_loss, 
            "val_fraud_f1": fraud_f1
        }, prog_bar=True, sync_dist=True)
        
        return {
            "val_loss": val_loss,
            "val_fraud_f1": fraud_f1,
        }
    
    def configure_optimizers(self):
        # Use Lion optimizer for faster convergence
        try:
            from lion_pytorch import Lion
            optimizer = Lion(
                self.parameters(),
                lr=self.hparams.learning_rate,
                weight_decay=self.hparams.weight_decay
            )
            print("Using Lion optimizer")
        except ImportError:
            # Fall back to AdamW if Lion is not available
            optimizer = torch.optim.AdamW(
                self.parameters(),
                lr=self.hparams.learning_rate,
                weight_decay=self.hparams.weight_decay
            )
            print("Using AdamW optimizer")
        
        # Use CosineAnnealingWarmRestarts instead of OneCycleLR
        # This scheduler handles resuming from checkpoints better
        scheduler = {
            "scheduler": torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer,
                T_0=5,  # Restart every 5 epochs
                T_mult=1,  # Keep the same cycle length after restart
                eta_min=self.min_lr,  # Minimum learning rate
            ),
            "interval": "epoch",
            "frequency": 1,
            "name": "learning_rate"
        }
        
        return [optimizer], [scheduler]

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        """
        Prediction step for generating fraud probabilities and predictions.
        """
        masked_seqs, masked_pos, x, _ = batch  # Unpack (masked_seqs, masked_pos, orig_seqs, label)
        
        # Extract account IDs from the batch
        try:
            account_ids = batch[4] if len(batch) > 4 else None
        except:
            account_ids = None  # Fallback if not available
        
        # Forward pass
        logits = self(x, account_ids)
        
        # Get probabilities
        probs = torch.sigmoid(logits)
        
        # Get binary predictions
        preds = (probs > 0.5).float()
        
        # Update account predictions
        if account_ids is not None:
            with torch.no_grad():
                probs_cpu = probs.detach().cpu().numpy()
                self.update_account_predictions(account_ids, probs_cpu)
        
        return {"probs": probs, "preds": preds}


if __name__ == "__main__":
    # Test model creation
    model = Classifier(feature_dim=68)
    print(model)
    
    # Test with random input
    batch_size, seq_len, feature_dim = 4, 100, 68
    x = torch.randn(batch_size, seq_len, feature_dim)
    mask = torch.ones(batch_size, seq_len)
    output = model(x, mask)
    print(f"Output shape: {output.shape}")
