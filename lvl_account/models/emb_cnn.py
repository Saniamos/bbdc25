import torch
import pytorch_lightning as pl
from torch import nn
import torch.nn.functional as F
import numpy as np

# as determined by uploads, see readme
N_FRAUDSTERS = {
    'train': 1411,
    'val': 1472,
    'test': 1267
}

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
        min_lr=1e-5,
        num_accounts=50_000
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
        
        # Build the CNN model with 8 layers
        # Note: update input channels to include extra embedding dimension
        self.cnn_layers = nn.Sequential(
            # Layer 1: Changed input channels to (feature_dim + account_embed_dim)
            nn.Conv1d(feature_dim, hidden_dim, kernel_size=3, padding=1, bias=False),
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

    def forward(self, x, external_account_ids_enc=None):
        """
        Updated to ignore external_account_ids_enc and learn embedding from x.
        """
        batch_size, seq_len, _ = x.shape

        # Permute to [batch_size, feature_dim + 1, seq_len] for Conv1d
        x = x.permute(0, 2, 1)
        
        # Apply CNN layers
        x = self.cnn_layers(x)

        TODO: implement this. i think there should be somehting here, where we can actually learn the embedding
        ie we should have accounts interacting in a large enough batch and thus we should be able to somehow integrate this into the architecture
        
        # Flatten and pass through classifier
        x = x.view(batch_size, -1)
        logits = self.classifier(x)
        
        return logits
    
    def update_account_predictions(self, account_ids:torch.Tensor, predictions):
        """
        Update the account predictions dictionary with new predictions
        
        Args:
            account_ids: List of account IDs
            predictions: Corresponding predictions (probabilities)
        """
        self.account_predictions[account_ids.int().squeeze()] = predictions.squeeze()
        
    def training_step(self, batch, batch_idx):
        x = batch['padded_features']
        y = batch['label']
        account_ids = batch['account_id_enc']
        external_account_ids_enc = batch['external_account_ids_enc']
        
        logits = self(x)
        y = y.float()  # Ensure labels are float for BCE loss
        
        pos_weight = torch.tensor([8], device=logits.device)
        loss = F.binary_cross_entropy_with_logits(
            logits.view(-1), y.view(-1), 
            pos_weight=pos_weight
        )       
        
        # Log metrics including the new regularization loss
        self.log("train_loss", loss, prog_bar=True, on_epoch=True, sync_dist=True)
        
        return loss

    def validation_step(self, batch, batch_idx):
        # Updated for dictionary return from __getitem__
        x = batch['padded_features']
        y = batch['label']
        account_ids = batch['account_id_enc']
        external_account_ids_enc = batch['external_account_ids_enc']
        
        # Forward pass
        logits = self(x)
        y = y.float()
        
        # Calculate weighted BCE loss
        pos_weight = torch.tensor([8], device=logits.device)
        val_loss = F.binary_cross_entropy_with_logits(
            logits.view(-1), y.view(-1),
            pos_weight=pos_weight
        )
        
        # Calculate metrics
        probs = torch.sigmoid(logits.view(-1))
        preds = (probs > 0.5).float()
        
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
        # Updated for dictionary return from __getitem__
        x = batch['padded_features']
        external_account_ids_enc = batch['external_account_ids_enc']
        
        # Forward pass
        logits = self(x)
        
        # Get probabilities
        probs = torch.sigmoid(logits)
        
        # Get binary predictions
        preds = (probs > 0.5).float()
        
        return {"probs": probs, "preds": preds}

    


if __name__ == "__main__":
    # Test model creation
    model = Classifier(feature_dim=68)
    print(model)
    
    # Test with random input: now provide a proper 1D account ID for each sample
    batch_size, seq_len, feature_dim = 4, 100, 68
    x = torch.randn(batch_size, seq_len, feature_dim)
    # Now external_account_ids_enc is not needed.
    output = model(x)
    print(f"Output shape: {output.shape}")
