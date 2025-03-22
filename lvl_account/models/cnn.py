import torch
import pytorch_lightning as pl
from torch import nn
import torch.nn.functional as F

class Classifier(pl.LightningModule):
    """
    Enhanced CNN-based model for fraud detection with FC layers between CNN blocks.
    Uses adaptive pooling to handle variable sequence lengths.
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
        
        # Calculate optimal pooling sizes for 2048 sequence length
        # After first CNN block with pooling (2048 -> 1024)
        self.pool1_size = 32  # Reduced from 1024 to 32 for efficiency
        # After second CNN block with pooling (1024 -> 512)
        self.pool2_size = 16   # Reduced from 512 to 16
        # After third CNN block with pooling (512 -> 256)
        self.pool3_size = 8    # Reduced from 256 to 8
        
        # CNN Block 1
        self.cnn_block1 = nn.Sequential(
            nn.Conv1d(feature_dim, hidden_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout),
            
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.SiLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Dropout(dropout),
        )
        
        # Adaptive pooling and FC after CNN block 1
        self.pool1 = nn.AdaptiveAvgPool1d(self.pool1_size)
        self.fc1 = nn.Sequential(
            nn.Linear(hidden_dim * self.pool1_size, hidden_dim),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout)
        )
        
        # CNN Block 2
        self.cnn_block2 = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim*2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(hidden_dim*2),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout),
            
            nn.Conv1d(hidden_dim*2, hidden_dim*2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(hidden_dim*2),
            nn.SiLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Dropout(dropout),
        )
        
        # Adaptive pooling and FC after CNN block 2
        self.pool2 = nn.AdaptiveAvgPool1d(self.pool2_size)
        self.fc2 = nn.Sequential(
            nn.Linear(hidden_dim*2 * self.pool2_size, hidden_dim*2),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout)
        )
        
        # CNN Block 3
        self.cnn_block3 = nn.Sequential(
            nn.Conv1d(hidden_dim*2, hidden_dim*4, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(hidden_dim*4),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout),
            
            nn.Conv1d(hidden_dim*4, hidden_dim*4, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(hidden_dim*4),
            nn.SiLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Dropout(dropout),
        )
        
        # Adaptive pooling and FC after CNN block 3
        self.pool3 = nn.AdaptiveAvgPool1d(self.pool3_size)
        self.fc3 = nn.Sequential(
            nn.Linear(hidden_dim*4 * self.pool3_size, hidden_dim*4),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout)
        )
        
        # CNN Block 4 (Final)
        self.cnn_block4 = nn.Sequential(
            nn.Conv1d(hidden_dim*4, hidden_dim*2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(hidden_dim*2),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout),
            
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

    def forward(self, x):
        """
        Args:
            x: Transaction sequences [batch_size, seq_len, feature_dim]            
        Returns:
            Fraud probability logits
        """
        batch_size = x.shape[0]
        
        # Permute to [batch_size, feature_dim, seq_len] for Conv1d
        x = x.permute(0, 2, 1)
        
        # CNN Block 1
        x = self.cnn_block1(x)
        
        # Apply adaptive pooling and FC layer
        x_pooled = self.pool1(x)
        x_flat = x_pooled.view(batch_size, -1)
        fc_out1 = self.fc1(x_flat)
        
        # Reshape back for CNN - use the fixed pooled size
        x = fc_out1.view(batch_size, -1, 1).expand(batch_size, -1, self.pool1_size)
        
        # CNN Block 2
        x = self.cnn_block2(x)
        
        # Apply adaptive pooling and FC layer
        x_pooled = self.pool2(x)
        x_flat = x_pooled.view(batch_size, -1)
        fc_out2 = self.fc2(x_flat)
        
        # Reshape back for CNN - use the fixed pooled size
        x = fc_out2.view(batch_size, -1, 1).expand(batch_size, -1, self.pool2_size)
        
        # CNN Block 3
        x = self.cnn_block3(x)
        
        # Apply adaptive pooling and FC layer
        x_pooled = self.pool3(x)
        x_flat = x_pooled.view(batch_size, -1)
        fc_out3 = self.fc3(x_flat)
        
        # Reshape back for CNN - use the fixed pooled size
        x = fc_out3.view(batch_size, -1, 1).expand(batch_size, -1, self.pool3_size)
        
        # CNN Block 4 (Final)
        x = self.cnn_block4(x)
        
        # Flatten and pass through classifier
        x = x.view(batch_size, -1)
        logits = self.classifier(x)
        
        return logits
    
    def training_step(self, batch, batch_idx):
        # Updated for dictionary return from __getitem__
        x = batch['padded_features']
        y = batch['label']
        logits = self(x)
        y = y.float()  # Ensure labels are float for BCE loss
        
        # Calculate weighted BCE loss - giving higher weight to minority class (fraud)
        # Since fraud is 12% of the dataset, we use pos_weight = 88/12 ≈ 7.33
        pos_weight = torch.tensor([7.33], device=logits.device)
        loss = F.binary_cross_entropy_with_logits(logits.view(-1), y.view(-1), pos_weight=pos_weight)
        self.log("train_loss", loss, prog_bar=True, on_epoch=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        # Updated for dictionary return from __getitem__
        x = batch['padded_features']
        y = batch['label']
        logits = self(x)
        
        y = y.float()
        pos_weight = torch.tensor([7.33], device=logits.device)
        val_loss = F.binary_cross_entropy_with_logits(logits.view(-1), y.view(-1), pos_weight=pos_weight)
        probs = torch.sigmoid(logits.view(-1))
        preds = (probs > 0.5).float()
        
        # Calculate F1 score specifically for fraud class (class 1)
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
        x = batch['padded_features']
        
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
    
    # Test with random input
    batch_size, seq_len, feature_dim = 4, 100, 68
    x = torch.randn(batch_size, seq_len, feature_dim)
    mask = torch.ones(batch_size, seq_len)
    output = model(x, mask)
    print(f"Output shape: {output.shape}")
