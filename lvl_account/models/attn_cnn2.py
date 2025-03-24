import torch
import pytorch_lightning as pl
from torch import nn
import torch.nn.functional as F

class AttentionBlock(nn.Module):
    """Enhanced self-attention block with pre-normalization"""
    def __init__(self, embed_dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim)
        )
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # Pre-normalization for better training stability
        residual = x
        x_norm = self.norm1(x)
        attn_output, _ = self.attention(x_norm, x_norm, x_norm)
        x = residual + self.dropout(attn_output)
        
        # FFN with pre-normalization
        residual = x
        x_norm = self.norm2(x)
        ffn_output = self.ffn(x_norm)
        x = residual + self.dropout(ffn_output)
        return x

class ConvDownsampleBlock(nn.Module):
    """Lightweight convolutional block for downsampling"""
    def __init__(self, in_channels, out_channels, dropout=0.1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.SiLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),  # Downsample sequence length
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        return self.conv(x)

class Classifier(pl.LightningModule):
    """
    Attention-focused model for fraud detection with minimal CNN layers
    """
    def __init__(
        self,
        feature_dim,
        pretrained_model_path=None,  # Not used but kept for compatibility
        weight_decay=0.01,
        learning_rate=1e-4,
        dropout=0.2,
        freeze_bert=False,  # Not used but kept for compatibility
        hidden_dim=256,
        warmup_steps=100,
        max_lr=5e-4,
        min_lr=1e-5,
        num_attention_layers=6,  # Number of attention blocks
        num_heads=8  # Number of attention heads
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
        
        # 1. Initial embedding with 1x1 convolution
        self.embedding = nn.Conv1d(feature_dim, hidden_dim, kernel_size=1)
        
        # 2. Just two CNN blocks for downsampling (reduces sequence length by 4x)
        self.cnn_blocks = nn.ModuleList([
            ConvDownsampleBlock(hidden_dim, hidden_dim, dropout),
            ConvDownsampleBlock(hidden_dim, hidden_dim, dropout)
        ])
        
        # 3. Attention-focused processing blocks
        self.attention_blocks = nn.ModuleList([
            AttentionBlock(hidden_dim, num_heads=num_heads, dropout=dropout)
            for _ in range(num_attention_layers)
        ])
        
        # 4. Final pooling and classification
        self.pooling = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.AdaptiveAvgPool1d(1)
        )
        
        self.classifier = nn.Linear(hidden_dim, 1)
        
        # Initialize weights for faster convergence
        self._init_weights()
        self.account_predictions = {}  # Initialize account predictions
        
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
        batch_size, seq_len, _ = x.shape
        
        # 1. Embed and prepare for CNN (B, F, S)
        x = x.permute(0, 2, 1)
        x = self.embedding(x)
        
        # 2. Apply two CNN blocks for downsampling
        for cnn_block in self.cnn_blocks:
            x = cnn_block(x)
        
        # 3. Prepare for attention (B, S/4, D)
        x = x.permute(0, 2, 1)
        
        # 4. Apply attention blocks
        for attn_block in self.attention_blocks:
            x = attn_block(x)
        
        # 5. Apply layer norm before pooling
        x = self.pooling[0](x)  # LayerNorm
        
        # 6. Permute back for pooling (B, D, S/4)
        x = x.permute(0, 2, 1)
        
        # 7. Pool and get final embedding
        x = self.pooling[1](x)  # AdaptiveAvgPool1d
        x = x.squeeze(-1)  # (B, D)
        
        # 8. Final classification
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
        
        # Calculate weighted BCE loss for consistency with training
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
