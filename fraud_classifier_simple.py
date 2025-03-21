import torch
import pytorch_lightning as pl
from torch import nn
import torch.nn.functional as F

class Classifier(pl.LightningModule):
    """
    Simple CNN-based model for fraud detection in transaction sequences.
    Uses 8 layers of CNNs to process transaction data.
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
        
        # Build the CNN model with 8 layers
        # Layer 1-2: First Conv Block - optimized to use SiLU activation for better gradients
        self.cnn_layers = nn.Sequential(
            # Layer 1
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

    def forward(self, x, mask=None):
        """
        Args:
            x: Transaction sequences [batch_size, seq_len, feature_dim]
            mask: Optional mask for valid positions [batch_size, seq_len]
            
        Returns:
            Fraud probability logits
        """
        batch_size, seq_len, _ = x.shape
        
        # Apply masking if provided - efficient implementation
        if mask is not None:
            # Apply masking more efficiently - avoiding expand_as which creates new tensors
            x = x * mask.unsqueeze(-1)
        
        # Permute to [batch_size, feature_dim, seq_len] for Conv1d
        x = x.permute(0, 2, 1)
        
        # Apply CNN layers
        x = self.cnn_layers(x)
        
        # Flatten and pass through classifier
        x = x.view(batch_size, -1)
        logits = self.classifier(x)
        
        return logits
    
    def training_step(self, batch, batch_idx):
        x, _, _, y = batch  # Unpack (masked_seqs, masked_pos, orig_seqs, label)
        
        # Create sequence mask (identifies non-padding elements)
        seq_mask = torch.any(x != 0, dim=-1).float()  # [batch_size, seq_len]
        
        # Forward pass
        logits = self(x, seq_mask)
        y = y.float()  # Ensure labels are float for BCE loss
        
        # Calculate weighted BCE loss - giving higher weight to minority class (fraud)
        # Since fraud is 12% of the dataset, we use pos_weight = 88/12 ≈ 7.33
        pos_weight = torch.tensor([7.33], device=logits.device)
        loss = F.binary_cross_entropy_with_logits(
            logits.view(-1), y.view(-1), 
            pos_weight=pos_weight
        )
        
        # Log metrics
        self.log("train_loss", loss, prog_bar=True, on_epoch=True, sync_dist=True)
        
        return loss

    def validation_step(self, batch, batch_idx):
        x, _, _, y = batch
        
        # Create sequence mask - optimized
        seq_mask = torch.any(x != 0, dim=-1).float()
        
        # Forward pass
        logits = self(x, seq_mask)
        y = y.float()
        
        # Calculate weighted BCE loss for consistency with training
        pos_weight = torch.tensor([7.33], device=logits.device)
        val_loss = F.binary_cross_entropy_with_logits(
            logits.view(-1), y.view(-1),
            pos_weight=pos_weight
        )
        
        # For proper evaluation metrics, use unweighted predictions
        probs = torch.sigmoid(logits.view(-1))
        preds = (probs > 0.5).float()
        correct = (preds == y.view(-1)).float().sum()
        total = y.numel()
        
        # Log metrics
        self.log_dict({
            "val_loss": val_loss, 
            "val_acc": correct / total
        }, prog_bar=True, sync_dist=True)
        
        return {"val_loss": val_loss}
    
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
        
        # Add learning rate scheduler with warmup and cosine decay
        scheduler = {
            "scheduler": torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=self.max_lr,
                total_steps=self.trainer.estimated_stepping_batches,
                pct_start=0.1,  # Warmup for 10% of training
                div_factor=self.max_lr/self.hparams.learning_rate,  # Initial LR
                final_div_factor=self.hparams.learning_rate/self.min_lr,  # Final LR
                three_phase=False,
                anneal_strategy='cos',
            ),
            "interval": "step",
            "frequency": 1,
            "name": "learning_rate"
        }
        
        return [optimizer], [scheduler]

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        """
        Prediction step for generating fraud probabilities and predictions.
        """
        x, _, _, _ = batch
        
        # Create sequence mask - optimized
        seq_mask = torch.any(x != 0, dim=-1).float()
        
        # Forward pass
        with torch.cuda.amp.autocast():  # Use mixed precision for inference
            logits = self(x, seq_mask)
            
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
