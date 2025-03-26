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
    def __init__(
        self,
        feature_dim,
        pretrained_model_path=None,
        weight_decay=0.01,
        learning_rate=1e-4,
        dropout=0.2,
        freeze_bert=False,
        hidden_dim=128,
        warmup_steps=100,
        max_lr=5e-4,
        min_lr=1e-5,
        num_accounts=50_000,
        pred_state_dim=16  # new: dimension of per-sample hidden state via GRU
    ):
        super().__init__()
        self.save_hyperparameters()
        
        self.feature_dim = feature_dim
        self.dropout_rate = dropout
        self.hidden_dim = hidden_dim
        self.warmup_steps = warmup_steps
        self.max_lr = max_lr
        self.min_lr = min_lr
        
        # Instead, add a GRU layer to process the input sequence
        self.gru = nn.GRU(input_size=feature_dim, hidden_size=pred_state_dim, batch_first=True)
        # Map GRU final hidden state to a fraud feature (a scalar)
        self.gru_to_score = nn.Linear(pred_state_dim, 1)
        
        # Build the CNN model: change input channels from (feature_dim+1) to feature_dim 
        self.cnn_layers = nn.Sequential(
            nn.Conv1d(feature_dim, hidden_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.SiLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_dim, hidden_dim*2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(hidden_dim*2),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_dim*2, hidden_dim*2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(hidden_dim*2),
            nn.SiLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_dim*2, hidden_dim*4, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(hidden_dim*4),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_dim*4, hidden_dim*4, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(hidden_dim*4),
            nn.SiLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_dim*4, hidden_dim*2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(hidden_dim*2),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_dim*2, hidden_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.SiLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
        )
        
        # Update classifier to accept hidden_dim+1 features (adding fraud feature)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim+1, 64),
            nn.SiLU(inplace=True),
            nn.Dropout(self.dropout_rate/2),
            nn.Linear(64, 1)
        )
        
        # Persistent GRU state to be carried across batches
        self.hidden_state = None

        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using He initialization"""
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
        batch_size, seq_len, _ = x.shape
        # Initialize or adjust persistent hidden state if needed
        if self.hidden_state is None or self.hidden_state.size(1) != batch_size:
            self.hidden_state = torch.zeros(1, batch_size, self.hparams.pred_state_dim, device=x.device)
        # Use persistent hidden state in GRU forward pass
        _, h = self.gru(x, self.hidden_state)
        # Update persistent hidden state (detach to avoid backprop across batches if undesired)
        self.hidden_state = h.detach()
        fraud_feature = self.gru_to_score(h.squeeze(0))  # (batch_size, 1)
        fraud_feature = fraud_feature.unsqueeze(1).expand(batch_size, seq_len, 1)
        # Process x through CNN.
        x_cnn = x.permute(0, 2, 1)               # (batch_size, feature_dim, seq_len)
        cnn_feat = self.cnn_layers(x_cnn)        # (batch_size, hidden_dim, 1)
        cnn_feat = cnn_feat.view(batch_size, -1) # (batch_size, hidden_dim)
        # Concatenate fraud feature.
        combined = torch.cat([cnn_feat, fraud_feature[:, 0, :]], dim=1)  # (batch_size, hidden_dim+1)
        logits = self.classifier(combined)
        return logits
    
    def training_step(self, batch, batch_idx):
        x = batch['padded_features']
        y = batch['label']
        logits = self(x)
        y = y.float()
        pos_weight = torch.tensor([8], device=logits.device)
        loss = F.binary_cross_entropy_with_logits(logits.view(-1), y.view(-1), pos_weight=pos_weight)
        self.log("train_loss", loss, prog_bar=True, on_epoch=True, sync_dist=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x = batch['padded_features']
        y = batch['label']
        logits = self(x)
        y = y.float()
        pos_weight = torch.tensor([8], device=logits.device)
        val_loss = F.binary_cross_entropy_with_logits(logits.view(-1), y.view(-1), pos_weight=pos_weight)
        probs = torch.sigmoid(logits.view(-1))
        preds = (probs > 0.5).float()
        epsilon = 1e-7
        fraud_tp = torch.logical_and(y.view(-1) == 1, preds == 1).sum().float()
        fraud_fp = torch.logical_and(y.view(-1) == 0, preds == 1).sum().float()
        fraud_fn = torch.logical_and(y.view(-1) == 1, preds == 0).sum().float()
        fraud_precision = fraud_tp / (fraud_tp + fraud_fp + epsilon)
        fraud_recall = fraud_tp / (fraud_tp + fraud_fn + epsilon)
        fraud_f1 = 2 * (fraud_precision * fraud_recall) / (fraud_precision + fraud_recall + epsilon)
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
        
        # Forward pass
        logits = self(x)
        
        # Get probabilities
        probs = torch.sigmoid(logits)
        
        # Get binary predictions
        preds = (probs > 0.5).float()
        
        return {"probs": probs, "preds": preds}

    def reset_account_pred_state(self, num_accounts=None):
        pass  # no longer used since state is not persistent.

    def on_train_epoch_start(self):
        # Optionally do not reset to allow state carry over across the entire training phase
        pass

    def on_validation_start(self):
        # Reset the persistent state before validation to avoid contamination from training
        self.hidden_state = None

    def on_predict_start(self):
        # Reset the persistent state before prediction
        self.hidden_state = None


if __name__ == "__main__":
    # Test model creation
    model = Classifier(feature_dim=68)
    print(model)
    
    # Test with random input
    batch_size, seq_len, feature_dim = 4, 100, 68
    x = torch.randn(batch_size, seq_len, feature_dim)
    mask = torch.ones(batch_size, seq_len)
    output = model(x)
    print(f"Output shape: {output.shape}")
