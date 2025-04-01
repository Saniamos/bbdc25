import torch
import pytorch_lightning as pl
from torch import nn
import torch.nn.functional as F
import numpy as np
from models.simple_cnn import Classifier as SimpleCNN
from models.attn_cnn import AttentionBlock


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
        pretrained_model_path="/home/yale/Repositories/bbdc25/lvl_account/saved_models/simple_cnn/logs/simple_cnn/version_58/simple_cnn-epoch=25-val_fraud_f1=0.9965.ckpt",
        weight_decay=0.01,
        learning_rate=1e-4,
        dropout=0.2,
        freeze_pretrained_model=False,
        hidden_dim=256,
        warmup_steps=100,
        max_lr=5e-4,
        min_lr=1e-5,
        encoder_dim = 64,
        num_accounts=50_000
    ):
        super().__init__()
        self.save_hyperparameters()
        
        self.feature_dim = feature_dim
        self.dropout_rate = dropout
        self.hidden_dim = hidden_dim
        self.warmup_steps = warmup_steps
        self.max_lr = max_lr
        self.min_lr = min_lr
        
        # Instead of building the CNN layers manually, load pretrained cnn_layers from SimpleCNN.
        simple_cnn = SimpleCNN(feature_dim=self.hparams.feature_dim)
        pretrained_cnn = torch.load(self.hparams.pretrained_model_path, map_location=torch.device('cpu'))
        if "state_dict" in pretrained_cnn:
            pretrained_cnn = pretrained_cnn["state_dict"]
        # Load the checkpoint into the simple_cnn model (using strict=False to bypass missing keys)
        simple_cnn.load_state_dict(pretrained_cnn, strict=False)
        # Use the pretrained cnn_layers and freeze them.
        self.cnn_layers = simple_cnn.cnn_layers
        for param in self.cnn_layers.parameters():
            param.requires_grad = not freeze_pretrained_model
        
        # Default value for accounts with no previous predictions
        self.default_prediction = 0.5
        # Dictionary to store account_id -> fraud_score mappings
        self.reset_account_pred_state(num_accounts=num_accounts)
        
        self.account_enc = nn.Sequential(
            nn.Linear(2048, encoder_dim),
            AttentionBlock(encoder_dim, num_heads=4, dropout=dropout),            
            nn.Linear(encoder_dim, encoder_dim)
        )

        # Final classifier
        self.classifier = nn.Sequential(
            # Two attention blocks
            AttentionBlock(hidden_dim + encoder_dim, num_heads=4, dropout=dropout),
            AttentionBlock(hidden_dim + encoder_dim, num_heads=4, dropout=dropout),
            
            # Final classification layer
            nn.Linear(hidden_dim + encoder_dim, 1)
        )

    def forward(self, x, external_account_ids_enc):
        """
        Args:
            x: Transaction sequences [batch_size, seq_len, feature_dim]
            external_account_ids_enc: Optional list of account IDs for the batch
            
        Returns:
            Fraud probability logits
        """
        batch_size, seq_len, _ = x.shape
        
        # Permute to [batch_size, feature_dim + 1, seq_len] for Conv1d
        x = x.permute(0, 2, 1)
        
        # Apply CNN layers
        x = self.cnn_layers(x).permute(0, 2, 1)

        fraud_scores = self.account_predictions[external_account_ids_enc]
        # if self.overwrite_rand_account_predictions:
        #     progress = self.current_epoch / (self.trainer.max_epochs / 2) if self.trainer and self.trainer.max_epochs > 0 else 0
        #     random_ratio = 0.30 * (1 - progress)
        #     fraud_scores = torch.where(torch.rand_like(fraud_scores) < random_ratio, 
        #                             torch.rand_like(fraud_scores), 
        #                             fraud_scores)
        fraud_scores = self.account_enc(fraud_scores.permute(0, 2, 1))
        x = torch.cat([x, fraud_scores], dim=2)
        
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
        
        logits = self(x, external_account_ids_enc)
        y = y.float()  # Ensure labels are float for BCE loss
        
        pos_weight = torch.tensor([8], device=logits.device)
        loss = F.binary_cross_entropy_with_logits(
            logits.view(-1), y.view(-1), 
            pos_weight=pos_weight
        )
        
        # Update account predictions for next batch/epoch
        with torch.no_grad():
            probs_detached = torch.sigmoid(logits.view(-1)).detach()
            self.update_account_predictions(account_ids, probs_detached)
                  
        # Log loss and metrics
        self.log("train_loss", loss, prog_bar=True, on_epoch=True, sync_dist=True)

        return loss

    def validation_step(self, batch, batch_idx):
        # Updated for dictionary return from __getitem__
        x = batch['padded_features']
        y = batch['label']
        account_ids = batch['account_id_enc']
        external_account_ids_enc = batch['external_account_ids_enc']
        
        # Forward pass
        logits = self(x, external_account_ids_enc)
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
        
        # Update account predictions for next validation batch
        with torch.no_grad():
            probs_cpu = probs.detach()
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
            "val_fraud_f1": fraud_f1,
            "val_fn": fraud_fn,
            "val_fp": fraud_fp,
        }, prog_bar=True, sync_dist=True)
        
        return {
            "val_loss": val_loss,
            "val_fraud_f1": fraud_f1,
            "val_fn": fraud_fn,
            "val_fp": fraud_fp,
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
        account_ids = batch['account_id_enc']
        external_account_ids_enc = batch['external_account_ids_enc']
        
        # Forward pass
        logits = self(x, external_account_ids_enc)
        
        # Get probabilities
        probs = torch.sigmoid(logits)
        
        # Get binary predictions
        preds = (probs > 0.5).float()
        
        # Update account predictions
        with torch.no_grad():
            probs_cpu = probs.detach()
            self.update_account_predictions(account_ids, probs_cpu)
        
        return {"probs": probs, "preds": preds}

    @property
    def account_predictions(self):
        return getattr(self, self._cur_pred_key)

    def reset_account_pred_state(self, num_accounts=50_000):
        # essentially: create a store for account predictions and switch depending on the phase
        self.register_buffer("account_pred_train", torch.full((num_accounts,), self.default_prediction, dtype=torch.float16))
        self.register_buffer("account_pred_val", torch.full((num_accounts,), self.default_prediction, dtype=torch.float16))
        self.register_buffer("account_pred_test", torch.full((num_accounts,), self.default_prediction, dtype=torch.float16))

    def on_train_epoch_start(self):
        self.overwrite_rand_account_predictions = True
        self._cur_pred_key = "account_pred_train"

    def on_validation_epoch_start(self):
        self.overwrite_rand_account_predictions = False
        self._cur_pred_key = "account_pred_val"

    def on_test_epoch_start(self):
        self.overwrite_rand_account_predictions = False
        self._cur_pred_key = "account_pred_test"


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
