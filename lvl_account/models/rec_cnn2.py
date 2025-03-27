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
        pretrained_model_path="/home/yale/Repositories/bbdc25/lvl_account/saved_models/simple_cnn/logs/simple_cnn/rec_cnn_base/simple_cnn-epoch=14-val_fraud_f1=0.9264.ckpt",
        weight_decay=0.01,
        learning_rate=1e-4,
        dropout=0.2,
        freeze_pretrained_model=True,
        hidden_dim=256,
        warmup_steps=100,
        max_lr=5e-4,
        min_lr=1e-5,
        pred_state_dim=256
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
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout/2),
            nn.Linear(64, 1)
        )
    
    def add_prediction_feature(self, x, external_account_ids_enc):
        """
        Add a feature channel to input data using per-account hidden state.
        
        Args:
            x: Transaction sequences [batch_size, seq_len, feature_dim]
            external_account_ids_enc: Tensor of account IDs.
                Expected shape: [batch_size, 1]. If an extra dimension exists (e.g. [batch_size, seq_len, 1]),
                the first column is used.
            
        Returns:
            x concatenated with fraud feature [batch_size, seq_len, feature_dim + 1]
        """
        if external_account_ids_enc.dim() > 2:
            external_account_ids_enc = external_account_ids_enc[:, 0]
        hidden = self.account_hidden_state[external_account_ids_enc.squeeze()]  # [batch_size, pred_state_dim]
        fraud_feature = self.hidden_to_score(hidden)  # [batch_size, 1]
        batch_size, seq_len, _ = x.shape
        fraud_feature = fraud_feature.unsqueeze(1).expand(batch_size, seq_len, 1)
        return torch.cat([x, fraud_feature], dim=2)
    
    def forward(self, x, external_account_ids_enc=None):
        batch_size, seq_len, _ = x.shape
        if (external_account_ids_enc is not None):
            ids = external_account_ids_enc.squeeze()
            hidden_old = self.account_hidden_state[ids]  # [batch_size, pred_state_dim]
            fraud_feature_old = self.hidden_to_score(hidden_old)  # [batch_size, 1]
            fraud_feature_old = fraud_feature_old.unsqueeze(1).expand(batch_size, seq_len, 1)
            x_aug = torch.cat([x, fraud_feature_old], dim=2)
            x_aug = x_aug.permute(0, 2, 1)
            h = self.cnn_layers(x_aug)
            h = h.view(batch_size, -1)
            logits_initial = self.classifier(h)
            probs = torch.sigmoid(logits_initial.view(-1))
            updated_state = self.rnn_cell(probs.unsqueeze(1), hidden_old)
            updated_state = updated_state.to(self.account_hidden_state.dtype)  # ensure dtype match
            self.account_hidden_state[ids] = updated_state
            fraud_feature_updated = self.hidden_to_score(updated_state)
            fraud_feature_updated = fraud_feature_updated.unsqueeze(1).expand(batch_size, seq_len, 1)
            x_new = torch.cat([x, fraud_feature_updated], dim=2)
            x_new = x_new.permute(0, 2, 1)
            h_new = self.cnn_layers(x_new)
            h_new = h_new.view(batch_size, -1)
            logits_final = self.classifier(h_new)
            return logits_final
        else:
            zeros = torch.zeros(batch_size, seq_len, 1, device=x.device)
            x = torch.cat([x, zeros], dim=2)
            x = x.permute(0, 2, 1)
            x = self.cnn_layers(x)
            x = x.view(batch_size, -1)
            logits = self.classifier(x)
            return logits
    
    def update_account_hidden_state(self, account_ids: torch.Tensor, predictions):
        """
        Update the per-account hidden state with new fraud observations,
        allowing gradients to flow through the GRU cell update.
        
        Args:
            account_ids: Tensor of account IDs [batch_size, 1] or [batch_size]
            predictions: Fraud probabilities [batch_size], as a float value between 0 and 1
        """
        # Ensure account_ids is a 1D tensor.
        account_ids = account_ids.squeeze().long()
        # Prepare predictions as input to GRU cell: shape [batch_size, 1]
        input_obs = predictions.unsqueeze(1)
        # Get current hidden state for these accounts.
        current_state = self.account_hidden_state[account_ids]
        # Update via GRU cell.
        updated_state = self.rnn_cell(input_obs, current_state)
        updated_state = updated_state.to(self.account_hidden_state.dtype)  # ensure dtype match
        # Update the buffer without detaching so that gradients flow.
        self.account_hidden_state[account_ids] = updated_state
    
    def training_step(self, batch, batch_idx):
        x = batch['padded_features']
        y = batch['label']
        account_ids = batch['account_id_enc']
        external_account_ids_enc = batch['external_account_ids_enc']
        
        logits = self(x, external_account_ids_enc)
        y = y.float()
        pos_weight = torch.tensor([8], device=logits.device)
        loss = F.binary_cross_entropy_with_logits(logits.view(-1), y.view(-1), pos_weight=pos_weight)
        
        # Update hidden state with gradients flowing.
        probs = torch.sigmoid(logits.view(-1))
        self.update_account_hidden_state(account_ids, probs)
        
        self.log("train_loss", loss, prog_bar=True, on_epoch=True, sync_dist=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x = batch['padded_features']
        y = batch['label']
        account_ids = batch['account_id_enc']
        external_account_ids_enc = batch['external_account_ids_enc']
        
        logits = self(x, external_account_ids_enc)
        y = y.float()
        pos_weight = torch.tensor([8], device=logits.device)
        val_loss = F.binary_cross_entropy_with_logits(logits.view(-1), y.view(-1), pos_weight=pos_weight)
        
        probs = torch.sigmoid(logits.view(-1))
        preds = (probs > 0.5).float()
        
        self.update_account_hidden_state(account_ids, probs)
        
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

    def reset_account_pred_state(self, num_accounts=None):
        pass # only needed so that the model is run twice on val and pred for the account_hidden_state to be properly initialized


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
