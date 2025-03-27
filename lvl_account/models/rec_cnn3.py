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
        pred_state_dim=256,
        atten_dim = 128,
        encoder_dim = 64
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
        self.gru = nn.GRU(input_size=self.hparams.hidden_dim + encoder_dim, hidden_size=pred_state_dim, batch_first=True)
        
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

        # self.account_enc = nn.Sequential(
        #     nn.Linear(2048, self.hparams.hidden_dim),
        #     nn.SiLU(inplace=True),
        #     nn.Dropout(dropout),
        #     nn.LayerNorm(self.hparams.hidden_dim),
        #     nn.Linear(self.hparams.hidden_dim, self.hparams.hidden_dim),
        #     nn.SiLU(inplace=True),
        #     nn.LayerNorm(self.hparams.hidden_dim),
        # )
        self.account_enc = nn.Sequential(
            nn.Linear(2048, encoder_dim),
            # AttentionBlock(64, num_heads=4, dropout=dropout),            
            # nn.Linear(64, 64)
        )
        # self.classifier = nn.Sequential(
        #     nn.Linear(self.hparams.pred_state_dim, atten_dim),
        #     AttentionBlock(atten_dim, num_heads=4, dropout=dropout),            
        #     nn.Linear(atten_dim, 1)
        # )

        self.classifier = nn.Sequential(
            nn.Linear(self.hparams.pred_state_dim, 64),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout/2),
            nn.Linear(64, 1)
        )
        
        # Persistent GRU state to be carried across batches
        self.hidden_state = None
        self.state_dropout = nn.Dropout(self.hparams.dropout)

    def forward(self, x, external_account_ids_enc):
        batch_size, seq_len, _ = x.shape
        # Process x through CNN.
        x_cnn = x.permute(0, 2, 1)               # (batch_size, feature_dim, seq_len)
        cnn_feat = self.cnn_layers(x_cnn)        # (batch_size, hidden_dim, 1)
        cnn_feat = cnn_feat.view(batch_size, -1) # (batch_size, hidden_dim)
        # Unsqueeze cnn_feat to add a sequence dimension for GRU.
        cnn_feat_seq = cnn_feat.unsqueeze(1)     # (batch_size, 1, hidden_dim)
        external_account_ids_enc = self.account_enc(external_account_ids_enc.to(self.dtype).permute(0, 2, 1))
        combined_input = torch.cat([cnn_feat_seq, external_account_ids_enc], dim=2)  # (batch_size, 1, hidden_dim + atten_dim)

        # Initialize or adjust persistent hidden state if needed.
        if self.hidden_state is None or self.hidden_state.size(1) != batch_size:
            self.hidden_state = torch.zeros(1, batch_size, self.hparams.pred_state_dim, device=x.device)
        # Use persistent hidden state in GRU forward pass.
        _, h = self.gru(combined_input, self.hidden_state)
        new_state = self.state_dropout(h)
        self.hidden_state = new_state.detach()
        # Squeeze GRU state: shape becomes (batch_size, pred_state_dim)
        gru_feature = new_state.squeeze(0)
        
        # Pass the GRU feature through the classifier.
        logits = self.classifier(gru_feature)
        return logits
    
    def training_step(self, batch, batch_idx):
        x = batch['padded_features']
        y = batch['label']
        external_account_ids_enc = batch['external_account_ids_enc']
        logits = self(x, external_account_ids_enc)
        y = y.float()
        pos_weight = torch.tensor([8], device=logits.device)
        loss = F.binary_cross_entropy_with_logits(logits.view(-1), y.view(-1), pos_weight=pos_weight)
        self.log("train_loss", loss, prog_bar=True, on_epoch=True, sync_dist=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x = batch['padded_features']
        y = batch['label']
        external_account_ids_enc = batch['external_account_ids_enc']
        logits = self(x, external_account_ids_enc)
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
