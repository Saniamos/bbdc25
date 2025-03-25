import torch
import pytorch_lightning as pl
from torch import nn
import torch.nn.functional as F

class AttentionBlock(nn.Module):
    """Self-attention block for sequence processing"""
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
        # Self-attention with residual connection and normalization
        attn_output, _ = self.attention(x, x, x)
        x = x + self.dropout(attn_output)
        x = self.norm1(x)
        
        # FFN with residual connection and normalization
        ffn_output = self.ffn(x)
        x = x + self.dropout(ffn_output)
        x = self.norm2(x)
        return x
    
class AutoEncoder(pl.LightningModule):
    def __init__(self, feature_dim,
                 weight_decay=0.01,
                 learning_rate=1e-4,
                 dropout=0.2,
                 embed_shape=256,
                 warmup_steps=100,
                 max_lr=5e-4,
                 min_lr=1e-5):
        super().__init__()
        self.save_hyperparameters()
        
        self.input_dim = feature_dim
        self.dropout = dropout
        self.warmup_steps = warmup_steps
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.hidden_dim = embed_shape
        
        self.encoder = nn.Sequential(
            nn.Conv1d(self.input_dim, self.hidden_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(self.hidden_dim),
            nn.SiLU(inplace=True),
            # replaced MaxPool1d with learnable downsampling block
            nn.Conv1d(self.hidden_dim, self.hidden_dim, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(self.hidden_dim),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv1d(self.hidden_dim, self.hidden_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(self.hidden_dim),
            nn.SiLU(inplace=True),
            # replaced MaxPool1d with learnable downsampling block
            nn.Conv1d(self.hidden_dim, self.hidden_dim, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(self.hidden_dim),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv1d(self.hidden_dim, self.hidden_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(self.hidden_dim),
            nn.SiLU(inplace=True)
        )
        
        # Two attention blocks
        self.attention = nn.Sequential(
            AttentionBlock(self.hidden_dim, num_heads=4, dropout=dropout),
            AttentionBlock(self.hidden_dim, num_heads=4, dropout=dropout)
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(self.hidden_dim, self.hidden_dim, kernel_size=2, stride=2),
            nn.BatchNorm1d(self.hidden_dim),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout),
            nn.ConvTranspose1d(self.hidden_dim, self.hidden_dim, kernel_size=2, stride=2),
            nn.BatchNorm1d(self.hidden_dim),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv1d(self.hidden_dim, self.input_dim, kernel_size=1)
        )
        
        self.criterion = nn.MSELoss()
        self._init_weights()
    
    
    def forward(self, x):
        # x: (B, 2048, feature_dim)
        # Cast x to model's dtype for mixed precision compatibility
        x = x.permute(0, 2, 1)                   # (B, feature_dim, 2048)
        enc = self.encoder(x)                    # (B, hidden_dim, 256)
        enc = enc.permute(0, 2, 1)                # (B, 256, hidden_dim)
        attn = self.attention(enc)                # (B, 256, hidden_dim)
        attn = attn.permute(0, 2, 1)              # (B, hidden_dim, 256)
        dec = self.decoder(attn)             # (B, input_dim, 2048)
        out = dec.permute(0, 2, 1)                # (B, 2048, input_dim)
        return out
    
    def training_step(self, batch, batch_idx):
        x = batch['padded_features']
        x = x.to(self.dtype)  # ensure target dtype matches the model's dtype
        x_hat = self.forward(x)
        loss = self.criterion(x_hat, x)
        self.log("train_loss", loss, prog_bar=True, on_epoch=True, sync_dist=True)
        return loss
    
    def predict_step(self, batch, batch_idx):
        x = batch['padded_features']
        x = x.to(self.dtype)  # ensure target dtype matches the model's dtype
        return self.forward(x)
    
    def configure_optimizers(self):
        try:
            from lion_pytorch import Lion
            optimizer = Lion(
                self.parameters(),
                lr=self.hparams.learning_rate,
                weight_decay=self.hparams.weight_decay
            )
            print("Using Lion optimizer")
        except ImportError:
            optimizer = torch.optim.AdamW(
                self.parameters(),
                lr=self.hparams.learning_rate,
                weight_decay=self.hparams.weight_decay
            )
            print("Using AdamW optimizer")
        
        scheduler = {
            "scheduler": torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer,
                T_0=5,
                T_mult=1,
                eta_min=self.min_lr,
            ),
            "interval": "epoch",
            "frequency": 1,
            "name": "learning_rate"
        }
        return [optimizer], [scheduler]
    
    def _init_weights(self):
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
