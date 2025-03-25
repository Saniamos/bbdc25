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
                 embed_shape=(256, 64),  # embed_shape: (reduced_seq_len, latent_channels)
                 warmup_steps=100,
                 max_lr=5e-4,
                 min_lr=1e-5):
        super().__init__()
        self.save_hyperparameters()
        
        self.feature_dim = feature_dim
        self.embed_shape = embed_shape  # e.g. (256, 50): time steps and latent channels
        self.dropout = dropout
        self.warmup_steps = warmup_steps
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.hidden_dim = 256  # base hidden dimension
        
        # Encoder: operate in 1D. Input shape: (B, feature_dim, 2048)
        self.encoder = nn.Sequential(
            # Block 1: from feature_dim to hidden_dim; downsample sequence length: 2048 -> 1024
            nn.Conv1d(feature_dim, self.hidden_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(self.hidden_dim),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv1d(self.hidden_dim, self.hidden_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(self.hidden_dim),
            nn.SiLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Dropout(dropout),
            # Block 2: 1024 -> 512
            nn.Conv1d(self.hidden_dim, self.hidden_dim*2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(self.hidden_dim*2),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv1d(self.hidden_dim*2, self.hidden_dim*2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(self.hidden_dim*2),
            nn.SiLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Dropout(dropout),
            # Block 3: 512 -> 256
            nn.Conv1d(self.hidden_dim*2, self.hidden_dim*4, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(self.hidden_dim*4),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv1d(self.hidden_dim*4, self.hidden_dim*4, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(self.hidden_dim*4),
            nn.SiLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Dropout(dropout)
        )
        # After three downsamples: sequence length 2048/(2*2*2)=256, channels: hidden_dim*4
        
        # Projection from high channels to latent dimension (embed_shape[1])
        self.projection = nn.Conv1d(self.hidden_dim*4, embed_shape[1], kernel_size=1)
        
        # Two attention blocks (operating over the time dimension, which is 256)
        self.attention = nn.Sequential(
            AttentionBlock(embed_shape[1], num_heads=4, dropout=dropout),
            AttentionBlock(embed_shape[1], num_heads=4, dropout=dropout)
        )
        
        # Decoder: mirror the encoder.
        # First, inverse attention and projection.
        self.decoder_projection = nn.Conv1d(embed_shape[1], self.hidden_dim*4, kernel_size=1)
        
        self.decoder = nn.Sequential(
            # Block 1: Upsample: 256 -> 512
            nn.ConvTranspose1d(self.hidden_dim*4, self.hidden_dim*2, kernel_size=2, stride=2),
            nn.BatchNorm1d(self.hidden_dim*2),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout),
            # Block 2: Upsample: 512 -> 1024
            nn.ConvTranspose1d(self.hidden_dim*2, self.hidden_dim, kernel_size=2, stride=2),
            nn.BatchNorm1d(self.hidden_dim),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout),
            # Block 3: Upsample: 1024 -> 2048, output channels = feature_dim
            nn.ConvTranspose1d(self.hidden_dim, feature_dim, kernel_size=2, stride=2)
        )
        
        self.criterion = nn.MSELoss()
        self._init_weights()
    
    def forward(self, x):
        # x: (B, 2048, feature_dim); permute to (B, feature_dim, 2048)
        x = x.permute(0, 2, 1)
        enc = self.encoder(x)                   # shape: (B, hidden_dim*4, 256)
        proj = self.projection(enc)             # shape: (B, embed_channels, 256)
        # Permute for attention: (B, 256, embed_channels)
        proj = proj.permute(0, 2, 1)
        attn = self.attention(proj)             # shape: (B, 256, embed_channels)
        # Permute back for decoder: (B, embed_channels, 256)
        attn = attn.permute(0, 2, 1)
        dec_proj = self.decoder_projection(attn)  # shape: (B, hidden_dim*4, 256)
        dec = self.decoder(dec_proj)              # shape: (B, feature_dim, 2048)
        out = dec.permute(0, 2, 1)                  # (B, 2048, feature_dim)
        return out
    
    def training_step(self, batch, batch_idx):
        x = batch['padded_features']
        x_hat = self.forward(x)
        loss = self.criterion(x_hat, x)
        self.log("train_loss", loss, prog_bar=True, on_epoch=True, sync_dist=True)
        return loss
    
    def predict_step(self, batch, batch_idx):
        x = batch['padded_features']
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
