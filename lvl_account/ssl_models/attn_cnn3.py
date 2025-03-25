import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl


class LearnableDownsample(nn.Module):
    """Learnable downsampling layer to replace MaxPool"""
    def __init__(self, in_channels, reduction_factor=2, dropout=0.1):
        super().__init__()
        self.reduction_factor = reduction_factor
        
        # Strided convolution for downsampling (more parameters than MaxPool)
        self.down_conv = nn.Conv1d(
            in_channels, 
            in_channels,
            kernel_size=3, 
            stride=reduction_factor,
            padding=1,
            bias=False
        )
        
        # Channel attention to learn which features to preserve
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(in_channels, in_channels // 4, kernel_size=1),
            nn.SiLU(inplace=True),
            nn.Conv1d(in_channels // 4, in_channels, kernel_size=1),
            nn.Sigmoid()
        )
        
        self.norm = nn.BatchNorm1d(in_channels)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # Apply strided convolution
        x = self.down_conv(x)
        
        # Apply channel-wise attention
        attn = self.channel_attention(x)
        x = x * attn
        
        x = self.norm(x)
        x = self.dropout(x)
        return x


class LearnableUpsample(nn.Module):
    """Learnable upsampling layer for reconstruction"""
    def __init__(self, in_channels, scale_factor=2, dropout=0.1):
        super().__init__()
        self.scale_factor = scale_factor
        
        # Transposed convolution for upsampling
        self.up_conv = nn.ConvTranspose1d(
            in_channels,
            in_channels,
            kernel_size=4,
            stride=scale_factor,
            padding=1,
            bias=False
        )
        
        # Refine features after upsampling
        self.refine = nn.Conv1d(
            in_channels,
            in_channels,
            kernel_size=3,
            padding=1,
            bias=False
        )
        
        self.norm = nn.BatchNorm1d(in_channels)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        x = self.up_conv(x)
        x = F.silu(x)
        x = self.refine(x)
        x = self.norm(x)
        x = self.dropout(x)
        return x


class AutoEncoder(pl.LightningModule):
    """
    Autoencoder that compresses 2048x500 to 256x50 with exact reconstruction.
    Uses learnable alternatives to MaxPool.
    """
    def __init__(
        self,
        input_dim=2048,
        seq_len=500,
        latent_dim=256,
        latent_seq_len=50,
        learning_rate=1e-4,
        weight_decay=0.01,
        dropout=0.1
    ):
        super().__init__()
        self.save_hyperparameters()
        
        # Calculate reduction factors
        self.feature_reduction = input_dim // latent_dim  # 8x
        self.seq_reduction = seq_len // latent_seq_len    # 10x
        
        # Feature compression stages
        # 2048 -> 1024 -> 512 -> 256
        self.encoder_stages = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(input_dim, input_dim // 2, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm1d(input_dim // 2),
                nn.SiLU(inplace=True),
                nn.Dropout(dropout),
                LearnableDownsample(input_dim // 2, reduction_factor=2, dropout=dropout),
            ),
            nn.Sequential(
                nn.Conv1d(input_dim // 2, input_dim // 4, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm1d(input_dim // 4),
                nn.SiLU(inplace=True),
                nn.Dropout(dropout),
                LearnableDownsample(input_dim // 4, reduction_factor=2, dropout=dropout),
            ),
            nn.Sequential(
                nn.Conv1d(input_dim // 4, latent_dim, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm1d(latent_dim),
                nn.SiLU(inplace=True),
                nn.Dropout(dropout),
            )
        ])
        
        # Sequence length compression
        # Reduce sequence length from 500 to 50
        # Need to reduce by 10x, so use 2x reduction 3 times and 1.25x reduction once
        self.seq_downsamplers = nn.ModuleList([
            LearnableDownsample(latent_dim, reduction_factor=2, dropout=dropout),
            LearnableDownsample(latent_dim, reduction_factor=2, dropout=dropout),
            LearnableDownsample(latent_dim, reduction_factor=2, dropout=dropout),
            nn.Sequential(
                # Downsample by 1.25x factor (500/8 = 62.5 -> 50)
                nn.Conv1d(latent_dim, latent_dim, kernel_size=5, stride=5//4, padding=1, bias=False),
                nn.BatchNorm1d(latent_dim),
                nn.SiLU(inplace=True),
            ),
        ])
        
        # Feature decompression stages
        # 256 -> 512 -> 1024 -> 2048
        self.decoder_stages = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(latent_dim, latent_dim, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm1d(latent_dim),
                nn.SiLU(inplace=True),
                nn.Dropout(dropout),
            ),
            nn.Sequential(
                nn.Conv1d(latent_dim, input_dim // 4, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm1d(input_dim // 4),
                nn.SiLU(inplace=True),
                nn.Dropout(dropout),
                LearnableUpsample(input_dim // 4, scale_factor=2, dropout=dropout),
            ),
            nn.Sequential(
                nn.Conv1d(input_dim // 4, input_dim // 2, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm1d(input_dim // 2),
                nn.SiLU(inplace=True),
                nn.Dropout(dropout),
                LearnableUpsample(input_dim // 2, scale_factor=2, dropout=dropout),
            ),
            nn.Sequential(
                nn.Conv1d(input_dim // 2, input_dim, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm1d(input_dim),
                nn.SiLU(inplace=True),
            )
        ])
        
        # Sequence length decompression
        # Expand sequence length from 50 to 500
        self.seq_upsamplers = nn.ModuleList([
            nn.Sequential(
                # Upsample by 1.25x factor (50 -> 62.5)
                nn.ConvTranspose1d(latent_dim, latent_dim, kernel_size=6, stride=5//4, padding=1, bias=False),
                nn.BatchNorm1d(latent_dim),
                nn.SiLU(inplace=True),
            ),
            LearnableUpsample(latent_dim, scale_factor=2, dropout=dropout),
            LearnableUpsample(latent_dim, scale_factor=2, dropout=dropout),
            LearnableUpsample(latent_dim, scale_factor=2, dropout=dropout),
        ])
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights using Kaiming initialization for better training dynamics"""
        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.ConvTranspose1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def encode(self, x):
        """Encode input to latent representation"""
        # Input shape: [batch_size, seq_len, input_dim]
        batch_size, seq_len, input_dim = x.shape
        
        # Permute to [batch_size, input_dim, seq_len] for Conv1d
        x = x.permute(0, 2, 1)
        
        # Feature dimension reduction
        for stage in self.encoder_stages:
            x = stage(x)
            
        # Sequence length reduction
        for downsampler in self.seq_downsamplers:
            x = downsampler(x)
        
        # Return latent representation
        # Shape: [batch_size, latent_dim, latent_seq_len]
        return x
    
    def decode(self, z):
        """Decode latent representation back to original dimensions"""
        # Input shape: [batch_size, latent_dim, latent_seq_len]
        
        # Sequence length expansion
        for upsampler in self.seq_upsamplers:
            z = upsampler(z)
            
        # Feature dimension expansion
        for stage in self.decoder_stages:
            z = stage(z)
            
        # Permute back to [batch_size, seq_len, input_dim]
        z = z.permute(0, 2, 1)
        
        return z
    
    def forward(self, x):
        """Forward pass through the autoencoder"""
        z = self.encode(x)
        x_hat = self.decode(z)
        return x_hat, z
    
    def training_step(self, batch, batch_idx):
        """Training step with MSE loss for exact reconstruction"""
        x = batch['padded_features']  # Assuming the same batch format as the CNN model
        
        # Forward pass
        x_hat, z = self(x)
        
        # Calculate reconstruction loss (MSE for exact reconstruction)
        loss = F.mse_loss(x_hat, x)
        
        # Log metrics
        self.log("train_loss", loss, prog_bar=True, on_epoch=True, sync_dist=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        """Validation step"""
        x = batch['padded_features']
        
        # Forward pass
        x_hat, z = self(x)
        
        # Calculate reconstruction loss
        val_loss = F.mse_loss(x_hat, x)
        
        # Calculate additional metrics for monitoring
        # Mean absolute error for a different perspective on reconstruction quality
        mae = F.l1_loss(x_hat, x)
        
        # Log metrics
        self.log_dict({
            "val_loss": val_loss,
            "val_mae": mae
        }, prog_bar=True, sync_dist=True)
        
        return {"val_loss": val_loss, "val_mae": mae}
    
    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler"""
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay
        )
        
        scheduler = {
            "scheduler": torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer,
                T_0=5,  # Restart every 5 epochs
                T_mult=1,  # Keep the same cycle length after restart
                eta_min=self.hparams.learning_rate / 10,  # Minimum learning rate
            ),
            "interval": "epoch",
            "frequency": 1,
            "name": "learning_rate"
        }
        
        return [optimizer], [scheduler]


if __name__ == "__main__":
    # Test model creation
    model = AutoEncoder(input_dim=2048, seq_len=500, latent_dim=256, latent_seq_len=50)
    print(model)
    
    # Test with random input
    batch_size, seq_len, input_dim = 2, 500, 2048
    x = torch.randn(batch_size, seq_len, input_dim)
    x_hat, z = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Latent shape: {z.shape}")
    print(f"Output shape: {x_hat.shape}")