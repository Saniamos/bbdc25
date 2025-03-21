import torch
import pytorch_lightning as pl
from torch import nn
import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, f1_score
import warnings
from ssl.bert import TransactionBERTModel

# Try to import Flash Attention - if available
try:
    from flash_attn import flash_attn_func, flash_attn_qkvpacked_func
    from flash_attn.modules.mha import FlashSelfAttention
    HAS_FLASH_ATTENTION = True
    print("Flash Attention is available and will be used for faster attention computation!")
except ImportError:
    HAS_FLASH_ATTENTION = False
    warnings.warn("Flash Attention not available. Install with: pip install flash-attn")

# Try to import xFormers for memory efficient attention
try:
    import xformers
    import xformers.ops
    HAS_XFORMERS = True
    print("xFormers is available and will be used for memory-efficient attention!")
except ImportError:
    HAS_XFORMERS = False
    warnings.warn("xFormers not available. For memory-efficient attention, install with: pip install xformers")


def f1_loss(y_true, y_pred):
    """
    Calculate differentiable F1 loss for binary classification in PyTorch.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted logits
    
    Returns:
        1 - mean(F1 score) as a loss value
    """
    # Ensure inputs are float tensors
    y_true = y_true.float()
    y_pred = y_pred.float()
    
    # Differentiable versions of TP, FP, FN using soft predictions
    # instead of hard binary decisions
    tp = torch.sum(y_true * y_pred, dim=0)
    fp = torch.sum((1 - y_true) * y_pred, dim=0)
    fn = torch.sum(y_true * (1 - y_pred), dim=0)
    
    # Calculate precision and recall with smooth operations
    epsilon = 1e-7  # Small epsilon to avoid division by zero
    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)
    
    # Calculate F1 score with smooth operations
    f1 = 2 * precision * recall / (precision + recall + epsilon)
    
    # Return 1 - mean(F1) as the loss
    return 1 - torch.mean(f1)

class EfficientMultiheadAttention(nn.Module):
    """Memory-efficient multi-head attention using either Flash Attention or xFormers if available."""
    
    def __init__(self, embed_dim, num_heads, dropout=0.0, batch_first=True):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.batch_first = batch_first
        
        # QKV projections
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        
        # Choose the most efficient attention implementation available
        if HAS_FLASH_ATTENTION:
            self.attn_impl = "flash"
            self.flash_attn = FlashSelfAttention(softmax_scale=None, attention_dropout=dropout)
        elif HAS_XFORMERS:
            self.attn_impl = "xformers"
        else:
            self.attn_impl = "vanilla"
            self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=batch_first)
    
    def forward(self, x, key_padding_mask=None):
        if not self.batch_first:
            x = x.transpose(0, 1)  # Convert to batch_first format
        
        batch_size, seq_len, _ = x.shape
        
        if self.attn_impl == "flash":
            # Flash Attention implementation
            q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
            k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
            v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
            
            # Apply key padding mask if provided
            if key_padding_mask is not None:
                # Flash attention expects mask where 0 = keep, 1 = mask
                # Our key_padding_mask is the opposite: True = mask
                mask = ~key_padding_mask
                # Convert to float and unsqueeze for broadcasting
                mask = mask.unsqueeze(1).unsqueeze(2).float()
                # Apply large negative value to masked positions
                k = k * mask - 1e9 * (1 - mask)
            
            # Perform flash attention
            context = self.flash_attn(q, k, v)
            context = context.reshape(batch_size, seq_len, self.embed_dim)
            
        elif self.attn_impl == "xformers":
            # xFormers memory-efficient attention
            q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
            k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
            v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
            
            # Create attention mask from key_padding_mask
            attn_mask = None
            if key_padding_mask is not None:
                attn_mask = key_padding_mask.unsqueeze(1).unsqueeze(2)
            
            # Use xFormers memory-efficient attention
            context = xformers.ops.memory_efficient_attention(
                q, k, v, 
                attn_mask=attn_mask,
                p=self.dropout if self.training else 0.0
            )
            context = context.transpose(1, 2).reshape(batch_size, seq_len, self.embed_dim)
            
        else:
            # Fallback to standard PyTorch implementation
            context, _ = self.attn(x, x, x, key_padding_mask=key_padding_mask)
        
        return self.out_proj(context)

class Classifier(pl.LightningModule):
    """
    BERT-based model for fraud detection in transaction sequences,
    leveraging pre-trained weights from SSL task.
    """
    def __init__(
        self,
        feature_dim,
        pretrained_model_path=None,
        weight_decay=0.01,
        learning_rate=1e-4,
        dropout=0.2,
        freeze_bert=False,
        classifier_hidden_dim=128,
        num_attention_heads=4,  # Parameter for multi-head attention
        use_multi_pooling=True, # Parameter to enable multiple pooling strategies
        use_efficient_attention=True  # New parameter to use optimized attention
    ):
        super().__init__()
        self.save_hyperparameters()
        
        # Initialize the transaction BERT model
        self.bert_model = TransactionBERTModel(feature_dim=feature_dim)
        
        # Load pre-trained weights if provided
        if pretrained_model_path:
            print(f"Loading pre-trained weights from {pretrained_model_path}")
            state_dict = torch.load(pretrained_model_path, map_location='cpu')            
            # Load weights
            self.bert_model.load_state_dict(state_dict, strict=False)
            print("Pre-trained weights loaded successfully")
        
        # Freeze BERT model if specified
        if freeze_bert:
            for param in self.bert_model.parameters():
                param.requires_grad = False
            print("BERT model frozen")
        
        # Calculate appropriate attention dimension that's divisible by num_attention_heads
        self.attention_dim = feature_dim
        if feature_dim % num_attention_heads != 0:
            # Find the nearest multiple of num_attention_heads
            self.attention_dim = (feature_dim // num_attention_heads) * num_attention_heads
            print(f"Feature dimension {feature_dim} is not divisible by {num_attention_heads} attention heads.")
            print(f"Using projection to attention dimension: {self.attention_dim}")
            # Add projection layer to convert feature_dim to attention_dim
            self.dim_projection = nn.Linear(feature_dim, self.attention_dim)
        else:
            self.dim_projection = None
        
        # Use the efficient attention implementation if requested
        if use_efficient_attention:
            self.multi_head_attn = EfficientMultiheadAttention(
                embed_dim=self.attention_dim,
                num_heads=num_attention_heads,
                dropout=dropout,
                batch_first=True
            )
        else:
            # Fallback to standard PyTorch implementation
            self.multi_head_attn = nn.MultiheadAttention(
                embed_dim=self.attention_dim,
                num_heads=num_attention_heads,
                dropout=dropout,
                batch_first=True
            )
        
        # Attention pooling layer to aggregate sequence information - now with multi-head support
        self.attention_pool = nn.Sequential(
            nn.Linear(feature_dim, num_attention_heads),
            nn.Tanh(),  # Tanh activation can help with attention stability
            nn.Linear(num_attention_heads, 1),
            nn.Softmax(dim=1)
        )
        
        # Enhanced classifier with skip connections and more non-linearity
        classifier_input_dim = feature_dim * (3 if use_multi_pooling else 1)
        
        self.classifier = nn.Sequential(
            nn.Linear(classifier_input_dim, classifier_hidden_dim),
            nn.LayerNorm(classifier_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(classifier_hidden_dim, classifier_hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout/2),  # Less dropout in later layers
            nn.Linear(classifier_hidden_dim // 2, 1),
        )
        
        self.use_multi_pooling = use_multi_pooling

    def forward(self, x, mask=None):
        """
        Args:
            x: Transaction sequences [batch_size, seq_len, feature_dim]
            mask: Optional mask for valid positions [batch_size, seq_len]
            
        Returns:
            Fraud probability
        """
        batch_size, seq_len, _ = x.shape
        
        # We don't need masked features for classification, so create dummy tensor
        dummy_masked_pos = torch.zeros_like(x, dtype=torch.bool)
        
        # Get BERT sequence representations
        bert_output = self.bert_model(x, dummy_masked_pos)
        
        # Project dimensions if needed for attention compatibility
        if self.dim_projection is not None:
            attn_input = self.dim_projection(bert_output)
        else:
            attn_input = bert_output
        
        # Apply multi-head self-attention for better sequence understanding
        if mask is not None:
            # Prepare attention mask (True indicates positions to be masked)
            attn_mask = ~mask.bool() if hasattr(self.multi_head_attn, 'attn_impl') else mask.bool()
            
            # Use different calling convention depending on attention implementation
            if isinstance(self.multi_head_attn, EfficientMultiheadAttention):
                attn_output = self.multi_head_attn(attn_input, key_padding_mask=attn_mask)
            else:
                # Standard PyTorch MultiheadAttention
                attn_output, _ = self.multi_head_attn(
                    attn_input, 
                    attn_input, 
                    attn_input,
                    key_padding_mask=attn_mask
                )
            
            # Residual connection - need to project back if dimensions differ
            if self.dim_projection is not None:
                # Project attention output back to original dimension if needed
                attn_output = self.dim_projection(attn_output)
                
            bert_output = bert_output + attn_output
        else:
            # No masking needed
            if isinstance(self.multi_head_attn, EfficientMultiheadAttention):
                attn_output = self.multi_head_attn(attn_input)
            else:
                attn_output, _ = self.multi_head_attn(attn_input, attn_input, attn_input)
                
            # Residual connection - need to project back if dimensions differ
            if self.dim_projection is not None:
                # Project attention output back to original dimension if needed
                attn_output = nn.functional.linear(attn_output, self.dim_projection.weight.t())
                
            bert_output = bert_output + attn_output
        
        # Multi-pooling strategy (combine attention, max, and average pooling)
        if self.use_multi_pooling:
            # 1. Attention pooling
            attention_weights = self.attention_pool(bert_output)
            if mask is not None:
                mask_expanded = mask.unsqueeze(-1)
                attention_weights = attention_weights * mask_expanded
                attention_weights = attention_weights / (attention_weights.sum(dim=1, keepdim=True) + 1e-8)
            attn_pooled = torch.bmm(attention_weights.transpose(1, 2), bert_output).squeeze(1)
            
            # 2. Max pooling
            if mask is not None:
                mask_expanded = mask.unsqueeze(-1)
                # Set masked positions to large negative value before max
                masked_output = bert_output * mask_expanded - 1e9 * (1 - mask_expanded)
                max_pooled = torch.max(masked_output, dim=1)[0]
            else:
                max_pooled = torch.max(bert_output, dim=1)[0]
                
            # 3. Average pooling
            if mask is not None:
                # Compute proper average considering only valid positions
                mask_expanded = mask.unsqueeze(-1)
                sum_pooled = torch.sum(bert_output * mask_expanded, dim=1)
                # Avoid division by zero by adding epsilon
                avg_pooled = sum_pooled / (torch.sum(mask_expanded, dim=1) + 1e-8)
            else:
                avg_pooled = torch.mean(bert_output, dim=1)
            
            # Concatenate all pooling results
            pooled_output = torch.cat([attn_pooled, max_pooled, avg_pooled], dim=1)
        else:
            # Use only attention pooling as before
            attention_weights = self.attention_pool(bert_output)
            if mask is not None:
                mask_expanded = mask.unsqueeze(-1)
                attention_weights = attention_weights * mask_expanded
                attention_weights = attention_weights / (attention_weights.sum(dim=1, keepdim=True) + 1e-8)
            pooled_output = torch.bmm(attention_weights.transpose(1, 2), bert_output).squeeze(1)
            
        # Apply classifier
        logits = self.classifier(pooled_output)
        return logits
    
    def training_step(self, batch, batch_idx):
        x, _, _, y = batch  # Unpack (masked_seqs, masked_pos, orig_seqs, label)
        
        # Create sequence mask (identifies non-padding elements)
        seq_mask = (x.sum(dim=-1) != 0).float()  # [batch_size, seq_len]
        
        # Forward pass
        logits = self(x, seq_mask)
        y = y.float()  # Ensure labels are float for loss functions
    
        # Calculate F1 loss - use sigmoid to get probabilities before calculating F1 loss
        loss = nn.BCEWithLogitsLoss()(logits.view(-1), y.view(-1))
        # loss = f1_loss(y.view(-1), logits.view(-1))
        
        # Log metrics - emphasizing F1-related metrics
        self.log("train_loss", loss, prog_bar=True, on_epoch=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, _, _, y = batch
        
        # Create sequence mask
        seq_mask = (x.sum(dim=-1) != 0).float()  # [batch_size, seq_len]
        
        # Forward pass
        logits = self(x, seq_mask)
        y = y.float()
        
        # Calculate F1 loss - use probabilities directly
        val_loss = nn.BCEWithLogitsLoss()(logits.view(-1), y.view(-1))
        # val_loss = f1_loss(y.view(-1), logits.view(-1))
        
        # Log metrics - emphasizing F1-related metrics
        self.log("val_loss", val_loss, prog_bar=True)
        
        return {"val_loss": val_loss}
    
    def configure_optimizers(self):
        # Only optimize classifier parameters if BERT is frozen
        if self.hparams.freeze_bert:
            params = self.classifier.parameters()
        else:
            params = self.parameters()
            
        optimizer = torch.optim.AdamW(
            params,
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay
        )
        
        return optimizer

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        """
        Prediction step for generating fraud probabilities and predictions.
        
        Args:
            batch: The input batch from the dataloader
            batch_idx: The batch index
            dataloader_idx: The dataloader index (if using multiple)
            
        Returns:
            Dictionary containing probabilities and binary predictions
        """
        x, _, _, _ = batch  # Unpack (masked_seqs, masked_pos, orig_seqs, label)
        
        # Create sequence mask (identifies non-padding elements)
        seq_mask = (x.sum(dim=-1) != 0).float()  # [batch_size, seq_len]
        
        # Forward pass
        logits = self(x, seq_mask)
        
        # Get probabilities
        probs = torch.sigmoid(logits)
        
        # Get binary predictions
        preds = (probs > 0.5).float()
        
        return {"probs": probs, "preds": preds}


if __name__ == "__main__":
    # Test model creation
    model = FraudBERTClassifier(
        feature_dim=68,
        pretrained_model_path="./saved_models/transaction_bert/final-model.pt"
    )
    print(model)
