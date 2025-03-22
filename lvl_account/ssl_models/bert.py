import torch
import pytorch_lightning as pl
from transformers import BertModel, get_linear_schedule_with_warmup, BertConfig

class NonLinearProjection(torch.nn.Module):
    """
    CNN-based projection module that uses 1D convolutions to transform features.
    Maps transaction features to BERT's expected hidden dimension while capturing
    local patterns across the sequence.
    """
    def __init__(self, in_dim, out_dim, dropout=0.1, kernel_size=3):
        super().__init__()
        hidden_dim = (in_dim + out_dim) // 2  # Intermediary dimension
        padding = (kernel_size - 1) // 2  # Same padding to maintain sequence length
        
        # Using CNNs instead of linear layers
        self.conv1 = torch.nn.Conv1d(in_channels=in_dim, out_channels=hidden_dim, 
                                    kernel_size=kernel_size, padding=padding)
        self.activation = torch.nn.GELU()
        self.conv2 = torch.nn.Conv1d(in_channels=hidden_dim, out_channels=out_dim, 
                                    kernel_size=kernel_size, padding=padding)
        self.dropout = torch.nn.Dropout(dropout)
        self.norm = torch.nn.LayerNorm(out_dim)
        
    def forward(self, x):
        # Input shape: [batch_size, seq_len, in_dim]
        
        # Transpose for convolution: [batch_size, in_dim, seq_len]
        x = x.transpose(1, 2)
        
        # Apply convolutions
        x = self.conv1(x)  # [batch_size, hidden_dim, seq_len]
        x = self.activation(x)
        x = self.dropout(x)
        x = self.conv2(x)  # [batch_size, out_dim, seq_len]
        
        # Transpose back: [batch_size, seq_len, out_dim]
        x = x.transpose(1, 2)
        
        # Apply layer normalization
        x = self.norm(x)
        
        return x

class TransactionBERTModel(pl.LightningModule):
    """
    BERT model for transaction data pretraining, using Hugging Face BERT as the core model.
    """
    def __init__(self,
                 feature_dim,
                 weight_decay=0.01,
                 learning_rate=1e-4,
                 warmup_steps=1000,
                 dropout=0.1,
                 fine_tune_ffn=False,
                 model_size = "base",
                 use_gradient_checkpointing=True):
        super().__init__()
        self.save_hyperparameters()
    
        # Use smaller model variants - using BertModel instead of BertForMaskedLM
        # This removes the unused MLM prediction head
        if model_size == "tiny":
            self.bert = BertModel.from_pretrained("prajjwal1/bert-tiny")  # 2 layers, 128 hidden size
        elif model_size == "small":
            self.bert = BertModel.from_pretrained("prajjwal1/bert-small") # 4 layers, 512 hidden size
        else:
            # Use base model but reduce number of layers
            config = BertConfig.from_pretrained("bert-base-uncased")
            config.num_hidden_layers = 8  # Reduce layers for faster training
            self.bert = BertModel.from_pretrained("bert-base-uncased", config=config)

        if hasattr(torch, 'nn') and hasattr(torch.nn, 'functional') and hasattr(torch.nn.functional, 'scaled_dot_product_attention'):
            # Use Flash Attention when available
            print("Flash Attention is available. Using for faster training.")
            self.bert.config.use_flash_attention = True
        
        # Enable gradient checkpointing to save memory
        if use_gradient_checkpointing:
            self.bert.gradient_checkpointing_enable()
        
        # Get BERT's hidden dimension
        self.bert_hidden_dim = self.bert.config.hidden_size  # Typically 768 for bert-base-uncased
        
        # Freeze specific parameters
        for name, param in self.bert.named_parameters():
            # By default, freeze all parameters
            param.requires_grad = False
            
            # Unfreeze LayerNorm parameters
            if 'LayerNorm' in name:
                param.requires_grad = True
            # Unfreeze positional embeddings
            elif 'position_embeddings' in name:
                param.requires_grad = True
            # Conditionally unfreeze MLP/FFN layers
            elif ('intermediate' in name or 'output' in name) and fine_tune_ffn:
                param.requires_grad = True
        
        # Create non-linear input projection (features -> BERT hidden dim)
        self.input_projection = NonLinearProjection(
            in_dim=feature_dim,
            out_dim=self.bert_hidden_dim,
            dropout=dropout
        )
        
        # Create non-linear output projection (BERT hidden dim -> features)
        self.output_projection = NonLinearProjection(
            in_dim=self.bert_hidden_dim,
            out_dim=feature_dim,
            dropout=dropout
        )
        
    def forward(self, masked_seqs, masked_pos):
        """
        Forward pass for the model.
        
        Args:
            masked_seqs: Transaction sequences with masked values [batch_size, seq_len, feature_dim]
            masked_pos: Boolean tensor indicating masked positions [batch_size, seq_len, feature_dim]
            
        Returns:
            Reconstructed features at masked positions
        """
        batch_size, seq_len, _ = masked_seqs.shape
        
        # Project features to BERT's hidden dimension
        embeddings = self.input_projection(masked_seqs)
        
        # Create attention mask (1 for real tokens, 0 for padding)
        attention_mask = torch.ones(batch_size, seq_len, device=masked_seqs.device)
        
        # Get the max position embedding size from BERT's config
        max_position_length = self.bert.config.max_position_embeddings
        
        # Create position_ids to control positional embeddings
        position_ids = torch.arange(seq_len, dtype=torch.long, device=masked_seqs.device) % max_position_length
        position_ids = position_ids.expand((batch_size, -1))
        
        # Changed from BertForMaskedLM to BertModel - direct encoder access 
        outputs = self.bert(
            inputs_embeds=embeddings,
            attention_mask=attention_mask,
            position_ids=position_ids,
            token_type_ids=torch.zeros(embeddings.size()[:-1], dtype=torch.long, device=masked_seqs.device),
            return_dict=True
        )
        
        # BertModel returns last_hidden_state directly
        sequence_output = outputs.last_hidden_state
        
        # Project back to feature dimension
        reconstructed_seqs = self.output_projection(sequence_output)
        
        return reconstructed_seqs
    
    def training_step(self, batch, batch_idx):
        masked_seqs = batch['masked_features']
        masked_pos = batch['masked_pos']
        orig_seqs = batch['padded_features']
        
        # Get reconstructed sequences
        recon_seqs = self(masked_seqs, masked_pos)
        
        # Calculate loss only for masked positions
        loss_mask = masked_pos.float()
        loss = torch.nn.functional.mse_loss(recon_seqs * loss_mask, orig_seqs * loss_mask, reduction='none')
        loss = (loss.sum(dim=-1) / loss_mask.sum(dim=-1).clamp(min=1.0)).mean()
        
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        masked_seqs = batch['masked_features']
        masked_pos = batch['masked_pos']
        orig_seqs = batch['padded_features']
        
        # Get reconstructed sequences
        recon_seqs = self(masked_seqs, masked_pos)
        
        # Calculate loss only for masked positions
        loss_mask = masked_pos.float()
        loss = torch.nn.functional.mse_loss(recon_seqs * loss_mask, orig_seqs * loss_mask, reduction='none')
        loss = (loss.sum(dim=-1) / loss_mask.sum(dim=-1).clamp(min=1.0)).mean()
        
        self.log("val_loss", loss, prog_bar=True, on_epoch=True)
        return loss
    
    def on_fit_start(self):
        """
        Called at the beginning of training.
        Compile the model here instead of in __init__ to ensure state_dict compatibility
        """
        # Print parameter statistics
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.parameters())
        print(f"Trainable parameters: {trainable_params:,} ({trainable_params/total_params:.2%} of total)")
    
    def configure_optimizers(self):
        # Use AdamW optimizer with weight decay
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.named_parameters()
                         if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [p for n, p in self.named_parameters()
                         if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        
        optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters,
            lr=self.hparams.learning_rate
        )
        
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.hparams.warmup_steps,
            num_training_steps=self.trainer.estimated_stepping_batches
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }


if __name__ == "__main__":
    TransactionBERTModel(feature_dim=68)