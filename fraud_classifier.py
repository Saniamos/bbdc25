import torch
import pytorch_lightning as pl
from torch import nn
import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, f1_score

# Add parent directory to path to import from SSL module
import sys
sys.path.append('./models/features/ssl') # we need to do this weird thing, as the scripts inside of the task folder should also be able to run on their own
from bert import TransactionBERTModel


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

class FraudBERTClassifier(pl.LightningModule):
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
        freeze_bert=True,
        classifier_hidden_dim=128
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
        
        # Create classifier head
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, classifier_hidden_dim),
            nn.LayerNorm(classifier_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(classifier_hidden_dim, 1),
        )
        
        # Attention pooling layer to aggregate sequence information
        self.attention_pool = nn.Sequential(
            nn.Linear(feature_dim, 1),
            nn.Softmax(dim=1)
        )
        
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
        
        # Apply attention pooling over the sequence
        attention_weights = self.attention_pool(bert_output)
        
        # Apply mask if provided (to handle variable sequence lengths)
        if mask is not None:
            mask_expanded = mask.unsqueeze(-1)  # [batch_size, seq_len, 1]
            attention_weights = attention_weights * mask_expanded
            attention_weights = attention_weights / (attention_weights.sum(dim=1, keepdim=True) + 1e-8)
            
        # Weighted sum of sequence representations
        pooled_output = torch.bmm(attention_weights.transpose(1, 2), bert_output)  # [batch_size, 1, feature_dim]
        pooled_output = pooled_output.squeeze(1)  # [batch_size, feature_dim]
        
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
        # loss = nn.BCEWithLogitsLoss()(logits.view(-1), y.view(-1))
        loss = f1_loss(y.view(-1), logits.view(-1))
        
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
        # loss = nn.BCEWithLogitsLoss()(logits.view(-1), y.view(-1))
        val_loss = f1_loss(y.view(-1), logits.view(-1))
        
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
