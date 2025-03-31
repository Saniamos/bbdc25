import numpy as np
import torch
import pytorch_lightning as pl
import torch.nn.functional as F
from xgboost import XGBClassifier

class Classifier(pl.LightningModule):
    def __init__(self, feature_dim, learning_rate=1e-4, weight_decay=0.01, **kwargs):
        super().__init__()
        self.automatic_optimization = False  # disable automatic optimization for non-gradient model
        self.feature_dim = feature_dim
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.xgb_model = None
        self.train_features = []
        self.train_labels = []
    
    def forward(self, x):
        # x: [batch, seq_len, feature_dim]
        flattened = x.view(x.size(0), -1).detach().cpu().numpy()  # flatten instead of pooling
        if self.xgb_model is not None:
            prob = self.xgb_model.predict_proba(flattened)[:, 1]
        else:
            prob = np.zeros(flattened.shape[0])
        prob = np.clip(prob, 1e-7, 1 - 1e-7)
        logits = np.log(prob / (1 - prob))
        return torch.tensor(logits, dtype=torch.float32, device=x.device)
    
    def training_step(self, batch, batch_idx):
        x = batch['padded_features']  # [batch, seq_len, feature_dim]
        y = batch['label']            # [batch]
        flattened = x.view(x.size(0), -1).detach().cpu().numpy()  # flatten instead of pooling
        self.train_features.append(flattened)
        self.train_labels.append(y.detach().cpu().numpy())
        loss = torch.tensor(0.0, requires_grad=True, device=x.device)
        self.log("train_loss", loss)
        return loss
    
    def on_train_epoch_end(self):
        # Train XGBoost only once
        if self.xgb_model is None:
            X = np.concatenate(self.train_features, axis=0)
            y = np.concatenate(self.train_labels, axis=0)
            # Use all available cores by setting n_jobs=-1 for faster training and prediction
            self.xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', n_jobs=-1)
            self.xgb_model.fit(X, y)
    
    def validation_step(self, batch, batch_idx):
        x = batch['padded_features']
        y = batch['label'].to(x.device)
        flattened = x.view(x.size(0), -1).detach().cpu().numpy()  # flatten instead of pooling
        return {"flattened": flattened, "label": y.cpu().numpy()}
    
    def validation_epoch_end(self, outputs):
        # Aggregate flattened features and labels from all batches
        X_val = np.concatenate([out["flattened"] for out in outputs], axis=0)
        y_val = np.concatenate([out["label"] for out in outputs], axis=0).astype(np.float32)
        
        # Run XGBoost prediction on entire validation set once
        if self.xgb_model is not None:
            prob = self.xgb_model.predict_proba(X_val)[:, 1]
        else:
            prob = np.zeros(X_val.shape[0])
        prob = np.clip(prob, 1e-7, 1 - 1e-7)
        logits = np.log(prob / (1 - prob))
        # Compute loss and metrics in vectorized fashion
        logits_tensor = torch.tensor(logits, dtype=torch.float32)
        y_tensor = torch.tensor(y_val, dtype=torch.float32)
        pos_weight = torch.tensor([7.33])
        val_loss = F.binary_cross_entropy_with_logits(logits_tensor.view(-1), y_tensor.view(-1), pos_weight=pos_weight)
        
        probs_tensor = torch.sigmoid(logits_tensor)
        preds = (probs_tensor > 0.5).float()
        
        fraud_tp = ( (y_tensor == 1) & (preds == 1) ).sum().float()
        fraud_fp = ( (y_tensor == 0) & (preds == 1) ).sum().float()
        fraud_fn = ( (y_tensor == 1) & (preds == 0) ).sum().float()
        epsilon = 1e-7
        fraud_precision = fraud_tp / (fraud_tp + fraud_fp + epsilon)
        fraud_recall = fraud_tp / (fraud_tp + fraud_fn + epsilon)
        fraud_f1 = 2 * (fraud_precision * fraud_recall) / (fraud_precision + fraud_recall + epsilon)
        
        # Log aggregated metrics
        self.log("val_loss", val_loss, prog_bar=True)
        self.log("val_fraud_f1", fraud_f1, prog_bar=True)
        return {"val_loss": val_loss, "val_fraud_f1": fraud_f1}
    
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x = batch['padded_features']
        flattened = x.view(x.size(0), -1).detach().cpu().numpy()  # flatten instead of pooling
        return {"flattened": flattened}
    
    def predict_epoch_end(self, outputs):
        X_pred = np.concatenate([out["flattened"] for out in outputs], axis=0)
        if self.xgb_model is not None:
            prob = self.xgb_model.predict_proba(X_pred)[:, 1]
        else:
            prob = np.zeros(X_pred.shape[0])
        prob = np.clip(prob, 1e-7, 1 - 1e-7)
        logits = np.log(prob / (1 - prob))
        logits_tensor = torch.tensor(logits, dtype=torch.float32)
        probs_tensor = torch.sigmoid(logits_tensor)
        preds = (probs_tensor > 0.5).float()
        return {"probs": probs_tensor, "preds": preds}
    
    def configure_optimizers(self):
        # No optimizer needed for a non-gradient model
        return []