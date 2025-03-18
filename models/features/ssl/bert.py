import torch
import pytorch_lightning as pl
from transformers import BertForMaskedLM, BertConfig, get_linear_schedule_with_warmup

class BERTMaskedLM(pl.LightningModule):
    def __init__(self, 
                 vocab_size,
                 pretrained_model_name="bert-base-uncased",
                 finetune_mlp=True,
                 learning_rate=5e-5,
                 weight_decay=0.01,
                 warmup_steps=1000):
        """
        BERT model for Masked Language Modeling fine-tuning.
        """
        super().__init__()
        self.save_hyperparameters()
        
        # Load pretrained model
        self.model = BertForMaskedLM.from_pretrained(pretrained_model_name)
        # Resize token embeddings if vocab_size is different
        self.model.resize_token_embeddings(vocab_size)
            
        # Freeze specific parameters
        for name, param in self.model.named_parameters():
            # By default, freeze all parameters
            param.requires_grad = False
            
            # Unfreeze LayerNorm parameters
            if 'LayerNorm' in name:
                param.requires_grad = True
            # Unfreeze positional embeddings
            elif 'position_embeddings' in name:
                param.requires_grad = True
            # Conditionally unfreeze MLP/FFN layers
            elif ('intermediate' in name or 'output' in name) and finetune_mlp:
                param.requires_grad = True
    
    def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels=None):
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            labels=labels
        )
    
    def training_step(self, batch, batch_idx):
        outputs = self(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            token_type_ids=batch.get('token_type_ids', None),
            labels=batch['labels']
        )
        
        loss = outputs.loss
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        outputs = self(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            token_type_ids=batch.get('token_type_ids', None),
            labels=batch['labels']
        )
        
        loss = outputs.loss
        self.log("val_loss", loss, prog_bar=True, on_epoch=True)
        return loss
    
    def on_fit_start(self):
        """Print parameter statistics at the beginning of training."""
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.parameters())
        print(f"Trainable parameters: {trainable_params:,} ({trainable_params/total_params:.2%} of total)")
    
    def configure_optimizers(self):
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() 
                          if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters() 
                          if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        
        optimizer = torch.optim.AdamW(optimizer_grouped_parameters, 
                                    lr=self.hparams.learning_rate)
        
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