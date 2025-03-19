import os
import click
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Subset
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from sklearn.model_selection import train_test_split
import numpy as np

from bert import TransactionBERTModel
from dataloader import prepare_dataset, load_all


def prep_hpsearch_dataloaders(data_version, seed, batch_size, num_workers, load_fn=load_all):
    # Set seed for reproducibility
    pl.seed_everything(seed)
    
    # Prepare datasets
    print("Loading and preparing datasets...")
    dataset = prepare_dataset(data_version, load_fn)
    
    # Get all labels to create a stratified split
    all_labels = dataset.get_fraud_labels_idx()
    
    # Create stratified train/validation indices
    indices = np.arange(len(dataset))
    train_indices, val_indices = train_test_split(
        indices, 
        test_size=0.2,  # 20% for validation
        random_state=seed,
        stratify=all_labels  # This ensures the split preserves the class distribution
    )
    
    # Create Subset objects
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    
    print(f"Dataset split: {len(train_dataset)} training samples, {len(val_dataset)} validation samples")
    print(f"Stratified by fraudster class to maintain class distribution in both sets")
    
    # Get feature dimension from dataset
    feature_dim = dataset.feature_dim
    print(f"Feature dimension: {feature_dim}")
    
    # Create data loaders
    print("Creating data loaders...")
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        prefetch_factor=3
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        prefetch_factor=3
    )

    return train_loader, val_loader, feature_dim


@click.command()
# Data parameters
@click.option("--data_version", default="ver05", type=str, help="Data version to use")

# Model parameters
@click.option("--output_dir", default="./saved_models/transaction_bert", type=str, help="Output directory for models")

# Training parameters
# while a larger batch size like 48 would fit, it for some reason just takes ages in training. i'm assuming it fits in vram, but the optimizations for a smaller set are better
@click.option("--batch_size", default=24, type=int, help="Batch size for training")
@click.option("--num_train_epochs", default=15, type=int, help="Number of training epochs")
@click.option("--val_every_epoch", default=3, type=int, help="Number of training epochs after which to run validation")
@click.option("--gradient_accumulation_steps", default=1, type=int, help="Gradient accumulation steps")
@click.option("--fp16", is_flag=True, help="Use 16-bit precision")

@click.option("--out_name", default="final-model.ckpt", type=str, help="Name of the final model file")
@click.option("--continue_from_checkpoint", default=None, type=str, help="Path to a checkpoint to continue training")
@click.option("--fine_tune_ffn", is_flag=True, default=False, help="Fine-tune the feed-forward network")

# Other parameters
@click.option("--seed", default=42, type=int, help="Random seed")
@click.option("--num_workers", default=4, type=int, help="Number of workers for data loading")
@click.option("--patience", default=3, type=int, help="Early stopping patience")
@click.option("--dry_run", is_flag=True, help="Perform a dry run (setup but no training)")
def main(data_version, output_dir, batch_size, num_train_epochs, val_every_epoch,
         gradient_accumulation_steps, fp16, out_name, continue_from_checkpoint, fine_tune_ffn, seed, num_workers, patience, dry_run):
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # Set seed for reproducibility
    pl.seed_everything(seed)
    
    train_loader, val_loader, feature_dim = prep_hpsearch_dataloaders(data_version, seed, batch_size, num_workers)
    
    # Initialize model
    print(f"Initializing Transaction BERT model with Hugging Face BERT")
    model = TransactionBERTModel(
        feature_dim=feature_dim,
        fine_tune_ffn=fine_tune_ffn
    )
    
    # Apply torch.compile if available (PyTorch 2.0+)
    # This is the preferred way to compile models in Lightning
    if hasattr(torch, 'compile'):
        try:
            print("Compiling model with torch.compile()...")
            model = torch.compile(model, mode="reduce-overhead")
            print("Model compiled successfully")
        except Exception as e:
            print(f"Model compilation failed: {e}. Training with uncompiled model.")
    
    # Setup checkpointing
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath=output_dir,
        filename='transaction-bert-{epoch:02d}-{val_loss:.4f}',
        save_top_k=3,
        mode='min',
        save_weights_only=True
    )
    
    # Early stopping
    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        patience=patience,
        mode='min'
    )
    
    # Setup tensorboard logger
    logger = TensorBoardLogger(
        save_dir=os.path.join(output_dir, 'logs'),
        name='transaction-bert'
    )
    
    # Initialize trainer
    trainer = pl.Trainer(
        max_epochs=num_train_epochs,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1,
        callbacks=[checkpoint_callback, early_stop_callback],
        logger=logger,
        precision='16-mixed',
        accumulate_grad_batches=gradient_accumulation_steps,
        fast_dev_run=dry_run,
        check_val_every_n_epoch=val_every_epoch,
        benchmark=True,  # Optimize CUDA operations
    )
    
    # Train the model
    print("Starting training...")
    torch.set_float32_matmul_precision('high')
    trainer.fit(model, train_loader, val_loader, **({'ckpt_path': continue_from_checkpoint} if continue_from_checkpoint else {}))
    
    # Save the final model using Lightning instead of just the state dict
    final_model_path = os.path.join(output_dir, out_name)
    if os.path.exists(final_model_path):
        os.remove(final_model_path)
    trainer.save_checkpoint(final_model_path)
    
    print(f"Training completed. Final model saved to {final_model_path}")
    

if __name__ == "__main__":
    # example usage (run from project dir):  
    # python3 train_bert_mlm.py 
    # python3 train_bert_mlm.py --dry_run
    # python3 train_bert_mlm.py --num_train_epochs=5
    # python3 train_bert_mlm.py --continue_from_checkpoint=./saved_models/transaction_bert/transaction-bert-epoch=09-val_loss=0.0419.ckpt
    main()