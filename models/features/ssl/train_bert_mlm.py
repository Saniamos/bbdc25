import os
import click
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

from bert import TransactionBERTModel
from dataloader import prepare_dataset, load_train_test, load_val


@click.command()
# Data parameters
@click.option("--data_version", default="ver05", type=str, help="Data version to use")

# Model parameters
@click.option("--output_dir", default="./saved_models/transaction_bert", type=str, help="Output directory for models")

# Training parameters
@click.option("--batch_size", default=24, type=int, help="Batch size for training")
@click.option("--num_train_epochs", default=15, type=int, help="Number of training epochs")
@click.option("--gradient_accumulation_steps", default=1, type=int, help="Gradient accumulation steps")
@click.option("--fp16", is_flag=True, help="Use 16-bit precision")

@click.option("--out_name", default="final-model.pt", type=str, help="Name of the final model file")
@click.option("--continue_from_checkpoint", default=None, type=str, help="Path to a checkpoint to continue training")
@click.option("--fine_tune_ffn", is_flag=True, default=False, help="Fine-tune the feed-forward network")

# Other parameters
@click.option("--seed", default=42, type=int, help="Random seed")
@click.option("--num_workers", default=5, type=int, help="Number of workers for data loading")
@click.option("--patience", default=3, type=int, help="Early stopping patience")
@click.option("--dry_run", is_flag=True, help="Perform a dry run (setup but no training)")

def main(data_version, output_dir, batch_size, num_train_epochs,
         gradient_accumulation_steps, fp16, out_name, continue_from_checkpoint, fine_tune_ffn, seed, num_workers, patience, dry_run):
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Set seed for reproducibility
    pl.seed_everything(seed)
    
    # Prepare datasets
    print("Loading and preparing datasets...")
    # train_dataset, val_dataset = prepare_datasets(data_version)
    # train_dataset, val_dataset, test_dataset = prepare_datasets(data_version)
    train_dataset = prepare_dataset(data_version, load_train_test)
    val_dataset = prepare_dataset(data_version, load_val)
    
    # Get feature dimension from dataset
    feature_dim = train_dataset.feature_dim
    print(f"Feature dimension: {feature_dim}")
    
    # Create data loaders
    print("Creating data loaders...")
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    # Initialize model
    print(f"Initializing Transaction BERT model with Hugging Face BERT")
    model = TransactionBERTModel(
        feature_dim=feature_dim,
        fine_tune_ffn=fine_tune_ffn
    )
    
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
        check_val_every_n_epoch=5,
        benchmark=True,  # Optimize CUDA operations
    )
    
    # Train the model
    print("Starting training...")
    torch.set_float32_matmul_precision('high')
    trainer.fit(model, train_loader, val_loader, **({'ckpt_path': continue_from_checkpoint} if continue_from_checkpoint else {}))
    
    # Save the final model
    final_model_path = os.path.join(output_dir, out_name)
    if os.path.exists(final_model_path):
        os.remove(final_model_path)
    torch.save(model.state_dict(), final_model_path)
    
    print(f"Training completed. Final model saved to {final_model_path}")
    

if __name__ == "__main__":
    # example usage (run from project dir):  
    # python3 train_bert_mlm.py 
    # python3 train_bert_mlm.py --dry_run
    # python3 train_bert_mlm.py --num_train_epochs=5
    # python3 train_bert_mlm.py --continue_from_checkpoint=./saved_models/transaction_bert/transaction-bert-epoch=09-val_loss=0.0419.ckpt
    main()